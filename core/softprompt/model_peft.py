from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import Adafactor

from peft import PromptTuningConfig, get_peft_model, TaskType


@dataclass
class TrainConfig:
    virtual_tokens: int = 16
    lr: float = 5e-3
    epochs: int = 2
    batch_size: int = 1
    max_seq_len: int = 80
    style_id: str = "default"


class SoftPromptTrainer:
    def __init__(self, model, tokenizer, cfg: TrainConfig, device: str = "cpu"):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device

    # ---------- PEFT wrapper ----------
    def _wrap_peft(self):
        task_type = TaskType.SEQ_2_SEQ_LM if getattr(self.model.config, "is_encoder_decoder", False) else TaskType.CAUSAL_LM
        pt_cfg = PromptTuningConfig(
            task_type=task_type,
            num_virtual_tokens=self.cfg.virtual_tokens,
            prompt_tuning_init="RANDOM",
            tokenizer_name_or_path=self.tok.name_or_path,
        )
        peft_model = get_peft_model(self.model, pt_cfg)
        try:
            peft_model.print_trainable_parameters()
        except Exception:
            pass
        return peft_model

    # ---------- dataset prep ----------
    def _prep_dataset(self, jsonl_path: str):
        ds = load_dataset("json", data_files=jsonl_path, split="train")

        if getattr(self.model.config, "is_encoder_decoder", False):
            # Seq2Seq (например, FLAN-T5)
            def map_fn(ex):
                instr = (ex.get("instruction") or "")[:256]
                inp   = (ex.get("input") or "")[:256]
                tgt   = (ex.get("output") or "")[:512]
                src = f"{instr}\n{inp}".strip()
                enc = self.tok(
                    src, max_length=self.cfg.max_seq_len, truncation=True,
                    padding=False, add_special_tokens=True
                )
                lab = self.tok(
                    tgt, max_length=self.cfg.max_seq_len, truncation=True,
                    padding=False, add_special_tokens=True
                )
                labels = lab["input_ids"]
                pad_id = self.tok.pad_token_id
                if pad_id is not None:
                    labels = [(-100 if t == pad_id else t) for t in labels]
                return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
        else:
            # Causal LM
            instr_tmpl = "{instruction}\n{input}\n###\n"

            def map_fn(ex):
                instr = (ex.get("instruction") or "")[:256]
                inp   = (ex.get("input") or "")[:256]
                out   = (ex.get("output") or "")
                prompt = instr_tmpl.format(instruction=instr, input=inp)

                enc_p = self.tok(prompt, max_length=self.cfg.max_seq_len, truncation=True, add_special_tokens=False)
                enc_o = self.tok(out,    max_length=self.cfg.max_seq_len, truncation=True, add_special_tokens=False)

                input_ids = enc_p["input_ids"] + enc_o["input_ids"]
                attn_mask = enc_p["attention_mask"] + enc_o["attention_mask"]
                labels    = [-100]*len(enc_p["input_ids"]) + enc_o["input_ids"]

                if len(input_ids) > self.cfg.max_seq_len:
                    input_ids = input_ids[:self.cfg.max_seq_len]
                    attn_mask = attn_mask[:self.cfg.max_seq_len]
                    labels    = labels[:self.cfg.max_seq_len]

                return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

        return ds.map(map_fn, remove_columns=ds.column_names, num_proc=1)

    # ---------- prompt embedding accessor ----------
    @staticmethod
    def _find_prompt_embedding_module(peft_model) -> torch.nn.Module:

        pe = getattr(peft_model, "prompt_encoder", None)
        if pe is None:
            raise RuntimeError("PEFT model has no attribute 'prompt_encoder'")

        # если у него напрямую есть embedding
        emb = getattr(pe, "embedding", None)
        if isinstance(emb, torch.nn.Embedding):
            return emb

        if isinstance(pe, torch.nn.ModuleDict):
            for name, sub in pe.items():
                # у подмодуля может быть .embedding
                sub_emb = getattr(sub, "embedding", None)
                if isinstance(sub_emb, torch.nn.Embedding):
                    return sub_emb
                # или сам подмодуль — Embedding
                if isinstance(sub, torch.nn.Embedding):
                    return sub
                # попробуем поискать по параметрам внутри sub
                for child in sub.modules():
                    if isinstance(child, torch.nn.Embedding):
                        return child

        # 3) Полный перебор всего дерева модулей в PEFT-модели
        for m in peft_model.modules():
            if isinstance(m, torch.nn.Embedding) and "prompt" in m.__class__.__name__.lower():
                return m

        # 4) Последний шанс: первый Embedding в prompt_encoder (или вообще любой Embedding)
        for m in (list(pe.modules()) if hasattr(pe, "modules") else []):
            if isinstance(m, torch.nn.Embedding):
                return m
        for m in peft_model.modules():
            if isinstance(m, torch.nn.Embedding):
                return m

        raise RuntimeError("Could not locate prompt embedding module inside PEFT model")

    # ---------- train ----------
    def train(self, dataset_jsonl: str, out_base_dir: str) -> Dict[str, Any]:
        # экономия памяти на CPU
        self.model.config.use_cache = False
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass

        style_dir = os.path.join(out_base_dir, self.cfg.style_id)
        os.makedirs(style_dir, exist_ok=True)

        peft_model = self._wrap_peft()
        peft_model.train()

        emb_module = self._find_prompt_embedding_module(peft_model)
        with torch.no_grad():
            W_init = emb_module.weight.detach().cpu().clone()

        try:
            import numpy as np
            np.save(os.path.join(style_dir, "softprompt_init.npy"), W_init.numpy())
            with open(os.path.join(style_dir, "softprompt_init.tsv"), "w", encoding="utf-8") as f:
                for row in W_init.numpy():
                    f.write("\t".join(map(str, row.tolist())) + "\n")
        except Exception:
            pass

        ds = self._prep_dataset(dataset_jsonl)

        args = TrainingArguments(
            output_dir=style_dir,
            per_device_train_batch_size=1,
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.lr,
            logging_steps=200,
            save_steps=10_000_000,     
            save_total_limit=1,
            fp16=False,
            bf16=False,
            report_to=[],
            dataloader_num_workers=0,
        )

        # 1) Adafactor с фиксированным lr — стабильно и без None
        optimizer = Adafactor(
            peft_model.parameters(),
            lr=self.cfg.lr,          # например 5e-3
            scale_parameter=True,
            relative_step=False,     # <- критично: не ставим True
            warmup_init=False,
        )

        # 2) Заглушка-шедулер: всегда 1.0, чтобы Trainer не создавал свой
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        trainer = Trainer(
            model=peft_model,
            args=args,
            train_dataset=ds,
            data_collator=None,
            optimizers=(optimizer, scheduler),   # <- передаём ОТДЕЛЬНО шедулер
        )

        t0 = time.time()
        trainer.train()
        secs = time.time() - t0

        with torch.no_grad():
            W_trained = emb_module.weight.detach().cpu().clone()

        try:
            import numpy as np
            np.save(os.path.join(style_dir, "softprompt_trained.npy"), W_trained.numpy())
            with open(os.path.join(style_dir, "softprompt_trained.tsv"), "w", encoding="utf-8") as f:
                for row in W_trained.numpy():
                    f.write("\t".join(map(str, row.tolist())) + "\n")

            D = W_trained - W_init
            per_tok_l2 = torch.norm(D, dim=1).tolist()
            total_l2 = float(torch.norm(D))
            cos = torch.nn.functional.cosine_similarity(W_init, W_trained, dim=1).tolist()

            with open(os.path.join(style_dir, "delta_stats.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "virtual_tokens": self.cfg.virtual_tokens,
                    "embed_dim": int(W_init.shape[1]),
                    "per_token_L2": per_tok_l2,
                    "total_L2": total_l2,
                    "per_token_cosine": cos,
                    "seconds": secs,
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "lr": self.cfg.lr,
                    "max_seq_len": self.cfg.max_seq_len
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        peft_model.save_pretrained(style_dir)

        with open(os.path.join(style_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "style_id": self.cfg.style_id,
                "seconds": secs,
                "virtual_tokens": self.cfg.virtual_tokens,
                "epochs": self.cfg.epochs,
                "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr,
                "max_seq_len": self.cfg.max_seq_len
            }, f, ensure_ascii=False, indent=2)

        return {"style_dir": style_dir, "seconds": secs}
