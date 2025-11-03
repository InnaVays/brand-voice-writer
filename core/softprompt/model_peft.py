from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

from peft import PromptTuningConfig, get_peft_model, TaskType

@dataclass
class TrainConfig:
    virtual_tokens: int = 16
    lr: float = 5e-3
    epochs: int = 2
    batch_size: int = 1
    max_seq_len: int = 80
    style_id: str = "default"
    grad_accum: int = 1
    gradient_checkpointing: bool = False


class SoftPromptTrainer:
    def __init__(self, model, tokenizer, cfg: TrainConfig, device: str = "cpu"):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device

        self.peft_model = self._wrap_peft()
        self.peft_model.to(self.device)

        if self.cfg.gradient_checkpointing:
            try:
                self.peft_model.gradient_checkpointing_enable()
            except Exception:
                pass


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
    def _prep_dataset(self, dataset_jsonl: str):
        """
        Expect JSONL with fields:
        - instruction 
        - input       
        - output     

        For CausalLM :
        input_ids  = [prompt_tokens + target_tokens]
        labels     = [-100 ... -100, target_tokens...]
        """
        tok = self.tok
        max_len = getattr(self.cfg, "max_seq_len", 512)  

        ds = load_dataset("json", data_files=dataset_jsonl, split="train")
        orig_cols = ds.column_names

        def map_fn(ex):
            instruction = ex.get("instruction", "") or ex.get("system", "") or ""
            inp         = ex.get("input", "")       or ex.get("source", "") or ""
            target      = ex.get("output", "")      or ex.get("target", "") or ex.get("response", "")

            if not inp:
                raise ValueError(f"No 'input'/'source' field in example: {ex}")
            if not target:
                raise ValueError(f"No 'output'/'target'/'response' field in example: {ex}")

            # prompt = instruction + input
            if instruction:
                prompt_text = instruction.strip() + "\n\n" + inp.strip()
            else:
                prompt_text = inp.strip()

            prompt_ids = tok(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )["input_ids"]

            target_ids = tok(
                target.strip(),
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )["input_ids"]

            full_ids = prompt_ids + target_ids
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100 

            pad_id = tok.pad_token_id
            if pad_id is None:
                pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

            cur_len = len(full_ids)
            if cur_len < max_len:
                pad_len = max_len - cur_len
                full_ids = full_ids + [pad_id] * pad_len
                labels   = labels   + [-100]   * pad_len
                attn_mask = [1] * cur_len + [0] * pad_len
            else:
                attn_mask = [1] * max_len

            return {
                "input_ids": full_ids,
                "attention_mask": attn_mask,
                "labels": labels,
            }

        ds = ds.map(
            map_fn,
            remove_columns=orig_cols,
            num_proc=None,  
            desc=f"Tokenizing dataset {dataset_jsonl}",
        )

        return ds

    # ---------- prompt embedding accessor ----------
    @staticmethod
    def _find_prompt_embedding_module(peft_model) -> torch.nn.Module:

        pe = getattr(peft_model, "prompt_encoder", None)
        if pe is None:
            raise RuntimeError("PEFT model has no attribute 'prompt_encoder'")

        emb = getattr(pe, "embedding", None)
        if isinstance(emb, torch.nn.Embedding):
            return emb

        if isinstance(pe, torch.nn.ModuleDict):
            for name, sub in pe.items():
                sub_emb = getattr(sub, "embedding", None)
                if isinstance(sub_emb, torch.nn.Embedding):
                    return sub_emb
                if isinstance(sub, torch.nn.Embedding):
                    return sub
                for child in sub.modules():
                    if isinstance(child, torch.nn.Embedding):
                        return child

        for m in peft_model.modules():
            if isinstance(m, torch.nn.Embedding) and "prompt" in m.__class__.__name__.lower():
                return m

        for m in (list(pe.modules()) if hasattr(pe, "modules") else []):
            if isinstance(m, torch.nn.Embedding):
                return m
        for m in peft_model.modules():
            if isinstance(m, torch.nn.Embedding):
                return m

        raise RuntimeError("Could not locate prompt embedding module inside PEFT model")

    def train(self, dataset_jsonl: str, out_base_dir: str):
        """
        out_base_dir = artifacts/softprompt/<style>/run_XXX
        """
        t0 = time.time()

        run_dir = Path(out_base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        ds = self._prep_dataset(dataset_jsonl)

        args = TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=self.cfg.batch_size,
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.lr,
            logging_steps=10,
            save_strategy="no",           
            report_to=[],                 
            gradient_accumulation_steps=self.cfg.grad_accum,
        )

        trainer = Trainer(
            model=self.peft_model,
            args=args,
            train_dataset=ds,
            tokenizer=self.tok,
        )

        trainer.train()

        self.peft_model.save_pretrained(str(run_dir))
        self.tok.save_pretrained(str(run_dir))

        seconds = time.time() - t0
        return {"seconds": seconds}