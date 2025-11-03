from __future__ import annotations
import os, json, math, time
from dataclasses import dataclass, asdict
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, TaskType, PeftModel

@dataclass
class TrainConfig:
    virtual_tokens: int = 40
    lr: float = 5e-3
    epochs: int = 3
    batch_size: int = 8
    max_seq_len: int = 512
    style_id: str = "default"

class SoftPromptTrainer:
    def __init__(self, model, tokenizer, cfg: TrainConfig, device: str):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device

    def _wrap_peft(self):
        init_text = "Our brand voice is warm, precise, and trustworthy."
        pt_cfg = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.cfg.virtual_tokens,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=init_text,
            tokenizer_name_or_path=self.tok.name_or_path,
        )
        peft_model = get_peft_model(self.model, pt_cfg)
        peft_model.print_trainable_parameters()
        return peft_model

    def _prep_dataset(self, jsonl_path: str):
        ds = load_dataset("json", data_files=jsonl_path, split="train")
        instr_tmpl = "Instruction: {instruction}\n\nInput:\n{input}\n\nOutput:\n"
        def tok_map(ex):
            prompt = instr_tmpl.format(**ex)
            ids = self.tok(
                prompt + ex["output"],
                max_length=self.cfg.max_seq_len,
                truncation=True,
                return_tensors=None,
            )
            return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"]}
        return ds.map(tok_map, remove_columns=ds.column_names, num_proc=1)

    def train(self, dataset_jsonl: str, out_base_dir: str) -> Dict[str, Any]:
        style_dir = os.path.join(out_base_dir, self.cfg.style_id)
        os.makedirs(style_dir, exist_ok=True)

        peft_model = self._wrap_peft()
        peft_model.train()

        ds = self._prep_dataset(dataset_jsonl)
        collator = DataCollatorForLanguageModeling(self.tok, mlm=False)

        args = TrainingArguments(
            output_dir=style_dir,
            per_device_train_batch_size=self.cfg.batch_size,
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.lr,
            logging_steps=10,
            save_steps=200,
            save_total_limit=1,
            bf16=(self.device == "cuda"),
            fp16=(self.device == "cuda"),
            report_to=[],
        )

        trainer = Trainer(
            model=peft_model,
            args=args,
            data_collator=collator,
            train_dataset=ds,
        )

        t0 = time.time()
        trainer.train()
        secs = time.time() - t0

        # Save PEFT adapter (soft prompt)
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
