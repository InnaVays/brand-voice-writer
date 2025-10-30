# model/gemma_peft.py
from __future__ import annotations
import os
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PeftModel,
    PeftConfig,
)

from model.base import TextModel, GenConfig, GenResult


DEFAULT_MODEL_ID = "google/gemma-2b-it"


def load_gemma_base(model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    mdl.to(device)
    return tok, mdl, device


def attach_prompt_tuning(
    model,
    num_virtual_tokens: int = 40,
    init_text: str = "Our brand voice is warm, confident, concise.",
):
    cfg = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=init_text,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def save_prompt_tuning(peft_model: PeftModel, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)


def load_prompt_tuning(base_model, adapter_dir: str) -> PeftModel:
    return PeftModel.from_pretrained(base_model, adapter_dir)


class GemmaPeftGenerator(TextModel):
    def __init__(
        self,
        tokenizer,
        model,
        device: str,
        system_preamble: str = "You rewrite texts in the user's own brand voice. Keep facts. Improve clarity.",
    ):
        self.tok = tokenizer
        self.mdl = model
        self.device = device
        self.system_preamble = system_preamble

    def _to_gen_cfg(self, cfg: Optional[GenConfig]) -> GenerationConfig:
        if cfg is None:
            cfg = GenConfig()
        return GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        )

    def generate(self, prompt: str, cfg: Optional[GenConfig] = None) -> GenResult:
        gcfg = self._to_gen_cfg(cfg)
        full = f"{self.system_preamble}\n\n{prompt}"
        inputs = self.tok(full, return_tensors="pt").to(self.mdl.device)
        with torch.no_grad():
            out_ids = self.mdl.generate(**inputs, generation_config=gcfg)
        text = self.tok.decode(out_ids[0], skip_special_tokens=True)
        # strip preamble
        if text.startswith(self.system_preamble):
            text = text[len(self.system_preamble):].lstrip()
        return GenResult(text=text, meta={"tokens": int(out_ids.numel())})

    def generate_n(self, prompt: str, n: int, cfg: Optional[GenConfig] = None) -> List[GenResult]:
        return [self.generate(prompt, cfg) for _ in range(max(1, n))]


def extract_softprompt_matrix(peft_model: PeftModel):
    # PEFT prompt encoder weights
    enc = peft_model.base_model.prompt_encoder
    w = enc.weight.detach().float().cpu().numpy()  # (num_virtual_tokens, hidden_dim)
    return w
