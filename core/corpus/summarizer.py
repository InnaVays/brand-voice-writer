from __future__ import annotations
from typing import Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = (
"Summarize the following text in a few neutral, factual sentences. "
"Avoid marketing language, emojis, and hype.\n\n"
"Text:\n{chunk}\n\nSummary:"
)

class SmallSummarizer:
    """
    1B Instruct model for neutral 1-2 sentence gists.
    Downloads on first use into summarizer.cache_dir. Reuses later.
    """
    def __init__(self, model_id: str, cache_dir: str = ".hf_cache", device: str = "auto", max_new_tokens: int = 60, temperature: float = 0.0):
        self.model_id = model_id
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._tok = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, use_fast=True, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=(torch.float16 if self.device in ("cuda","mps") else torch.float32),
        ).to(self.device)
        self._model.eval()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available(): return "cuda"
            try:
                if torch.backends.mps.is_available(): return "mps"
            except Exception:
                pass
            return "cpu"
        return device

    def _lazy_load(self):
        if self._tok is None or self._model is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir, use_fast=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)
            self._model.eval()

    def summarize(self, chunk: str) -> str:
        self._lazy_load()
        text = chunk.strip()[:]

        # 1) Chat-format (предпочтительно для Qwen / TinyLlama / Llama)
        try:
            messages = [
                {"role": "system", "content": "You write neutral, a few factual sentence summaries."},
                {"role": "user", "content": f"Summarize the following text in a few neutral, factual sentences without hype or emojis:\n\n{text}"}
            ]
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # fallback: обычный prompt
            prompt = (
                "Summarize the following text in a few neutral, factual sentences. "
                "Avoid marketing language, emojis, and hype.\n\n"
                f"Text:\n{text}\n\nSummary:"
            )

        inputs = self._tok(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                eos_token_id=getattr(self._tok, "eos_token_id", None),
                pad_token_id=getattr(self._tok, "pad_token_id", getattr(self._tok, "eos_token_id", None)),
                return_dict_in_generate=True,
            )

        seq = out.sequences[0]
        in_len = inputs["input_ids"].shape[1]
        new_tokens = seq[in_len:]
        s = self._tok.decode(new_tokens, skip_special_tokens=True).strip()

        for lead in ("Assistant:", "assistant:", "Summary:", "summary:"):
            if s.startswith(lead):
                s = s[len(lead):].lstrip()

        s = s.replace("\n", " ")

        return s

