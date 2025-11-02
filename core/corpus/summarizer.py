from __future__ import annotations
from typing import Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = (
"Summarize the following text in 2 neutral, factual sentences. "
"Avoid marketing language, emojis, and hype.\n\n"
"Text:\n{chunk}\n\nSummary:"
)

class LlamaSmallSummarizer:
    """
    1B Instruct model for neutral 1–2 sentence gists.
    Downloads on first use into summarizer.cache_dir. Reuses later.
    """
    def __init__(self, model_id: str, cache_dir: str = ".hf_cache", device: str = "auto", max_new_tokens: int = 96, temperature: float = 0.0):
        self.model_id = model_id
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tok: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

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

    def summarize(self, chunk: str, max_sentences: int = 2) -> str:
        self._lazy_load()
        prompt = PROMPT.format(chunk=chunk.strip()[:4000])
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                eos_token_id=self._tok.eos_token_id
            )
        text = self._tok.decode(out[0], skip_special_tokens=True)
        s = text.split("Summary:")[-1].strip()
        # trim to requested sentences (simple split)
        parts = [p.strip() for p in s.replace("\n", " ").split(". ") if p.strip()]
        s2 = ". ".join(parts[:max_sentences]).strip()
        if s2 and s2[-1] not in ".!?…": s2 += "."
        return s2
