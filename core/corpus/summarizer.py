from __future__ import annotations
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = (
    "Summarize the following text in 1-2 sentences capturing tone and key message.\n\n"
    "Text:\n{chunk}\n\nSummary:"
)

class GistBuilder:
    """
    Uses Gemma (same model) for cheap 1–2 sentence gists. Falls back to extractive if no GPU.
    """
    def __init__(self, model_ref: str = "google/gemma-2b-it", device: str = "auto", max_new_tokens: int = 64):
        self.model_ref = model_ref
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
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
            self._tok = AutoTokenizer.from_pretrained(self.model_ref)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_ref).to(self.device)

    def gist(self, chunk: str, max_sentences: int = 2) -> str:
        try:
            self._lazy_load()
            prompt = PROMPT_TEMPLATE.format(chunk=chunk.strip()[:2000])
            inputs = self._tok(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.2,
                    do_sample=False,
                    eos_token_id=self._tok.eos_token_id,
                )
            text = self._tok.decode(out[0], skip_special_tokens=True)
            summary = text.split("Summary:")[-1].strip()
            # keep to ~2 sentences
            parts = summary.split(". ")
            return ". ".join(parts[:max_sentences]).strip()
        except Exception:
            # fallback extractive: first ~2 sentences
            sents = chunk.strip().split(". ")
            return ". ".join(sents[:max_sentences]).strip()
