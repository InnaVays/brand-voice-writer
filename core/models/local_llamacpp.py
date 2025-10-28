# core/models/providers/local_llamacpp.py
from __future__ import annotations
import os
import random
import re
from typing import Optional, Dict, Any
from core.models.base import TextModel, GenerationConfig, GenerationResult


class LocalModel(TextModel):
    """
    Minimal, dependency-free 'local model' for the public repo.
    It DOES NOT use an LLM. It applies light prompt-aware transformations to simulate tone/style.
    Replace with a real provider (e.g., llama-cpp-python) in Pro or for your own builds.
    """

    def __init__(self, name: str = "simple-local"):
        self.name = name

    def _sample_sentence_endings(self, temperature: float) -> str:
        endings = [
            ".", ".", ".", "!", "…", "!"
        ]
        # temperature loosely affects exclamation probability
        k = 1 if temperature <= 0.3 else (2 if temperature <= 0.8 else 3)
        return "".join(random.sample(endings, k=1))

    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> GenerationResult:
        if cfg is None:
            cfg = GenerationConfig()
        if cfg.seed is not None:
            random.seed(cfg.seed)

        content_match = re.search(r"\[CONTENT\](.*?)\[/CONTENT\]", prompt, flags=re.DOTALL)
        content = content_match.group(1).strip() if content_match else prompt

        # Clip to max_tokens (approximate by words)
        words = content.split()
        if len(words) > cfg.max_tokens:
            words = words[:cfg.max_tokens]
        text = " ".join(words)

        # Very light temperature effect: add punctuation sample
        text = text.rstrip() + self._sample_sentence_endings(cfg.temperature)

        return GenerationResult(
            text=text,
            meta={
                "model": self.name,
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p
            }
        )

def load_local_model() -> TextModel:
    """
    Public default loader. In Pro you’ll swap this to llama-cpp or other backends.
    Supports env toggle later if you like.
    """
    _ = os.environ.get("BVR_LOCAL_MODEL", "simple")
    
    return SimpleLocalModel()
