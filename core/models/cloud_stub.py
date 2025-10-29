# core/models/providers/cloud_stub.py
from __future__ import annotations
from typing import Optional
from core.models.base import TextModel, GenerationConfig, GenerationResult


class CloudStubModel(TextModel):
    """
    Placeholder for a cloud LLM provider (OpenAI, etc.).
    This public repo ships a stub that raises a clear error.
    Swap in a real implementation in Pro or in your own private build.
    """
    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> GenerationResult:
        raise RuntimeError(
            "CloudStubModel is not available in the public edition. "
            "Use a local provider or install a private/cloud plugin."
        )
