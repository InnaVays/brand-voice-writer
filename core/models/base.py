from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    num_candidates: int = 1
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    text: str
    meta: Dict[str, Any]

class TextModel(Protocol):
    """
    Abstract model interface.
    """
    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> GenerationResult:
        pass

    def generate_n(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> List[GenerationResult]:
        """
        Convenience: produce multiple candidates.
        """
        if cfg is None:
            cfg = GenerationConfig()
        results = []
        for i in range(max(1, cfg.num_candidates)):
            local_cfg = GenerationConfig(
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                num_candidates=1,
                seed=None if cfg.seed is None else (cfg.seed + i)
            )
            results.append(self.generate(prompt, cfg=local_cfg))
        return results
