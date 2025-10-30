# model/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict, Any


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class GenResult:
    text: str
    meta: Dict[str, Any]


class TextModel(Protocol):
    def generate(self, prompt: str, cfg: Optional[GenConfig] = None) -> GenResult: ...
    def generate_n(self, prompt: str, n: int, cfg: Optional[GenConfig] = None) -> List[GenResult]: ...
