from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, str] | None = None


@dataclass
class BrandPack:
    mu: list[float]                    
    sigma: list[list[float]] | None    
    embed_dim: int                      
    softprompt_path: Optional[str] = None  
    index_path: Optional[str] = None      


@dataclass
class RunConfig:
    candidates: int = 3
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
