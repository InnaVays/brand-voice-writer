# core/pipeline/eval/scorer.py
from __future__ import annotations
from typing import Optional, Dict, Any
from core.style.space import BrandStyleSpace

def evaluate_candidate(
    brand: BrandStyleSpace,
    generated: str,
    source: Optional[str] = None,
    alpha_semantic: float = 0.4
) -> Dict[str, Any]:
    """
    Returns a dict with style_distance, semantic_similarity (if source provided) and aggregate score.
    """
    return brand.score(generated, source=source, alpha_semantic=alpha_semantic)
