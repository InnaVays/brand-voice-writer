# core/style/space.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np
from core.style.embed import style_embedding, semantic_embedding, cosine_sim


@dataclass
class BrandStats:
    mu: List[float]                  # centroid of style embeddings
    Sigma: List[List[float]]         # covariance matrix
    dim: int = 256
    n_samples: int = 0


class BrandStyleSpace:
    """
    Keeps brand style distribution (centroid and covariance).
    Provides distance-to-brand and semantics scoring helpers.
    """
    def __init__(self, stats: Optional[BrandStats] = None):
        self.stats = stats

    @staticmethod
    def fit(texts: List[str], dim: int = 256, eps: float = 1e-3) -> "BrandStyleSpace":
        if not texts:
            raise ValueError("No texts provided to fit BrandStyleSpace.")
        Z = np.stack([style_embedding(t, dim=dim) for t in texts], axis=0)  # (N, D)
        mu = Z.mean(axis=0)
        # covariance regularization
        X = Z - mu
        Sigma = (X.T @ X) / max(1, Z.shape[0] - 1)
        Sigma += eps * np.eye(dim, dtype=np.float32)
        stats = BrandStats(mu=mu.tolist(), Sigma=Sigma.tolist(), dim=dim, n_samples=len(texts))
        return BrandStyleSpace(stats=stats)

    def to_json(self) -> str:
        return json.dumps(asdict(self.stats))

    @staticmethod
    def from_json(s: str) -> "BrandStyleSpace":
        d = json.loads(s)
        stats = BrandStats(**d)
        return BrandStyleSpace(stats=stats)

    # --- scoring helpers ---
    def semantic_similarity(self, a: str, b: str) -> float:
        dim = self.stats.dim
        ea = semantic_embedding(a, dim=dim)
        eb = semantic_embedding(b, dim=dim)
        return cosine_sim(ea, eb)

    def score(self, generated: str, source: Optional[str] = None, alpha_semantic: float = 0.4) -> Dict[str, Any]:
        d_style = self.style_distance(generated)
        s_sem = 0.0
        if source is not None:
            s_sem = self.semantic_similarity(generated, source)
        # Invert distance (smaller is better); normalize softly
        score = -d_style + alpha_semantic * s_sem
        return {
            "style_distance": float(d_style),
            "semantic_similarity": float(s_sem),
            "score": float(score)
        }
