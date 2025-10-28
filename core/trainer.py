# core/softprompt/trainer.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
from core.style.embed import style_embedding


@dataclass
class SoftPromptWeights:
    tokens: int
    dim: int
    matrix: List[List[float]]  # shape: (tokens, dim)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "SoftPromptWeights":
        d = json.loads(s)
        return SoftPromptWeights(**d)


class SoftPromptTrainer:
    """
    Lightweight, dependency-free soft prompt 'training'.
    We approximate a brand soft prompt by stacking small perturbations around the brand centroid.
    This is a placeholder you can replace with real p-tuning later.
    """

    def __init__(self, tokens: int = 40, dim: int = 256, seed: Optional[int] = 42):
        self.tokens = tokens
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def fit(self, corpus_texts: List[str]) -> SoftPromptWeights:
        if not corpus_texts:
            raise ValueError("SoftPromptTrainer: empty corpus")
        Z = np.stack([style_embedding(t, dim=self.dim) for t in corpus_texts], axis=0)
        mu = Z.mean(axis=0)  # centroid
        # Create token vectors around centroid with small random noise
        matrix = []
        for _ in range(self.tokens):
            noise = self.rng.normal(loc=0.0, scale=0.02, size=self.dim).astype(np.float32)
            v = mu + noise
            v = v / (np.linalg.norm(v) + 1e-8)
            matrix.append(v.tolist())
        return SoftPromptWeights(tokens=self.tokens, dim=self.dim, matrix=matrix)

    @staticmethod
    def save(weights: SoftPromptWeights, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(weights.to_json())

    @staticmethod
    def load(path: str) -> SoftPromptWeights:
        with open(path, "r", encoding="utf-8") as f:
            return SoftPromptWeights.from_json(f.read())
