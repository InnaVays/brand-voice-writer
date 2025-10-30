# pipeline/scorer.py
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.schema import Candidate


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9а-яё\- ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")


def _hashing_vector(tokens: List[str], dim: int = 256, salt: str = "bvr") -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h = hash(salt + t)  # fast, stable within process; acceptable for relative scoring
        idx = (h % dim + dim) % dim
        vec[idx] += 1.0
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec = vec / nrm
    return vec


def style_embedding(text: str, dim: int = 256) -> np.ndarray:
    tokens = _tokenize(text)
    base = _hashing_vector(tokens, dim=dim, salt="style")

    # Rhythm / sentence-length features (4 dims)
    sents = re.split(r"[.!?]+", text)
    lens = [len(_tokenize(s)) for s in sents if s.strip()]
    if not lens:
        lens = [len(tokens)]
    r = np.array(
        [
            float(np.mean(lens)),
            float(np.std(lens)),
            float(np.median(lens)),
            float(len(sents)),
        ],
        dtype=np.float32,
    )
    r = r / (np.linalg.norm(r) + 1e-8)

    out = np.concatenate([base, np.pad(r, (0, max(0, base.shape[0] - 4)), constant_values=0.0)])[: base.shape[0]]
    out = out / (np.linalg.norm(out) + 1e-8)
    return out


def semantic_embedding(text: str, dim: int = 256) -> np.ndarray:
    tokens = _tokenize(text)
    return _hashing_vector(tokens, dim=dim, salt="semantic")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def mahalanobis(z: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> float:
    d = z - mu
    val = float(d @ Sigma_inv @ d.T)
    return float(math.sqrt(max(val, 0.0)))


@dataclass
class BrandSpace:
    mu: np.ndarray
    Sigma_inv: np.ndarray
    dim: int
    n: int


def build_brand_space(reference_texts: List[str], dim: int = 256, eps: float = 1e-3) -> BrandSpace:
    if not reference_texts:
        raise ValueError("build_brand_space: reference_texts is empty")
    Z = np.stack([style_embedding(t, dim=dim) for t in reference_texts], axis=0)  # (N, D)
    mu = Z.mean(axis=0)
    X = Z - mu
    Sigma = (X.T @ X) / max(1, Z.shape[0] - 1)
    Sigma += eps * np.eye(dim, dtype=np.float32)
    Sigma_inv = np.linalg.inv(Sigma)
    return BrandSpace(mu=mu, Sigma_inv=Sigma_inv, dim=dim, n=Z.shape[0])

def readability_proxy(text: str) -> float:
    """
    Very rough readability proxy in [0..1] (higher=more readable).
    Uses average sentence length + word length as a cheap surrogate.
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    avg_word_len = np.mean([len(w) for w in words])
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sent_len = np.mean([len(_tokenize(s)) for s in sents]) if sents else len(words)

    # Normalize into (0,1): penalize long words & sentences
    wl_term = max(0.0, 1.0 - (avg_word_len - 4.0) / 6.0)   # ~4-10 letters → 1..0
    sl_term = max(0.0, 1.0 - (avg_sent_len - 12.0) / 20.0) # ~12-32 words → 1..0
    return float(np.clip(0.5 * wl_term + 0.5 * sl_term, 0.0, 1.0))

def length_adherence(generated: str, source: Optional[str], target_ratio: float = 1.0) -> float:
    """
    Returns [0..1], 1 if length close to target_ratio * len(source).
    """
    gw = len(_tokenize(generated))
    if source is None:
        return 1.0
    sw = len(_tokenize(source))
    if sw == 0:
        return 1.0
    expected = target_ratio * sw
    rel_err = abs(gw - expected) / max(expected, 1.0)
    return float(np.clip(1.0 - rel_err, 0.0, 1.0))


@dataclass
class ScoreWeights:
    alpha_semantic: float = 0.4
    beta_length: float = 0.2
    gamma_readability: float = 0.1
    # style distance contributes negatively (smaller is better)


def score_text(
    text: str,
    brand: BrandSpace,
    source: Optional[str] = None,
    weights: ScoreWeights = ScoreWeights(),
    dim: int = 256,
) -> Dict[str, float]:
    # Style distance (Mahalanobis in brand space)
    z = style_embedding(text, dim=dim)
    d_style = mahalanobis(z, brand.mu, brand.Sigma_inv)  # smaller better

    # Semantic similarity vs. source (if given)
    s_sem = 0.0
    if source:
        ea = semantic_embedding(text, dim=dim)
        eb = semantic_embedding(source, dim=dim)
        s_sem = cosine_sim(ea, eb)  # [-1..1], typically [0..1]

    # Length adherence
    s_len = length_adherence(text, source, target_ratio=1.0)

    # Readability
    s_read = readability_proxy(text)

    # Aggregate: higher is better
    agg = -d_style + weights.alpha_semantic * s_sem + weights.beta_length * s_len + weights.gamma_readability * s_read

    return {
        "style_distance": float(d_style),
        "semantic_similarity": float(s_sem),
        "length_adherence": float(s_len),
        "readability": float(s_read),
        "score": float(agg),
    }


def rank_candidates(
    candidates_texts: List[str],
    brand: BrandSpace,
    source_text: Optional[str] = None,
    weights: ScoreWeights = ScoreWeights(),
    dim: int = 256,
) -> List[Candidate]:
    scored: List[Tuple[Candidate, float]] = []
    for t in candidates_texts:
        m = score_text(t, brand=brand, source=source_text, weights=weights, dim=dim)
        cand = Candidate(text=t, score=m["score"], meta=m)  # meta holds all metrics
        scored.append((cand, m["score"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]
