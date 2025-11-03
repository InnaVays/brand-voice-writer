#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize soft-prompt weights before/after training.

Outputs:
  - l2_shift.png
  - cosine_per_token.png
  - heatmap_init.png
  - heatmap_trained.png
  - heatmap_delta.png
  - pca_before_after.png
Also prints a small summary.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def _load_matrix(run_dir: str, base: str) -> np.ndarray | None:
    """Loads matrix from .npy or .tsv if present."""
    npy = os.path.join(run_dir, f"{base}.npy")
    tsv = os.path.join(run_dir, f"{base}.tsv")
    if os.path.exists(npy):
        return np.load(npy)
    if os.path.exists(tsv):
        rows = []
        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append([float(x) for x in line.split("\t")])
        return np.asarray(rows, dtype=np.float32)
    return None

def _safe_cosine(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=axis, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=axis, keepdims=True) + 1e-12
    return (a * b).sum(axis=axis) / (a_norm.squeeze() * b_norm.squeeze())

def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Center
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]           # 2 principal components
    proj = Xc @ comps.T      # (n,2)
    return proj, comps

def _save_bar(values: np.ndarray, title: str, out_png: str, ylabel: str):
    plt.figure()
    plt.bar(np.arange(len(values)), values)
    plt.title(title)
    plt.xlabel("virtual token index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _save_heatmap(M: np.ndarray, title: str, out_png: str):
    plt.figure()
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("embedding dim")
    plt.ylabel("virtual token index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _save_pca_arrows(B: np.ndarray, A: np.ndarray, out_png: str):
    """
    PCA of stacked [B; A] to get shared 2D space, then draw arrows B_i -> A_i.
    B: before (n,d), A: after (n,d)
    """
    stacked = np.vstack([B, A])
    proj, _ = _pca_2d(stacked)
    n = B.shape[0]
    Pb, Pa = proj[:n], proj[n:]

    plt.figure()
    # draw points and arrows
    plt.scatter(Pb[:,0], Pb[:,1], label="before", marker="o")
    plt.scatter(Pa[:,0], Pa[:,1], label="after", marker="x")
    for i in range(n):
        plt.arrow(Pb[i,0], Pb[i,1],
                  Pa[i,0] - Pb[i,0], Pa[i,1] - Pb[i,1],
                  head_width=0.02, length_includes_head=True)
    plt.title("Soft-prompt virtual tokens: PCA (before → after)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Directory with saved softprompt_*.npy/tsv and delta_stats.json")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Not a directory: {run_dir}")

    W0 = _load_matrix(run_dir, "softprompt_init")
    W1 = _load_matrix(run_dir, "softprompt_trained")

    if W0 is None or W1 is None:
        raise SystemExit("Missing softprompt_init.* or softprompt_trained.* in run dir")

    if W0.shape != W1.shape:
        raise SystemExit(f"Shape mismatch: init {W0.shape} vs trained {W1.shape}")

    D = W1 - W0
    per_tok_l2 = np.linalg.norm(D, axis=1)
    total_l2 = float(np.linalg.norm(D))
    cos = _safe_cosine(W0, W1, axis=1)

    # Try reading meta to print extra info (optional)
    meta_path = os.path.join(run_dir, "delta_stats.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    # Summary
    print("== Soft-prompt shift summary ==")
    print(f"run_dir:     {run_dir}")
    print(f"V x d:       {W0.shape[0]} x {W0.shape[1]}")
    print(f"total L2:    {total_l2:.6f}")
    print(f"min/mean/max per-token L2: {per_tok_l2.min():.6f} / {per_tok_l2.mean():.6f} / {per_tok_l2.max():.6f}")
    print(f"min/mean/max per-token cosine: {cos.min():.6f} / {cos.mean():.6f} / {cos.max():.6f}")
    if meta:
        print("meta:", {k: meta.get(k) for k in ["virtual_tokens", "embed_dim", "seconds", "epochs", "batch_size", "lr", "max_seq_len"]})

    # Plots
    _save_bar(per_tok_l2, "L2 shift per virtual token", os.path.join(run_dir, "l2_shift.png"), "L2 shift")
    _save_bar(cos, "Cosine similarity per virtual token", os.path.join(run_dir, "cosine_per_token.png"), "cosine")

    _save_heatmap(W0, "Soft-prompt (init) — embeddings", os.path.join(run_dir, "heatmap_init.png"))
    _save_heatmap(W1, "Soft-prompt (trained) — embeddings", os.path.join(run_dir, "heatmap_trained.png"))
    _save_heatmap(D,  "Delta (trained - init)", os.path.join(run_dir, "heatmap_delta.png"))

    _save_pca_arrows(W0, W1, os.path.join(run_dir, "pca_before_after.png"))

    print("Saved figures to:", run_dir)

if __name__ == "__main__":
    main()
