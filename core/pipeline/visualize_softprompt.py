# pipeline/visualize_softprompt.py
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from model.gemma_peft import load_gemma_base, load_prompt_tuning, extract_softprompt_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to PEFT adapter dir")
    ap.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    ap.add_argument("--out_dir", type=str, default="./figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok, base, device = load_gemma_base(args.model_id)
    peft = load_prompt_tuning(base, args.adapter)

    W = extract_softprompt_matrix(peft)  # (tokens, dim)
    np.save(os.path.join(args.out_dir, "softprompt.npy"), W)

    # Heatmap (normalized)
    Wn = (W - W.mean()) / (W.std() + 1e-8)
    plt.figure(figsize=(10, 4))
    plt.imshow(Wn, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    plt.colorbar()
    plt.title(f"Soft Prompt Weights (z-scored) — shape {W.shape}")
    plt.xlabel("Hidden dim")
    plt.ylabel("Virtual token index")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "softprompt_heatmap.png"), dpi=180)
    plt.close()

    # PCA to 2D (very quick)
    U, S, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
    PC = U[:, :2] * S[:2]
    plt.figure(figsize=(4, 4))
    plt.scatter(PC[:, 0], PC[:, 1], s=24)
    for i, (x, y) in enumerate(PC):
        plt.text(x, y, str(i), fontsize=8)
    plt.title("Soft Prompt Tokens — PCA(2D)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "softprompt_pca.png"), dpi=180)
    plt.close()

    print(f"[OK] Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
