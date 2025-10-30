# pipeline/posttrain_report.py
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime

import numpy as np


TEMPLATE = """# Brand Voice Rewriter — Training Report

**Adapter:** `{adapter}`  
**Model:** `{model_id}`  
**Date:** {date}

---

## Summary

- Virtual tokens: **{num_virtual_tokens}**
- Hidden dim: **{hidden_dim}**
- Training time (approx): **{train_minutes} minutes**
- Chunks used: **{num_chunks}** × ~120 words
- Epochs: **{epochs}**
- Batch size: **{batch_size}**
- Learning rate: **{lr}**

---

## Soft Prompt Visuals

- Heatmap: `softprompt_heatmap.png`
- PCA scatter: `softprompt_pca.png`

*(Saved under `{fig_dir}`)*

---

## Notes

- This adapter learns a style prior (soft prompt) without altering base model weights.
- Use `pipeline.rewrite` with `--adapter {adapter}` to rewrite drafts in the learned voice.

"""


def infer_meta(adapter_dir: str) -> dict:
    """
    Try to read training metadata if saved by you manually later.
    If not found, provide safe defaults.
    """
    meta = {
        "model_id": "google/gemma-2b-it",
        "num_virtual_tokens": 40,
        "hidden_dim": "unknown",
        "train_minutes": "~",
        "num_chunks": "~",
        "epochs": 2,
        "batch_size": 2,
        "lr": 5e-5,
    }
    # Optional: if you add a meta.json to adapter_dir later, we'll pull from it.
    meta_path = os.path.join(adapter_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta.update(json.load(f))
        except Exception:
            pass

    # If figures include softprompt.npy, we can deduce hidden dim & tokens
    npy_guess = os.path.join("figures", "softprompt.npy")
    if os.path.exists(npy_guess):
        try:
            W = np.load(npy_guess)
            meta["num_virtual_tokens"] = int(W.shape[0])
            meta["hidden_dim"] = int(W.shape[1])
        except Exception:
            pass
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to adapter dir (PEFT)")
    ap.add_argument("--fig_dir", default="figures", help="Directory with visualization outputs")
    ap.add_argument("--out", default="reports", help="Directory to save markdown report")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = infer_meta(args.adapter)

    report = TEMPLATE.format(
        adapter=args.adapter,
        model_id=meta["model_id"],
        date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        num_virtual_tokens=meta["num_virtual_tokens"],
        hidden_dim=meta["hidden_dim"],
        train_minutes=meta["train_minutes"],
        num_chunks=meta["num_chunks"],
        epochs=meta["epochs"],
        batch_size=meta["batch_size"],
        lr=meta["lr"],
        fig_dir=os.path.abspath(args.fig_dir),
    )
    # Save
    base = os.path.basename(os.path.normpath(args.adapter))
    out_path = os.path.join(args.out, f"report_{base}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Saved report → {out_path}")


if __name__ == "__main__":
    main()
