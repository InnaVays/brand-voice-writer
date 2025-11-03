from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()

import os
import argparse
from pathlib import Path
import yaml

from core.softprompt.model_loader import load_model
from core.softprompt.model_peft import SoftPromptTrainer, TrainConfig
from core.eval.metrics import RunMetrics, save_run_json
from core.models.registry import StyleRegistry

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass


def _next_run_dir(softprompt_root: str, style_id: str) -> Path:
    """
    Choose next run directory: artifacts/softprompt/<style>/run_XXX
    """
    style_root = Path(softprompt_root) / style_id
    style_root.mkdir(parents=True, exist_ok=True)

    existing = []
    for p in style_root.glob("run_*"):
        try:
            idx = int(p.name.split("_")[-1])
            existing.append(idx)
        except ValueError:
            continue

    next_id = (max(existing) + 1) if existing else 1
    run_name = f"run_{next_id:03d}"
    run_dir = style_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--style", required=True, help="Style ID (e.g., fintech)")
    ap.add_argument("--data", default=None, help="Override dataset path (optional)")
    ap.add_argument("--virtual-tokens", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bsz", type=int, default=None, help="Per-device batch size")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    reg = StyleRegistry(cfg["paths"]["registry"])

    dataset_path = args.data or reg.get(args.style).get("dataset")
    if not dataset_path or not os.path.exists(dataset_path):
        raise SystemExit(f"No dataset for style '{args.style}'. Run prepare_corpus first.")

    device = (args.device or cfg.get("device") or "cpu")
    model_id = cfg["model_id"]
    tok, model, device = load_model(model_id, device=device)

    vtok   = args.virtual_tokens or cfg.get("virtual_tokens", 16)
    epochs = args.epochs         or cfg.get("epochs", 3)
    bsz    = args.bsz            or cfg.get("batch_size", 4)
    lr     = args.lr             or cfg.get("lr", 3e-3)
    msl    = args.max_seq_len    or cfg.get("max_seq_len", 512)

    print(f"[CONFIG] style={args.style} vtok={vtok} epochs={epochs} "
          f"bsz={bsz} lr={lr} max_seq_len={msl} device={device}")

    # --- decide run dir ---
    softprompt_root = cfg["paths"]["softprompt"]
    run_dir = _next_run_dir(softprompt_root, args.style)
    print(f"[RUN_DIR] Saving adapter + logs to {run_dir}")

    trainer = SoftPromptTrainer(
        model, tok,
        TrainConfig(
            virtual_tokens=vtok,
            lr=lr,
            epochs=epochs,
            batch_size=bsz,
            max_seq_len=msl,
            style_id=args.style,
        ),
        device
    )

    res = trainer.train(dataset_jsonl=dataset_path, out_base_dir=str(run_dir))

    n_examples = sum(1 for _ in open(dataset_path, "r", encoding="utf-8"))
    metrics = RunMetrics(
        seconds=res.get("seconds", 0.0) if isinstance(res, dict) else 0.0,
        n_examples=n_examples,
        virtual_tokens=vtok,
        epochs=epochs,
        batch_size=bsz,
        lr=lr,
    )
    save_run_json(metrics.to_dict(), str(run_dir))

    reg.upsert_style(style_id=args.style, adapter_dir=str(run_dir))
    print(f"[OK] Trained adapter for '{args.style}' at {run_dir}")


if __name__ == "__main__":
    main()
