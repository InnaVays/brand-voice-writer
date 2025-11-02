from __future__ import annotations
import argparse, os
from pathlib import Path
import yaml

from core.softprompt.gemma_loader import load_gemma
from core.softprompt.gemma_peft import SoftPromptTrainer, TrainConfig
from core.models.registry import Registry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--style", required=True, help="style_id (folder under raw/ or dataset name)")
    ap.add_argument("--virtual-tokens", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bsz", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    registry = Registry(cfg["paths"]["registry"])

    style = registry.get_style(args.style)
    if not style or "dataset" not in style:
        raise SystemExit(f"Dataset for style '{args.style}' not found. Run prepare_corpus first.")

    model_id = cfg["models"]["rewriter_base"]["hf_id"]
    device   = cfg["models"]["rewriter_base"].get("device", "auto")
    tok, model, dev = load_gemma(model_id, device=device)

    tc = TrainConfig(
        virtual_tokens = args.virtual_tokens or 40,
        lr             = args.lr or 5e-3,
        epochs         = args.epochs or 3,
        batch_size     = args.bsz or 8,
        max_seq_len    = args.max_seq_len or 512,
        style_id       = args.style,
    )

    out_root = cfg["paths"]["softprompt_root"]
    trainer  = SoftPromptTrainer(model, tok, tc, dev)
    res      = trainer.train(style["dataset"], out_root)

    registry.upsert_style(args.style, peft_dir=res["style_dir"])
    print(f"[{args.style}] trained → {res['style_dir']}")

if __name__ == "__main__":
    main()
