from __future__ import annotations
import argparse, os, json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from core.models.gemma_loader import load_gemma
from core.models.gemma_peft import SoftPromptTrainer, TrainConfig
from core.eval.metrics import RunMetrics, save_run_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/train.jsonl")
    ap.add_argument("--out", default="artifacts/softprompt")
    ap.add_argument("--style", required=True, help="Style ID / name (e.g., my_brand)")
    ap.add_argument("--model", default="google/gemma-2b-it")
    ap.add_argument("--virtual-tokens", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    tok, model, device = load_gemma(args.model, device=args.device)
    cfg = TrainConfig(
        virtual_tokens=args.virtual_tokens,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.bsz,
        max_seq_len=args.max_seq_len,
        style_id=args.style
    )
    trainer = SoftPromptTrainer(model, tok, cfg, device)
    res = trainer.train(args.data, args.out)

    # simple run log
    metrics = RunMetrics(
        seconds=res["seconds"],
        n_examples=sum(1 for _ in open(args.data, "r", encoding="utf-8")),
        virtual_tokens=args.virtual_tokens,
        epochs=args.epochs,
        batch_size=args.bsz,
        lr=args.lr
    )
    run_dir = os.path.join(args.out, args.style)
    run_path = save_run_json(metrics.to_dict(), run_dir)
    print(f"Saved PEFT soft prompt: {res['style_dir']}")
    print(f"Saved run log: {run_path}")

if __name__ == "__main__":
    main()
