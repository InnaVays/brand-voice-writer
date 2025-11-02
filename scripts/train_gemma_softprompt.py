from __future__ import annotations
import argparse, os
from core.softprompt.gemma_loader import load_gemma
from core.softprompt.gemma_peft import SoftPromptTrainer, TrainConfig
from core.eval.metrics import RunMetrics, save_run_json
from core.models.registry import StyleRegistry
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--style", required=True, help="Style ID (e.g., fintech)")
    ap.add_argument("--data", default=None, help="Override dataset path (optional)")
    ap.add_argument("--virtual-tokens", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bsz", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    reg = StyleRegistry(cfg["paths"]["registry"])

    dataset_path = args.data or reg.get(args.style).get("dataset")
    if not dataset_path or not os.path.exists(dataset_path):
        raise SystemExit(f"No dataset for style '{args.style}'. Run prepare_corpus first.")

    vtok = args.virtual_tokens or cfg["virtual_tokens"]
    epochs = args.epochs or cfg["epochs"]
    bsz = args.bsz or cfg["batch_size"]
    lr = args.lr or cfg["lr"]
    msl = args.max_seq_len or cfg["max_seq_len"]
    device = args.device or cfg.get("device", "auto")
    model_id = cfg["model_id"]

    tok, model, device = load_gemma(model_id, device=device)
    trainer = SoftPromptTrainer(
        model, tok,
        TrainConfig(virtual_tokens=vtok, lr=lr, epochs=epochs, batch_size=bsz, max_seq_len=msl, style_id=args.style),
        device
    )
    res = trainer.train(dataset_jsonl=dataset_path, out_base_dir=cfg["paths"]["softprompt"])

    # log + registry update
    metrics = RunMetrics(seconds=res["seconds"], n_examples=sum(1 for _ in open(dataset_path, "r", encoding="utf-8")),
                         virtual_tokens=vtok, epochs=epochs, batch_size=bsz, lr=lr)
    run_dir = os.path.join(cfg["paths"]["softprompt"], args.style)
    save_run_json(metrics.to_dict(), run_dir)

    reg.upsert_style(style_id=args.style, adapter_dir=run_dir)
    print(f"[OK] Trained adapter for '{args.style}' at {run_dir}")

if __name__ == "__main__":
    main()
