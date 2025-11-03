from __future__ import annotations

# --- ultra-light CPU env BEFORE heavy imports ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("MKLDNN", "0")

from _bootstrap import add_repo_root
add_repo_root()

import argparse
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


def _ultra_light_defaults(cfg: dict, args) -> tuple:
    """
    Возвращает минимальные дефолты, если не переопределены аргументами/конфигом.
    Оптимизировано под CPU + soft prompt + очень маленький датасет.
    """
    # безопасные "низкие" дефолты
    DEF_VTOK = 24
    DEF_EPOCHS = 10
    DEF_BSZ = 1
    DEF_LR = 5e-3
    DEF_MAX_SEQ_LEN = 96  # при твоих 60 токенов I/O это с запасом

    vtok  = args.virtual_tokens or cfg.get("virtual_tokens", DEF_VTOK)
    epochs = args.epochs or cfg.get("epochs", DEF_EPOCHS)
    bsz   = args.bsz or cfg.get("batch_size", DEF_BSZ)
    lr    = args.lr or cfg.get("lr", DEF_LR)
    msl   = args.max_seq_len or cfg.get("max_seq_len", DEF_MAX_SEQ_LEN)

    # жёстко защёлкиваем вниз (если в конфиге выше)
    vtok  = min(vtok, DEF_VTOK)
    epochs = min(epochs, DEF_EPOCHS)
    bsz   = 1  # всегда 1 для экономии памяти
    msl   = min(msl, DEF_MAX_SEQ_LEN)

    return vtok, epochs, bsz, lr, msl


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

    vtok, epochs, bsz, lr, msl = _ultra_light_defaults(cfg, args)
    # по умолчанию — cpu
    device = (args.device or cfg.get("device") or "cpu")
    model_id = cfg["model_id"]

    # ВАЖНО: load_model должен грузить с low_cpu_mem_usage=True и device_map={"": "cpu"}
    # (это внутри самого load_model; если нет — добавь там эти флаги)
    tok, model, device = load_model(model_id, device=device)

    trainer = SoftPromptTrainer(
        model, tok,
        TrainConfig(
            virtual_tokens=vtok,
            lr=lr,
            epochs=epochs,
            batch_size=bsz,       # принудительно 1
            max_seq_len=msl,      # <= 96
            style_id=args.style
        ),
        device
    )

    res = trainer.train(dataset_jsonl=dataset_path, out_base_dir=cfg["paths"]["softprompt"])

    # log + registry update
    n_examples = sum(1 for _ in open(dataset_path, "r", encoding="utf-8"))
    metrics = RunMetrics(
        seconds=res["seconds"],
        n_examples=n_examples,
        virtual_tokens=vtok,
        epochs=epochs,
        batch_size=bsz,
        lr=lr
    )
    run_dir = os.path.join(cfg["paths"]["softprompt"], args.style)
    save_run_json(metrics.to_dict(), run_dir)

    reg.upsert_style(style_id=args.style, adapter_dir=run_dir)
    print(f"[OK] Trained ultra-light adapter for '{args.style}' at {run_dir}")


if __name__ == "__main__":
    main()
