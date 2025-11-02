from __future__ import annotations
import argparse, yaml
from core.models.registry import StyleRegistry
from core.softprompt.gemma_rewriter import GemmaRewriter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--style", required=True, help="Style ID (registered)")
    ap.add_argument("--in", dest="text", required=True)
    ap.add_argument("--tone", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    reg = StyleRegistry(cfg["paths"]["registry"])
    entry = reg.get(args.style)
    peft_dir = entry.get("adapter_dir")
    if not peft_dir or not os.path.isdir(peft_dir):
        raise SystemExit(f"No adapter for style '{args.style}'. Train it first.")

    device = args.device or cfg.get("device", "auto")
    rw = GemmaRewriter(model_id=cfg["model_id"], peft_dir=peft_dir, device=device)
    print(rw.rewrite(args.text, tone=args.tone))

if __name__ == "__main__":
    main()
