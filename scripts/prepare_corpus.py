from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()

import argparse, os
from pathlib import Path
from tqdm import tqdm
import yaml

from core.io.loaders import DocLoader
from core.io.cleaners import Cleaner
from core.corpus.chunker import Chunker
from core.corpus.summarizer import SmallSummarizer
from core.corpus.dataset import DatasetBuilder
from core.models.registry import StyleRegistry

def build_for_style(style_id: str, src_dir: Path, out_dir: Path, target_chunks: int, words_per_chunk: int, summarizer) -> str:
    loader = DocLoader()
    cleaner = Cleaner()
    chunker = Chunker()
    ds = DatasetBuilder()

    docs = loader.load_dir(str(src_dir))
    if not docs:
        return ""

    chunks_all = []
    for d in docs:
        txt = cleaner.normalize(d["text"])
        chunks = chunker.make_chunks(txt)
        chunks_all.extend(chunks)

    chunks_all = chunker.sample_target(chunks_all, target_n=1000000)

    pairs = []
    for c in tqdm(chunks_all, desc=f"Gisting[{style_id}]"):
        gist = summarizer.summarize(c)
        pairs.append((c, gist))

    out_path = out_dir / f"{style_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.build_jsonl(pairs, str(out_path))
    return str(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--raw", default=None, help="Override raw path")
    ap.add_argument("--out", default=None, help="Override datasets dir")
    ap.add_argument("--style", default=None, help="Single style id (optional). If omitted, uses subfolders of raw/")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    raw_root = Path(args.raw or cfg["paths"]["raw"])
    out_dir = Path(args.out or cfg["paths"]["datasets"])

    # summarizer model (download on first run, reuse later)
    s_cfg  = cfg["summarizer"]
    summarizer = SmallSummarizer(
        model_id=s_cfg["model_id"],
        cache_dir=s_cfg.get("cache_dir", ".hf_cache"),
        device=cfg.get("device", "auto"),
        max_new_tokens=s_cfg.get("max_new_tokens", 80),
        temperature=s_cfg.get("temperature", 0.0),
    )

    os.makedirs(out_dir, exist_ok=True)
    reg = StyleRegistry(cfg["paths"]["registry"])

    styles = []
    if args.style:
        styles = [args.style]
    else:
        # style folders are data/raw/<style_id>/*
        styles = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])

        # fallback: if raw contains files directly → single 'default' style
        if not styles:
            styles = ["default"]

    for sid in styles:
        src = raw_root if sid == "default" else raw_root / sid
        dataset_path = build_for_style(
            style_id=sid,
            src_dir=src,
            out_dir=out_dir,
            target_chunks=cfg["ingest"]["target_chunks"],
            words_per_chunk=cfg["ingest"]["words_per_chunk"],
            summarizer=summarizer,
        )
        if dataset_path:
            reg.upsert_style(style_id=sid, dataset_path=dataset_path)
            print(f"[OK] {sid} → {dataset_path}")
        else:
            print(f"[SKIP] {sid}: no documents found")

    print("Registered styles:", reg.list_styles())

if __name__ == "__main__":
    main()
