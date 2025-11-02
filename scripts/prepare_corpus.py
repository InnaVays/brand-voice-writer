from __future__ import annotations
import argparse, os, json
from pathlib import Path
from tqdm import tqdm
import yaml

from core.io.loaders import DocLoader
from core.io.cleaners import Cleaner
from core.corpus.chunker import Chunker
from core.corpus.summarizer import GistBuilder
from core.corpus.dataset import DatasetBuilder
from core.models.registry import Registry

def _find_styles(raw_dir: Path) -> dict[str, list[Path]]:
    styles: dict[str, list[Path]] = {}
    # style subdirs
    for p in raw_dir.iterdir():
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            if files:
                styles[p.name] = files
    # files directly in raw → default
    root_files = [f for f in raw_dir.iterdir() if f.is_file()]
    if root_files:
        styles.setdefault("default", []).extend(root_files)
    return styles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw"])
    datasets_dir = Path(cfg["paths"]["datasets"]); datasets_dir.mkdir(parents=True, exist_ok=True)
    registry = Registry(cfg["paths"]["registry"])

    loader = DocLoader()
    cleaner = Cleaner()
    chunker = Chunker()
    gist = GistBuilder(cfg, registry)
    ds = DatasetBuilder()

    styles = _find_styles(raw_dir)
    if not styles:
        raise SystemExit(f"No documents found in {raw_dir}. Create subfolders per style_id or place files in raw/.")

    target_chunks = int(cfg["ingest"]["target_chunks"])
    words_per_chunk = int(cfg["ingest"]["words_per_chunk"])

    for style_id, files in styles.items():
        # Load & chunk
        docs = []
        for f in files:
            docs.extend(loader.load_dir(f.parent if f.is_dir() else f.parent))
            break  # loader.load_dir reads whole dir; avoid duplicating per file
        if not docs:
            print(f"[{style_id}] no docs; skipping")
            continue

        chunks_all = []
        for d in docs:
            txt = cleaner.normalize(d["text"])
            chunks = chunker.make_chunks(txt, target_words=words_per_chunk)
            chunks_all.extend(chunks)

        chunks_all = chunker.sample_target(chunks_all, target_n=target_chunks)

        pairs = []
        for c in tqdm(chunks_all, desc=f"[{style_id}] Gisting"):
            pairs.append((c, gist.gist(c)))

        out_path = datasets_dir / f"{style_id}.jsonl"
        ds.build_jsonl(pairs, str(out_path))
        registry.upsert_style(style_id, dataset_path=str(out_path), meta={"n_examples": len(pairs)})

        print(f"[{style_id}] dataset -> {out_path} ({len(pairs)} examples)")

if __name__ == "__main__":
    main()