from __future__ import annotations
import argparse, os
from pathlib import Path
from tqdm import tqdm

from core.io.loaders import DocLoader
from core.io.cleaners import Cleaner
from core.corpus.chunker import Chunker
from core.corpus.summarizer import GistBuilder
from core.corpus.dataset import DatasetBuilder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/datasets/train.jsonl")
    ap.add_argument("--target-chunks", type=int, default=120)
    ap.add_argument("--words-per-chunk", type=int, default=120)
    ap.add_argument("--model", default="google/gemma-2b-it")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    loader = DocLoader()
    cleaner = Cleaner()
    chunker = Chunker()
    gist = GistBuilder(model_ref=args.model, device=args.device)
    ds = DatasetBuilder()

    docs = loader.load_dir(args.raw)
    if not docs:
        raise SystemExit("No documents found in data/raw")

    chunks_all = []
    for d in docs:
        txt = cleaner.normalize(d["text"])
        chunks = chunker.make_chunks(txt, target_words=args.words_per_chunk)
        chunks_all.extend(chunks)

    chunks_all = chunker.sample_target(chunks_all, target_n=args.target_chunks)

    pairs = []
    for c in tqdm(chunks_all, desc="Gisting"):
        pairs.append((c, gist.gist(c)))

    os.makedirs(Path(args.out).parent, exist_ok=True)
    ds.build_jsonl(pairs, args.out)
    print(f"Saved dataset to {args.out} with {len(pairs)} examples.")

if __name__ == "__main__":
    main()
