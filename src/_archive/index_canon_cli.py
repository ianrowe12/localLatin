from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from canon_retrieval import build_meta, load_texts, meta_stats, token_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canon meta.csv and token length audit.")
    parser.add_argument("--canon_root", required=True, help="Path to canon/ dataset.")
    parser.add_argument("--runs_root", required=True, help="Directory to store meta.csv.")
    parser.add_argument("--model_name", default="bowphs/LaTa", help="Tokenizer model name.")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for token lengths.")
    return parser.parse_args()


def batched_lengths(
    tokenizer, paths: List[str], max_length: int, batch_size: int
) -> np.ndarray:
    lengths: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        texts = load_texts(batch_paths)
        lengths.append(token_lengths(tokenizer, texts, max_length=max_length))
    return np.concatenate(lengths, axis=0)


def main() -> None:
    args = parse_args()
    canon_root = Path(args.canon_root)
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    meta_path = runs_root / "meta.csv"
    meta = build_meta(str(canon_root), str(meta_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lengths = batched_lengths(
        tokenizer, meta["path"].tolist(), args.max_length, args.batch_size
    )
    meta["token_length"] = lengths
    meta.to_csv(meta_path, index=False)

    stats = meta_stats(meta)
    winnable_fraction = stats.winnable_files / stats.total_files if stats.total_files else 0.0
    print(f"Meta saved to: {meta_path}")
    print(stats)
    print(
        f"Token length stats (max_length={args.max_length}): "
        f"mean={lengths.mean():.2f}, max={lengths.max()}, min={lengths.min()}"
    )
    print(f"Winnable fraction: {winnable_fraction:.4f}")


if __name__ == "__main__":
    main()
