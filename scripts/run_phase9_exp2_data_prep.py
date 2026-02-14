"""Phase 9 Experiment 2: MUSTS 50/50 train/test split for 5 languages.

Loads all available MUSTS splits per language from HuggingFace,
concatenates them, and applies a deterministic 50/50 train/test split.

Languages: english, french, sinhala, tamil, serbian

Outputs per language:
- runs/phase9/experiment2/{language}_split.csv
  columns: sentence_1, sentence_2, similarity, split
- runs/phase9/experiment2/split_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


LANGUAGES = ["english", "french", "sinhala", "tamil", "serbian"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Exp 2 data prep.")
    parser.add_argument("--out_dir", default="runs/phase9/experiment2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--languages", default=",".join(LANGUAGES),
        help="Comma-separated language names.",
    )
    return parser.parse_args()


def load_all_splits(language: str) -> pd.DataFrame:
    """Load all available splits for a MUSTS language and concatenate."""
    dataset_name = f"musts/{language}"
    frames = []
    for split_name in ["train", "validation", "test"]:
        try:
            ds = load_dataset(dataset_name, split=split_name)
            df = ds.to_pandas()
            frames.append(df)
            print(f"  Loaded {split_name}: {len(df)} rows")
        except Exception:
            pass
    if not frames:
        raise RuntimeError(f"No splits found for {dataset_name}")
    combined = pd.concat(frames, ignore_index=True)
    # Keep only needed columns and drop rows with None
    combined = combined[["sentence_1", "sentence_2", "similarity"]].dropna()
    combined["similarity"] = combined["similarity"].astype(float)
    return combined


def split_5050(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply deterministic 50/50 train/test split."""
    rng = np.random.default_rng(seed)
    n = len(df)
    indices = rng.permutation(n)
    split_point = n // 2
    labels = np.full(n, "test", dtype=object)
    labels[indices[:split_point]] = "train"
    df = df.copy()
    df["split"] = labels
    return df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    summary_rows = []

    for language in languages:
        print(f"\n{'='*40} {language} {'='*40}")
        df = load_all_splits(language)
        print(f"  Total after dedup/dropna: {len(df)}")

        df = split_5050(df, seed=args.seed)

        n_train = int((df["split"] == "train").sum())
        n_test = int((df["split"] == "test").sum())
        print(f"  Train: {n_train}, Test: {n_test}")

        out_path = out_dir / f"{language}_split.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

        summary_rows.append({
            "language": language,
            "total": len(df),
            "n_train": n_train,
            "n_test": n_test,
            "train_pct": round(100 * n_train / len(df), 1),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
