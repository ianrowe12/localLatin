"""Phase 12: Select a balanced subset of files for attribution analysis.

Selects ~30 similar files (from directories with >= 2 members) and
~30 dissimilar files (singletons), sampled from the full dataset.
Outputs a CSV with file_id, path, folder_id, category columns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select samples for Phase 12 attribution.")
    parser.add_argument("--split_csv", required=True, help="Path to phase9_split.csv.")
    parser.add_argument("--out_csv", required=True, help="Output sample CSV path.")
    parser.add_argument("--n_similar", type=int, default=30, help="Number of similar (winnable) files.")
    parser.add_argument("--n_dissimilar", type=int, default=30, help="Number of dissimilar (singleton) files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    split = pd.read_csv(args.split_csv)

    # Similar: winnable files (directory has >= 2 members)
    similar_pool = split[split["is_winnable"] == True]
    # Dissimilar: singletons (directory has 1 member)
    dissimilar_pool = split[split["is_singleton"] == True]

    n_sim = min(args.n_similar, len(similar_pool))
    n_dis = min(args.n_dissimilar, len(dissimilar_pool))

    sim_idx = rng.choice(len(similar_pool), size=n_sim, replace=False)
    dis_idx = rng.choice(len(dissimilar_pool), size=n_dis, replace=False)

    sim_sample = similar_pool.iloc[sim_idx].copy()
    sim_sample["category"] = "similar"
    dis_sample = dissimilar_pool.iloc[dis_idx].copy()
    dis_sample["category"] = "dissimilar"

    result = pd.concat([sim_sample, dis_sample], ignore_index=True)
    result = result[["file_id", "path", "folder_id", "folder_size", "is_winnable", "split", "category"]]
    result = result.sort_values("file_id").reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"Selected {n_sim} similar + {n_dis} dissimilar = {len(result)} files")
    print(f"  Similar from {sim_sample['folder_id'].nunique()} unique directories")
    print(f"  Split distribution: {result['split'].value_counts().to_dict()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
