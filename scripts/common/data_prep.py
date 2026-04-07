"""Phase 9 data preparation: 50/50 train/test split for canon dataset.

Split rules (targeting ~639 test / ~639 train):
- Doubletons (n=2): 1 train, 1 test
- Multi-file (n>=3): test_count = n // 2, train_count = n - test_count
  (train always gets >= ceil(n/2) >= 2 files)
- Singletons (n=1): distributed to balance totals toward 50/50

Outputs:
- phase9_split.csv  — compatible with extraction CLIs (--split_csv)
- train.tsv         — file_id, content, directory_id, semantically_equal_ids
- test.tsv          — same schema
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import list_txt_files, load_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 data prep: 50/50 split.")
    parser.add_argument(
        "--canon_root", default="canon",
        help="Root directory of the canon dataset.",
    )
    parser.add_argument(
        "--out_dir", default="runs/phase9",
        help="Output directory for split files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_meta(canon_root: str) -> pd.DataFrame:
    entries = list_txt_files(canon_root)
    df = pd.DataFrame(entries, columns=["folder_id", "filename", "path"])
    folder_sizes = df.groupby("folder_id")["filename"].transform("count")
    df["folder_size"] = folder_sizes
    df["is_singleton"] = df["folder_size"] == 1
    df["is_winnable"] = df["folder_size"] >= 2
    df = df.reset_index(drop=True)
    df["file_id"] = np.arange(len(df), dtype=np.int32)
    return df


def phase9_split(meta: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Apply 50/50 split logic.

    1. Assign doubletons and multi-file dirs deterministically.
    2. Count how many singletons need to go to test to reach ~50%.
    3. Randomly assign that many singletons to test, rest to train.
    """
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    split = np.full(len(meta), "", dtype=object)

    total_files = len(meta)
    target_test = total_files // 2  # 639 for 1278 total

    # Pass 1: assign doubletons and multi-file dirs
    test_count_so_far = 0
    singleton_indices = []

    for folder_id, group in meta.groupby("folder_id"):
        idx = group.index.to_numpy()
        n = len(idx)

        if n == 1:
            singleton_indices.append(idx[0])
            continue

        if n == 2:
            # 1 train, 1 test
            shuffled = rng.permutation(idx)
            split[shuffled[0]] = "train"
            split[shuffled[1]] = "test"
            test_count_so_far += 1
        else:
            # n >= 3: test_count = n // 2
            test_n = n // 2
            train_n = n - test_n
            shuffled = rng.permutation(idx)
            split[shuffled[:test_n]] = "test"
            split[shuffled[test_n:]] = "train"
            test_count_so_far += test_n

    # Pass 2: distribute singletons to reach target_test
    singletons_needed_in_test = max(0, target_test - test_count_so_far)
    singletons_needed_in_test = min(singletons_needed_in_test, len(singleton_indices))

    singleton_indices = np.array(singleton_indices)
    rng.shuffle(singleton_indices)
    split[singleton_indices[:singletons_needed_in_test]] = "test"
    split[singleton_indices[singletons_needed_in_test:]] = "train"

    meta["split"] = split

    # has_test_partner: True if folder has >= 2 files in the test set
    test_mask = meta["split"] == "test"
    folder_test_counts = meta[test_mask].groupby("folder_id").size()
    meta["has_test_partner"] = meta["folder_id"].map(
        lambda fid: folder_test_counts.get(fid, 0) >= 2
    )

    # is_test_query: test files that have a test partner (meaningful retrieval)
    meta["is_test_query"] = test_mask & meta["has_test_partner"]

    return meta


def build_tsv(meta: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Build TSV with columns: file_id, content, directory_id, semantically_equal_ids."""
    subset = meta[meta["split"] == split_name].copy()

    # For each file, find other files in same folder within the same split
    folder_groups = subset.groupby("folder_id")["file_id"].apply(list).to_dict()

    rows = []
    for _, row in subset.iterrows():
        fid = row["file_id"]
        folder = row["folder_id"]
        content = load_text(row["path"])
        partners = [str(x) for x in folder_groups[folder] if x != fid]
        sem_eq = ",".join(partners) if partners else ""
        rows.append({
            "file_id": fid,
            "content": content,
            "directory_id": folder,
            "semantically_equal_ids": sem_eq,
        })

    return pd.DataFrame(rows)


def print_summary(meta: pd.DataFrame) -> None:
    total = len(meta)
    n_train = int((meta["split"] == "train").sum())
    n_test = int((meta["split"] == "test").sum())

    n_folders = meta["folder_id"].nunique()
    folder_sizes = meta.groupby("folder_id")["filename"].count()
    n_singletons = int((folder_sizes == 1).sum())
    n_doubletons = int((folder_sizes == 2).sum())
    n_multi = int((folder_sizes >= 3).sum())

    test_mask = meta["split"] == "test"
    n_existing = int(meta.loc[test_mask, "has_test_partner"].sum())
    n_new = int(test_mask.sum()) - n_existing
    n_test_query = int(meta["is_test_query"].sum())

    # Verify no overlap
    train_ids = set(meta.loc[meta["split"] == "train", "file_id"])
    test_ids = set(meta.loc[meta["split"] == "test", "file_id"])
    overlap = train_ids & test_ids

    print("=" * 60)
    print("Phase 9 Split Summary")
    print("=" * 60)
    print(f"Total files:        {total}")
    print(f"Total folders:      {n_folders}")
    print(f"  Singletons:       {n_singletons}")
    print(f"  Doubletons:       {n_doubletons}")
    print(f"  Multi-file (>=3): {n_multi}")
    print(f"Train files:        {n_train} ({100*n_train/total:.1f}%)")
    print(f"Test files:         {n_test} ({100*n_test/total:.1f}%)")
    print(f"  Existing (has partner): {n_existing}")
    print(f"  New (no partner):       {n_new}")
    print(f"  Test queries:           {n_test_query}")
    print(f"Train/test overlap: {len(overlap)}")

    # Positive pairs in each split
    for split_name in ["train", "test"]:
        sub = meta[meta["split"] == split_name]
        pairs = 0
        for _, grp in sub.groupby("folder_id"):
            n = len(grp)
            pairs += n * (n - 1) // 2
        print(f"Positive pairs in {split_name}: {pairs}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building metadata...")
    meta = build_meta(args.canon_root)

    print("Applying 50/50 split...")
    meta = phase9_split(meta, seed=args.seed)

    # Save split CSV
    split_csv_path = out_dir / "phase9_split.csv"
    meta.to_csv(split_csv_path, index=False)
    print(f"Saved: {split_csv_path}")

    print_summary(meta)

    # Build and save TSVs
    print("Building train.tsv...")
    train_tsv = build_tsv(meta, "train")
    train_path = out_dir / "train.tsv"
    train_tsv.to_csv(train_path, sep="\t", index=False)
    print(f"Saved: {train_path} ({len(train_tsv)} rows)")

    print("Building test.tsv...")
    test_tsv = build_tsv(meta, "test")
    test_path = out_dir / "test.tsv"
    test_tsv.to_csv(test_path, sep="\t", index=False)
    print(f"Saved: {test_path} ({len(test_tsv)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
