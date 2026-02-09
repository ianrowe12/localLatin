"""Deterministic 50/50 file-level train/test split for the canon dataset.

Split rules (v2 — professor-approved):
- Singletons (1 file): distributed randomly ~50/50 between train and test.
- Doubletons (2 files): whole folders assigned randomly 50/50.
  37 folders -> train, 37 folders -> test.  Both files stay together.
- Multi-file (>=3 files): within each folder, files split ~50/50.

Evaluation: pairs are formed *within* each split.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from canon_retrieval import list_txt_files, load_text


def canon_train_test_split_v2(
    meta: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Add 'split' and 'is_test_query' columns with 50/50 rules."""
    rng = np.random.default_rng(random_seed)
    meta = meta.copy()
    split = np.full(len(meta), "train", dtype=object)

    # --- Singletons: random 50/50 ---
    singleton_mask = meta["folder_size"] == 1
    singleton_idx = meta.index[singleton_mask].to_numpy()
    rng.shuffle(singleton_idx)
    mid = len(singleton_idx) // 2
    split[singleton_idx[:mid]] = "train"
    split[singleton_idx[mid:]] = "test"

    # --- Doubletons: folder-level 50/50 ---
    doubleton_folders = (
        meta.loc[meta["folder_size"] == 2, "folder_id"].unique()
    )
    rng.shuffle(doubleton_folders)
    mid_d = len(doubleton_folders) // 2
    train_d_folders = set(doubleton_folders[:mid_d])
    test_d_folders = set(doubleton_folders[mid_d:])
    for folder_id, group in meta[meta["folder_size"] == 2].groupby("folder_id"):
        idx = group.index.to_numpy()
        if folder_id in train_d_folders:
            split[idx] = "train"
        else:
            split[idx] = "test"

    # --- Multi-file (>=3): within-folder ~50/50 ---
    for folder_id, group in meta[meta["folder_size"] >= 3].groupby("folder_id"):
        idx = group.index.to_numpy()
        n = len(idx)
        shuffled = rng.permutation(idx)
        train_n = n // 2
        if train_n == 0:
            train_n = 1
        split[shuffled[:train_n]] = "train"
        split[shuffled[train_n:]] = "test"

    meta["split"] = split

    # is_test_query: test files that have at least one same-folder partner
    # also in test (needed for meaningful retrieval evaluation)
    test_mask = meta["split"] == "test"
    folder_test_counts = meta[test_mask].groupby("folder_id").size()

    is_test_query = np.zeros(len(meta), dtype=bool)
    for i, row in meta.iterrows():
        if row["split"] != "test":
            continue
        fid = row["folder_id"]
        if fid in folder_test_counts and folder_test_counts[fid] >= 2:
            is_test_query[i] = True

    meta["is_test_query"] = is_test_query
    return meta


def build_meta_with_split_v2(
    canon_root: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build meta DataFrame from canon directory with v2 50/50 split."""
    entries = list_txt_files(canon_root)
    df = pd.DataFrame(entries, columns=["folder_id", "filename", "path"])
    folder_sizes = df.groupby("folder_id")["filename"].transform("count")
    df["folder_size"] = folder_sizes
    df["is_singleton"] = df["folder_size"] == 1
    df["is_winnable"] = df["folder_size"] >= 2
    df = df.reset_index(drop=True)
    df["file_id"] = np.arange(len(df), dtype=np.int32)
    df = canon_train_test_split_v2(df, random_seed=random_seed)
    return df


def generate_pairs_tsv(
    meta: pd.DataFrame,
    split_name: str,
    output_path: str,
) -> Dict:
    """Generate ALL pairwise combinations within a split as TSV.

    Columns: file_1 (text content), file_2 (text content),
             semantically_equal (0/1), comparison_id
    """
    split_df = meta[meta["split"] == split_name].reset_index(drop=True)
    n = len(split_df)

    # Pre-load all texts
    texts = {}
    for _, row in split_df.iterrows():
        fid = row["file_id"]
        text = load_text(row["path"])
        # Clean for TSV: replace tabs and newlines with spaces
        texts[fid] = " ".join(text.split())

    folder_ids = split_df.set_index("file_id")["folder_id"].to_dict()
    file_ids = split_df["file_id"].tolist()

    n_pos = 0
    n_neg = 0
    pair_idx = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file_1", "file_2", "semantically_equal", "comparison_id"])

        for i in range(len(file_ids)):
            for j in range(i + 1, len(file_ids)):
                fid_a = file_ids[i]
                fid_b = file_ids[j]
                same = 1 if folder_ids[fid_a] == folder_ids[fid_b] else 0
                comp_id = f"{split_name}_{pair_idx}"
                writer.writerow([texts[fid_a], texts[fid_b], same, comp_id])
                if same:
                    n_pos += 1
                else:
                    n_neg += 1
                pair_idx += 1

    return {
        "split": split_name,
        "n_files": n,
        "n_pairs": n_pos + n_neg,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "output_path": output_path,
    }


def split_summary_v2(meta: pd.DataFrame) -> dict:
    """Compute summary statistics for a v2 split."""
    total = len(meta)
    n_train = int((meta["split"] == "train").sum())
    n_test = int((meta["split"] == "test").sum())
    n_test_query = int(meta["is_test_query"].sum())

    folder_sizes = meta.groupby("folder_id")["filename"].count()
    n_singletons = int((folder_sizes == 1).sum())
    n_pairs = int((folder_sizes == 2).sum())
    n_large = int((folder_sizes >= 3).sum())

    # Singleton distribution
    singletons = meta[meta["folder_size"] == 1]
    n_sing_train = int((singletons["split"] == "train").sum())
    n_sing_test = int((singletons["split"] == "test").sum())

    # Doubleton folder distribution
    doubleton_train = 0
    doubleton_test = 0
    for fid, g in meta[meta["folder_size"] == 2].groupby("folder_id"):
        if g["split"].iloc[0] == "train":
            doubleton_train += 1
        else:
            doubleton_test += 1

    train_set = set(meta.loc[meta["split"] == "train", "file_id"])
    test_set = set(meta.loc[meta["split"] == "test", "file_id"])

    return {
        "total_files": total,
        "n_train": n_train,
        "n_test": n_test,
        "n_test_query": n_test_query,
        "n_folders": int(folder_sizes.shape[0]),
        "n_singleton_folders": n_singletons,
        "n_pair_folders": n_pairs,
        "n_large_folders": n_large,
        "singletons_train": n_sing_train,
        "singletons_test": n_sing_test,
        "doubleton_folders_train": doubleton_train,
        "doubleton_folders_test": doubleton_test,
        "train_test_overlap": len(train_set & test_set),
    }
