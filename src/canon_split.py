"""Deterministic file-level train/test split for the canon dataset.

Split rules:
- Folders with >=3 files: proportional split, at least 2 train per folder.
- Folders with exactly 2 files: both assigned to test (preserves the only
  testable positive pair).
- Singletons (1 file): assigned to train (can only form negatives).

Test query files search against ALL files (full gallery). The split only
governs which files' statistics are used for fitting (train) vs which
files serve as evaluation queries (test).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from canon_retrieval import list_txt_files


def canon_train_test_split(
    meta: pd.DataFrame,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Add 'split' and 'is_test_query' columns to meta DataFrame.

    Parameters
    ----------
    meta : DataFrame with columns folder_id, filename, path, folder_size,
           is_singleton, is_winnable, file_id.
    test_fraction : target fraction of files to hold out as test per folder.
    random_seed : seed for reproducible splits.

    Returns
    -------
    meta with two new columns:
        split : 'train' or 'test'
        is_test_query : True for test files with at least one same-folder
                        partner (i.e. they can be meaningfully queried).
    """
    rng = np.random.default_rng(random_seed)
    meta = meta.copy()
    split = np.full(len(meta), "train", dtype=object)

    for folder_id, group in meta.groupby("folder_id"):
        idx = group.index.to_numpy()
        n = len(idx)

        if n == 1:
            # Singleton -> train (can only form negatives)
            split[idx] = "train"
        elif n == 2:
            # Pair-only -> both test (preserves only testable positive pair)
            split[idx] = "test"
        else:
            # >=3 files: proportional split, at least 2 train
            test_n = max(int(round(n * test_fraction)), 1)
            # Ensure at least 2 train files
            test_n = min(test_n, n - 2)
            test_n = max(test_n, 1)
            shuffled = rng.permutation(idx)
            test_idx = shuffled[:test_n]
            train_idx = shuffled[test_n:]
            split[test_idx] = "test"
            split[train_idx] = "train"

    meta["split"] = split

    # is_test_query: test files that have at least one same-folder partner
    # (needed to be a meaningful retrieval query)
    test_mask = meta["split"] == "test"
    folder_test_counts = meta[test_mask].groupby("folder_id").size()
    folder_total = meta.groupby("folder_id").size()

    is_test_query = np.zeros(len(meta), dtype=bool)
    for i, row in meta.iterrows():
        if row["split"] != "test":
            continue
        fid = row["folder_id"]
        # Query is meaningful if there's at least one other file in the same folder
        if folder_total[fid] >= 2:
            is_test_query[i] = True

    meta["is_test_query"] = is_test_query
    return meta


def build_meta_with_split(
    canon_root: str,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build meta DataFrame from canon directory and add split columns."""
    entries = list_txt_files(canon_root)
    df = pd.DataFrame(entries, columns=["folder_id", "filename", "path"])
    folder_sizes = df.groupby("folder_id")["filename"].transform("count")
    df["folder_size"] = folder_sizes
    df["is_singleton"] = df["folder_size"] == 1
    df["is_winnable"] = df["folder_size"] >= 2
    df = df.reset_index(drop=True)
    df["file_id"] = np.arange(len(df), dtype=np.int32)
    df = canon_train_test_split(df, test_fraction=test_fraction, random_seed=random_seed)
    return df


def save_split(meta: pd.DataFrame, output_path: str) -> None:
    """Save split metadata to CSV."""
    meta.to_csv(output_path, index=False)


def load_split(path: str) -> pd.DataFrame:
    """Load split metadata from CSV."""
    return pd.read_csv(path)


def split_summary(meta: pd.DataFrame) -> dict:
    """Compute summary statistics for a split."""
    total = len(meta)
    n_train = int((meta["split"] == "train").sum())
    n_test = int((meta["split"] == "test").sum())
    n_test_query = int(meta["is_test_query"].sum())

    folder_sizes = meta.groupby("folder_id")["filename"].count()
    n_singletons = int((folder_sizes == 1).sum())
    n_pairs = int((folder_sizes == 2).sum())
    n_large = int((folder_sizes >= 3).sum())
    n_folders = int(folder_sizes.shape[0])

    # Verify no overlap
    train_set = set(meta.loc[meta["split"] == "train", "file_id"])
    test_set = set(meta.loc[meta["split"] == "test", "file_id"])
    overlap = train_set & test_set

    # Verify every test query has a same-folder partner
    test_queries = meta[meta["is_test_query"]]
    queryable = True
    for _, row in test_queries.iterrows():
        fid = row["folder_id"]
        partners = meta[meta["folder_id"] == fid]
        if len(partners) < 2:
            queryable = False
            break

    return {
        "total_files": total,
        "n_train": n_train,
        "n_test": n_test,
        "n_test_query": n_test_query,
        "n_folders": n_folders,
        "n_singleton_folders": n_singletons,
        "n_pair_folders": n_pairs,
        "n_large_folders": n_large,
        "train_test_overlap": len(overlap),
        "all_test_queries_have_partner": queryable,
    }
