"""Deterministic 50/50 file-level train/test split for the canon dataset.

Split rules (v2 — professor-approved):
- Singletons (1 file): distributed randomly ~50/50 between train and test.
- Doubletons (2 files): whole folders assigned randomly 50/50.
  37 folders -> train, 37 folders -> test.  Both files stay together.
- Multi-file (>=3 files): within each size-n class, train_n is varied
  across folders using stratified uniform allocation over {1, ..., n-1}.
  The remainder is weighted toward values nearest n/2 so the class average
  stays ~50/50 while every class contributes same-folder positive pairs
  to the train split (no class is forced to all-singletons-in-train).

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

    # --- Multi-file (>=3): varied per-folder allocation within each size class ---
    # For each size-n class of K folders, distribute train_n values uniformly
    # over {1, ..., n-1} via stratified allocation, with the remainder going
    # to values nearest n/2 so the class average stays ~n/2. Determinism is
    # preserved via the shared rng. This avoids the pathological case where
    # every size-3 folder is forced to (1 train, 2 test) and contributes zero
    # same-folder positive pairs to the training split.
    multi_mask = meta["folder_size"] >= 3
    if multi_mask.any():
        multi_df = meta.loc[multi_mask, ["folder_id", "folder_size"]]
        train_n_by_folder: Dict[str, int] = {}
        for size_n in sorted(multi_df["folder_size"].unique().tolist()):
            class_folder_ids = np.sort(
                multi_df.loc[multi_df["folder_size"] == size_n, "folder_id"].unique()
            )
            K = len(class_folder_ids)
            allowed = list(range(1, int(size_n)))
            n_allowed = len(allowed)
            base, rem = divmod(K, n_allowed)
            counts = [base] * n_allowed
            mid = size_n / 2.0
            order = sorted(range(n_allowed), key=lambda i: (abs(allowed[i] - mid), i))
            for r in range(rem):
                counts[order[r]] += 1
            values: List[int] = []
            for val, cnt in zip(allowed, counts):
                values.extend([val] * cnt)
            shuffled_folders = rng.permutation(class_folder_ids)
            shuffled_values = rng.permutation(np.asarray(values, dtype=np.int32))
            for fid, tn in zip(shuffled_folders, shuffled_values):
                train_n_by_folder[str(fid)] = int(tn)

        for folder_id, group in meta.loc[multi_mask].groupby("folder_id"):
            idx = group.index.to_numpy()
            train_n = train_n_by_folder[str(folder_id)]
            shuffled = rng.permutation(idx)
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


def canon_taskb_query_reference_split(
    meta: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign taskb_role (query/reference/train) and has_reference_dir within test set.

    Prerequisite: meta must already have 'split' column from canon_train_test_split_v2().

    Rules applied per directory's test files (n_test = number of test files):
        n_test=1, singleton dir: 50% coin flip → query (match "none") or reference
        n_test=1, multi-file dir: query with has_reference_dir=False
        n_test=2: 1 query, 1 reference
        n_test=3: 1 query, 2 reference
        n_test>=4: n_test//2 queries, rest reference
    """
    rng = np.random.default_rng(random_seed)
    meta = meta.copy()

    taskb_role = np.full(len(meta), "train", dtype=object)
    has_reference_dir = np.zeros(len(meta), dtype=bool)

    test_mask = meta["split"] == "test"

    for folder_id, group in meta[test_mask].groupby("folder_id"):
        idx = group.index.to_numpy()
        folder_size = int(group["folder_size"].iloc[0])
        n_test = len(idx)

        if n_test == 0:
            continue

        shuffled = rng.permutation(idx)

        if n_test == 1 and folder_size == 1:
            # Singleton dir: 50% → query (match "none"), 50% → reference
            if rng.random() < 0.5:
                taskb_role[shuffled[0]] = "query"
                has_reference_dir[shuffled[0]] = False
            else:
                taskb_role[shuffled[0]] = "reference"
        elif n_test == 1:
            # Multi-file dir with only 1 file in test (edge case)
            taskb_role[shuffled[0]] = "query"
            has_reference_dir[shuffled[0]] = False
        elif n_test == 2:
            taskb_role[shuffled[0]] = "query"
            taskb_role[shuffled[1]] = "reference"
            has_reference_dir[shuffled[0]] = True
        elif n_test == 3:
            taskb_role[shuffled[0]] = "query"
            taskb_role[shuffled[1]] = "reference"
            taskb_role[shuffled[2]] = "reference"
            has_reference_dir[shuffled[0]] = True
        else:
            # 4+ test files: ~50% query, rest reference
            n_query = n_test // 2
            for qi in range(n_query):
                taskb_role[shuffled[qi]] = "query"
                has_reference_dir[shuffled[qi]] = True
            for ri in range(n_query, n_test):
                taskb_role[shuffled[ri]] = "reference"

    meta["taskb_role"] = taskb_role
    meta["has_reference_dir"] = has_reference_dir
    return meta


def taskb_split_summary(meta: pd.DataFrame) -> dict:
    """Summary statistics for the Task B query/reference split."""
    n_queries = int((meta["taskb_role"] == "query").sum())
    n_reference = int((meta["taskb_role"] == "reference").sum())
    n_train = int((meta["taskb_role"] == "train").sum())

    queries = meta[meta["taskb_role"] == "query"]
    n_queries_with_ref = int(queries["has_reference_dir"].sum())
    n_queries_none = n_queries - n_queries_with_ref

    ref_dirs = meta[meta["taskb_role"] == "reference"]["folder_id"].nunique()

    return {
        "n_queries": n_queries,
        "n_queries_with_reference": n_queries_with_ref,
        "n_queries_none": n_queries_none,
        "n_reference_files": n_reference,
        "n_reference_directories": ref_dirs,
        "n_train": n_train,
    }


def log_pair_distribution(meta: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Compute per-directory positive pair counts for a given split.

    Returns DataFrame with columns: folder_id, folder_size, n_in_split, n_positive_pairs.
    Also prints a summary grouped by folder_size.
    """
    split_df = meta[meta["split"] == split_name]
    rows = []
    for folder_id, group in split_df.groupby("folder_id"):
        n = len(group)
        folder_size = int(group["folder_size"].iloc[0])
        n_pairs = n * (n - 1) // 2
        rows.append({
            "folder_id": folder_id,
            "folder_size": folder_size,
            "n_in_split": n,
            "n_positive_pairs": n_pairs,
        })

    result = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["folder_id", "folder_size", "n_in_split", "n_positive_pairs"]
    )

    print(f"\nPositive pair distribution for '{split_name}' split:")
    print(f"{'Folder size':>12} {'Dirs':>5} {'Files in split':>15} {'Positive pairs':>15}")
    for fs, grp in result.groupby("folder_size"):
        print(f"{fs:>12} {len(grp):>5} {int(grp['n_in_split'].sum()):>15} "
              f"{int(grp['n_positive_pairs'].sum()):>15}")

    total_pairs = int(result["n_positive_pairs"].sum())
    print(f"{'TOTAL':>12} {len(result):>5} {int(result['n_in_split'].sum()):>15} "
          f"{total_pairs:>15}")

    return result


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

    # has_test_partner: True if folder has >= 2 files in the test set
    test_mask = df["split"] == "test"
    folder_test_counts = df[test_mask].groupby("folder_id").size()
    df["has_test_partner"] = df["folder_id"].map(
        lambda fid: folder_test_counts.get(fid, 0) >= 2
    )

    # Task B query/reference split within test set
    df = canon_taskb_query_reference_split(df, random_seed=random_seed)

    return df


def recompute_derived_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """Recompute every column that is a function of (folder_id, split, taskb_role).

    Used after a label correction moves files between directories: the assignments
    themselves are kept, only the directory-derived bookkeeping is refreshed.
    """
    meta = meta.copy()

    test_mask = meta["split"] == "test"
    folder_test_counts = meta[test_mask].groupby("folder_id").size()
    meta["has_test_partner"] = meta["folder_id"].map(
        lambda fid: folder_test_counts.get(fid, 0) >= 2
    )
    meta["is_test_query"] = test_mask & meta["has_test_partner"]

    ref_folders = set(
        meta.loc[meta["taskb_role"] == "reference", "folder_id"].unique()
    )
    meta["has_reference_dir"] = (meta["taskb_role"] == "query") & meta[
        "folder_id"
    ].isin(ref_folders)
    return meta


def build_meta_with_carried_over_split(
    canon_root: str,
    prior_split_csv: str,
) -> pd.DataFrame:
    """Rebuild meta over `canon_root` while keeping a previous split assignment.

    Every file keeps the `split` and `taskb_role` it had in `prior_split_csv`
    (matched on filename, which is unique across the corpus). Directory-derived
    columns (`folder_id`, `folder_size`, `is_singleton`, `is_winnable`, `file_id`,
    `has_test_partner`, `is_test_query`, `has_reference_dir`) are recomputed from
    the corpus as it stands on disk.

    Motivation: re-running `build_meta_with_split_v2` after a directory-level label
    correction changes folder-size classes, which changes how the shared RNG stream
    is consumed and reshuffles unrelated files. Carrying the assignment over keeps
    the diff confined to the corrected directories, so results computed on the
    previous split stay comparable.
    """
    entries = list_txt_files(canon_root)
    df = pd.DataFrame(entries, columns=["folder_id", "filename", "path"])
    folder_sizes = df.groupby("folder_id")["filename"].transform("count")
    df["folder_size"] = folder_sizes
    df["is_singleton"] = df["folder_size"] == 1
    df["is_winnable"] = df["folder_size"] >= 2
    df = df.reset_index(drop=True)
    df["file_id"] = np.arange(len(df), dtype=np.int32)

    prior = pd.read_csv(prior_split_csv)
    for col in ("filename", "split", "taskb_role"):
        if col not in prior.columns:
            raise ValueError(f"{prior_split_csv} has no '{col}' column")
    if prior["filename"].duplicated().any():
        raise ValueError(f"{prior_split_csv} has duplicate filenames")

    missing = set(df["filename"]) - set(prior["filename"])
    if missing:
        raise ValueError(
            f"{len(missing)} file(s) absent from {prior_split_csv}, e.g. "
            f"{sorted(missing)[:5]}. Carry-over only handles files that moved "
            "between directories, not added or removed files."
        )
    dropped = set(prior["filename"]) - set(df["filename"])
    if dropped:
        raise ValueError(
            f"{len(dropped)} file(s) in {prior_split_csv} are no longer on disk, "
            f"e.g. {sorted(dropped)[:5]}."
        )

    prior_idx = prior.set_index("filename")
    df["split"] = df["filename"].map(prior_idx["split"])
    df["taskb_role"] = df["filename"].map(prior_idx["taskb_role"])
    df = recompute_derived_columns(df)

    # Keep the column order of build_meta_with_split_v2 output.
    return df[
        [
            "folder_id",
            "filename",
            "path",
            "folder_size",
            "is_singleton",
            "is_winnable",
            "file_id",
            "split",
            "is_test_query",
            "has_test_partner",
            "taskb_role",
            "has_reference_dir",
        ]
    ]


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
