"""Phase 12c Step 1: Select (query, positive_partner, negative_partner) triples.

Reads phase9_split.csv and picks test-set queries that have at least one
test-set partner in the same folder. For each query, selects one positive
partner (same folder) and one negative partner (different folder, test set).

Usage:
  python scripts/select_retrieval_pairs.py \
    --split_csv runs/phase9/phase9_split.csv \
    --out_csv runs/phase12c/retrieval_pairs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Select retrieval pairs for Phase 12c.")
    p.add_argument("--split_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    df = pd.read_csv(args.split_csv)
    test_df = df[df["split"] == "test"].copy()

    # Find test queries with at least one test partner in same folder
    queries = test_df[
        (test_df["is_winnable"] == True) &
        (test_df["has_test_partner"] == True) &
        (test_df["is_test_query"] == True)
    ].copy()

    print(f"Total test files: {len(test_df)}")
    print(f"Eligible test queries (is_test_query & has_test_partner): {len(queries)}")

    # Build folder → test file_ids map
    folder_to_test_ids = test_df.groupby("folder_id")["file_id"].apply(list).to_dict()

    # All test file_ids for negative sampling
    all_test_ids = set(test_df["file_id"].values)
    id_to_folder = dict(zip(test_df["file_id"], test_df["folder_id"]))
    id_to_path = dict(zip(test_df["file_id"], test_df["path"]))

    rows = []
    for _, qrow in queries.iterrows():
        q_fid = qrow["file_id"]
        q_folder = qrow["folder_id"]

        # Positive partner: same folder, test set, different file
        same_folder_ids = [fid for fid in folder_to_test_ids.get(q_folder, [])
                           if fid != q_fid]
        if not same_folder_ids:
            continue
        pos_id = rng.choice(same_folder_ids)

        # Negative partner: different folder, test set
        neg_candidates = [fid for fid in all_test_ids if id_to_folder[fid] != q_folder]
        neg_id = rng.choice(neg_candidates)

        rows.append({
            "query_file_id": q_fid,
            "query_path": qrow["path"],
            "query_folder_id": q_folder,
            "pos_partner_file_id": pos_id,
            "pos_partner_path": id_to_path[pos_id],
            "neg_partner_file_id": neg_id,
            "neg_partner_path": id_to_path[neg_id],
            "neg_partner_folder_id": id_to_folder[neg_id],
        })

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\nSelected {len(out_df)} query-partner triples")
    print(f"Unique query folders: {out_df['query_folder_id'].nunique()}")
    print(f"Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
