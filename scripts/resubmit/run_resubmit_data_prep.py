"""Resubmit data preparation: 50/50 train/test split for canon_labelled dataset.

Generates:
  phase_resubmit_split.csv  — file-level split assignment (compatible with --split_csv)
  train_task_a.tsv          — all train pairs (file_1, file_2, semantically_equal, comparison_id)
  test_task_a.tsv           — all test pairs (same schema)
  train_task_b.tsv          — file-level (file_id, content, directory_id, semantically_equal_ids)
  test_task_b.tsv           — same schema
  resubmit_summary.json     — dataset + split + pair statistics
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from canon_retrieval import load_text
from canon_split_v2 import (
    build_meta_with_split_v2,
    generate_pairs_tsv,
    log_pair_distribution,
    split_summary_v2,
    taskb_split_summary,
)


def build_task_b_tsv(meta, split_name, output_path):
    """Generate file-level TSV for a split.

    Columns: file_id, content, directory_id, semantically_equal_ids
    """
    subset = meta[meta["split"] == split_name]
    folder_groups = subset.groupby("folder_id")["file_id"].apply(list).to_dict()

    n_rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file_id", "content", "directory_id", "semantically_equal_ids"])
        for _, row in subset.iterrows():
            fid = row["file_id"]
            content = " ".join(load_text(row["path"]).split())
            dir_id = " ".join(row["folder_id"].split())
            partners = [str(x) for x in folder_groups[row["folder_id"]] if x != fid]
            writer.writerow([fid, content, dir_id, ",".join(partners)])
            n_rows += 1

    return n_rows


def main():
    parser = argparse.ArgumentParser(
        description="Resubmit data prep: 50/50 split + Task A/B TSVs"
    )
    parser.add_argument(
        "--canon_root", default="canon_labelled",
        help="Root directory of the labelled canon dataset.",
    )
    parser.add_argument(
        "--out_dir", default="runs/active/resubmit/data",
        help="Output directory for split files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Build metadata + apply v2 split ---
    print("Building metadata with v2 split...")
    meta = build_meta_with_split_v2(args.canon_root, random_seed=args.seed)
    print(f"  {len(meta)} files in {meta['folder_id'].nunique()} folders")

    # --- Step 2: Save split CSV ---
    split_csv_path = out / "phase_resubmit_split.csv"
    meta.to_csv(split_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"  Saved {split_csv_path}")

    # --- Step 3: Split summary ---
    summary = split_summary_v2(meta)
    print(f"  Train: {summary['n_train']}, Test: {summary['n_test']}")
    print(f"  Folders: {summary['n_singleton_folders']} singleton, "
          f"{summary['n_pair_folders']} doubleton, "
          f"{summary['n_large_folders']} multi-file")
    print(f"  Singletons: {summary['singletons_train']} train / {summary['singletons_test']} test")
    print(f"  Doubleton folders: {summary['doubleton_folders_train']} train / "
          f"{summary['doubleton_folders_test']} test")
    print(f"  Test queries: {summary['n_test_query']}")
    print(f"  Overlap: {summary['train_test_overlap']}")

    # --- Step 3b: Task B query/reference summary ---
    tb_summary = taskb_split_summary(meta)
    print(f"\n  Task B split:")
    print(f"    Queries: {tb_summary['n_queries']} "
          f"({tb_summary['n_queries_with_reference']} with reference, "
          f"{tb_summary['n_queries_none']} expecting 'none')")
    print(f"    Reference directories: {tb_summary['n_reference_directories']}")
    print(f"    Reference files: {tb_summary['n_reference_files']}")

    # --- Step 4: Task A — pairwise TSVs ---
    print("\nGenerating Task A (pairwise) TSVs...")
    print("  train_task_a.tsv (this may take a moment)...")
    train_a_info = generate_pairs_tsv(meta, "train", str(out / "train_task_a.tsv"))
    print(f"    {train_a_info['n_pairs']} pairs "
          f"({train_a_info['n_positive']} pos, {train_a_info['n_negative']} neg)")

    print("  test_task_a.tsv...")
    test_a_info = generate_pairs_tsv(meta, "test", str(out / "test_task_a.tsv"))
    print(f"    {test_a_info['n_pairs']} pairs "
          f"({test_a_info['n_positive']} pos, {test_a_info['n_negative']} neg)")

    # --- Step 4b: Positive pair distribution (diagnostic) ---
    print("\n--- Train pair distribution ---")
    train_pair_dist = log_pair_distribution(meta, "train")
    train_pair_dist.to_csv(out / "train_pair_distribution.csv", index=False)

    print("\n--- Test pair distribution ---")
    test_pair_dist = log_pair_distribution(meta, "test")
    test_pair_dist.to_csv(out / "test_pair_distribution.csv", index=False)

    # --- Step 5: Task B — file-level TSVs ---
    print("\nGenerating Task B (file-level) TSVs...")
    train_b_count = build_task_b_tsv(meta, "train", out / "train_task_b.tsv")
    print(f"  train_task_b.tsv: {train_b_count} rows")

    test_b_count = build_task_b_tsv(meta, "test", out / "test_task_b.tsv")
    print(f"  test_task_b.tsv: {test_b_count} rows")

    # --- Step 6: Combined summary JSON ---
    full_summary = {
        "dataset": {
            "canon_root": args.canon_root,
            "seed": args.seed,
        },
        "split": summary,
        "task_a_pairs": {
            "train": {
                "n_files": train_a_info["n_files"],
                "n_pairs": train_a_info["n_pairs"],
                "n_positive": train_a_info["n_positive"],
                "n_negative": train_a_info["n_negative"],
            },
            "test": {
                "n_files": test_a_info["n_files"],
                "n_pairs": test_a_info["n_pairs"],
                "n_positive": test_a_info["n_positive"],
                "n_negative": test_a_info["n_negative"],
            },
        },
        "task_b_files": {
            "train": train_b_count,
            "test": test_b_count,
        },
        "task_b_split": tb_summary,
    }
    summary_path = out / "resubmit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nSaved {summary_path}")

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
