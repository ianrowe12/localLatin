"""Task 1: Data Preparation — generate canon split + pair TSVs.

Outputs:
  meta_split_v2.csv   — file-level split assignment
  train.tsv           — all train pairs with text content
  test.tsv            — all test pairs with text content
  split_summary.json  — statistics
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from canon_split_v2 import (
    build_meta_with_split_v2,
    generate_pairs_tsv,
    split_summary_v2,
)


def main():
    parser = argparse.ArgumentParser(description="Generate canon 50/50 split + pair TSVs")
    parser.add_argument("--canon_root", required=True, help="Path to canon/ directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Building meta with v2 split...")
    meta = build_meta_with_split_v2(args.canon_root, random_seed=args.random_seed)

    # Save meta CSV
    meta_path = out / "meta_split_v2.csv"
    meta.to_csv(meta_path, index=False)
    print(f"  Saved {meta_path}")

    # Summary
    summary = split_summary_v2(meta)
    summary_path = out / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")
    print(f"  Train: {summary['n_train']}, Test: {summary['n_test']}")
    print(f"  Singletons: {summary['singletons_train']} train / {summary['singletons_test']} test")
    print(f"  Doubleton folders: {summary['doubleton_folders_train']} train / {summary['doubleton_folders_test']} test")
    print(f"  Overlap: {summary['train_test_overlap']}")

    # Generate pair TSVs
    print("\nGenerating train pairs TSV (this may take a moment)...")
    train_info = generate_pairs_tsv(meta, "train", str(out / "train.tsv"))
    print(f"  Train: {train_info['n_pairs']} pairs ({train_info['n_positive']} pos, {train_info['n_negative']} neg)")

    print("Generating test pairs TSV...")
    test_info = generate_pairs_tsv(meta, "test", str(out / "test.tsv"))
    print(f"  Test: {test_info['n_pairs']} pairs ({test_info['n_positive']} pos, {test_info['n_negative']} neg)")

    # Save pair info
    pair_info = {"train": train_info, "test": test_info}
    with open(out / "pair_summary.json", "w") as f:
        json.dump(pair_info, f, indent=2)

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
