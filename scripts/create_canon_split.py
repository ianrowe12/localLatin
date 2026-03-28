"""Create and lock a deterministic canon train/test split.

Usage:
    python scripts/create_canon_split.py \
      --canon_root canon/ \
      --output_csv runs/phase8_results/meta_split.csv \
      --test_fraction 0.2 \
      --random_seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_split import build_meta_with_split, save_split, split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canon train/test split.")
    parser.add_argument("--canon_root", required=True, help="Path to canon/ directory.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path for locked split.")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = build_meta_with_split(
        canon_root=args.canon_root,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )

    save_split(meta, str(output_path))

    summary = split_summary(meta)
    summary_path = output_path.parent / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Split saved: {output_path}")
    print(f"Summary saved: {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
