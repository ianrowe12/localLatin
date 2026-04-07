"""Combine per-model phase12d filtering CSVs into one aggregate table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-model phase12d filtering result CSVs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing filtering_results_*.csv files.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to the combined CSV to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    paths = sorted(input_dir.glob("filtering_results_*.csv"))
    if not paths:
        raise SystemExit(f"No phase12d CSVs found in {input_dir}")

    dfs = [pd.read_csv(path) for path in paths]
    combined = pd.concat(dfs, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"Combined {len(paths)} CSVs -> {output_csv}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
