from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote top screening configs to full retrieval eval (Phase 5)."
    )
    parser.add_argument("--screening_csv", required=True)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--repr", choices=["hidden"], default="hidden")
    parser.add_argument("--pooling", choices=["mean"], default="mean")
    parser.add_argument(
        "--output_csv",
        default="phase5_full_eval_scoreboard.csv",
        help="Output CSV path (default: alongside screening CSV).",
    )
    return parser.parse_args()


def run_eval(run_dir: Path, layer: int) -> None:
    import subprocess

    cmd = [
        "python",
        "src/eval_retrieval_cli.py",
        "--run_dir",
        str(run_dir),
        "--repr",
        "hidden",
        "--pooling",
        "mean",
        "--layers",
        str(layer),
    ]
    subprocess.run(cmd, check=True)


def summary_row(run_dir: Path, layer: int) -> Dict[str, object]:
    summary_path = run_dir / "hidden_layer_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    df = pd.read_csv(summary_path)
    row = df[df["layer"] == layer]
    if row.empty:
        raise ValueError(f"Layer {layer} not found in {summary_path}")
    return row.iloc[0].to_dict()


def append_row(path: Path, row: Dict[str, object]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    screening_path = Path(args.screening_csv)
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else screening_path.parent / "phase5_full_eval_scoreboard.csv"
    )

    df = pd.read_csv(screening_path)
    if df.empty:
        raise ValueError("Screening CSV is empty.")
    df = df.sort_values("acc@5_winnable", ascending=False)
    selected = df.drop_duplicates(subset=["derived_run_dir"]).head(args.top_n)

    for _, row in selected.iterrows():
        run_dir = Path(row["derived_run_dir"])
        run_eval(run_dir, args.layer)
        summary = summary_row(run_dir, args.layer)
        output_row = {
            "method": row.get("method", ""),
            "params": row.get("params", ""),
            "derived_run_dir": str(run_dir),
            **summary,
        }
        append_row(output_path, output_row)

    print(f"Full eval scoreboard: {output_path}")


if __name__ == "__main__":
    main()
