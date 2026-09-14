"""Assert that every sampled pair has a canonical NPZ carrying every method.

``build_attribution_run_manifest.py --require_complete`` cannot be used for the
issue #141 run: it also demands the metrics ``summary.csv``, and that file comes
from the ``--backend model`` metrics pass, which costs GPU hours the paper does
not need (the tables read ``summary_v2.csv`` from the CPU hidden-backend pass).
This checks the artifact side of completeness on its own, so the GPU job still
fails loudly when a pair is missing or lost its MaRC sidecar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

DEFAULT_METHODS = ("ig", "retrieval_mark")
VARIANTS = ("baseline", "abtt")


def artifact_path(artifacts_root: Path, model_name: str, example_id: int) -> Path:
    slug = model_name.replace("/", "_")
    return artifacts_root / slug / f"example{example_id:03d}_pair_example.npz"


def verify(
    examples_csv: Path,
    artifacts_root: Path,
    methods: Sequence[str],
) -> List[str]:
    """Return one problem string per pair that is missing or incomplete."""
    examples = pd.read_csv(examples_csv)
    problems: List[str] = []
    for row in examples.itertuples():
        path = artifact_path(artifacts_root, str(row.model_name), int(row.example_id))
        if not path.exists():
            problems.append(f"missing artifact: {path}")
            continue
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
        for method in methods:
            for variant in VARIANTS:
                key = f"pair_matrix_{method}_{variant}"
                if key not in keys:
                    problems.append(f"{path.name} ({row.model_name}): no {key}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_csv", required=True, type=Path)
    parser.add_argument("--artifacts_root", required=True, type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Attribution methods whose pair matrices must be present.",
    )
    parser.add_argument(
        "--max_report",
        type=int,
        default=10,
        help="How many problems to print before truncating.",
    )
    args = parser.parse_args()

    problems = verify(args.examples_csv, args.artifacts_root, args.methods)
    n_rows = len(pd.read_csv(args.examples_csv))
    print(f"pairs={n_rows} problems={len(problems)}")
    if problems:
        for line in problems[: args.max_report]:
            print(f"  {line}")
        if len(problems) > args.max_report:
            print(f"  ... {len(problems) - args.max_report} more")
        sys.exit(1)
    print(f"OK: {n_rows} pairs, every artifact carries {', '.join(args.methods)}")


if __name__ == "__main__":
    main()
