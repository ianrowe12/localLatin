from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from canon_retrieval import (
    sanity_checks,
    similarity_matrix,
    upper_triangle,
    upper_triangle_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast GAP screening (no threshold sweep)."
    )
    parser.add_argument("--run_dir", required=True, help="Run directory with embeddings.")
    parser.add_argument("--repr", choices=["ff1", "hidden"], required=True)
    parser.add_argument("--pooling", choices=["mean", "lasttok"], default="mean")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--output_json",
        default="gap_screen.json",
        help="Output JSON filename (written under run_dir).",
    )
    return parser.parse_args()


def embedding_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}_norm.npy"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = pd.read_csv(run_dir / "meta.csv")
    folder_ids = meta["folder_id"].tolist()

    emb_path = embedding_path(run_dir, args.repr, args.pooling, args.layer)
    emb = np.load(emb_path)

    start = perf_counter()
    sim = similarity_matrix(emb)
    checks = sanity_checks(sim)
    sim_upper = upper_triangle(sim)
    labels = upper_triangle_labels(folder_ids)
    same_avg = float(sim_upper[labels].mean()) if labels.any() else float("nan")
    diff_avg = float(sim_upper[~labels].mean()) if (~labels).any() else float("nan")
    gap = same_avg - diff_avg
    elapsed = perf_counter() - start

    payload = {
        "run_dir": str(run_dir),
        "repr": args.repr,
        "pooling": args.pooling,
        "layer": args.layer,
        "embeddings_path": emb_path.name,
        "num_files": int(emb.shape[0]),
        "num_dims": int(emb.shape[1]),
        "same_avg": same_avg,
        "diff_avg": diff_avg,
        "gap": gap,
        "diag_mean": checks["diag_mean"],
        "off_diag_mean": checks["off_diag_mean"],
        "elapsed_seconds": elapsed,
    }

    out_path = run_dir / args.output_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved GAP screen: {out_path}")


if __name__ == "__main__":
    main()
