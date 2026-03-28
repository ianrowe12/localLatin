from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from canon_retrieval import (
    accuracy_at_k,
    sanity_checks,
    similarity_matrix,
    sweep_thresholds,
    upper_triangle,
    upper_triangle_labels,
)
from cli_utils import extract_layer_numbers, parse_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics for embeddings.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing embeddings.")
    parser.add_argument("--repr", choices=["ff1", "hidden"], required=True)
    parser.add_argument("--pooling", choices=["mean", "lasttok"], default="mean")
    parser.add_argument("--layers", default="", help="Layer list, e.g. 1-12 or 1,3,5.")
    parser.add_argument("--threshold_bins", type=int, default=400)
    parser.add_argument(
        "--query_mask_col",
        default="",
        help="Optional boolean column in meta.csv to restrict queries (e.g. is_eval_query).",
    )
    return parser.parse_args()


def infer_layers(run_dir: Path, repr_name: str, pooling: str) -> List[int]:
    suffix = "" if pooling == "mean" else "_lasttok"
    pattern = rf"{repr_name}_layer(\\d+)_embeddings{suffix}_norm\\.npy$"
    paths = [p.as_posix() for p in run_dir.glob(f"{repr_name}_layer*_embeddings{suffix}_norm.npy")]
    return extract_layer_numbers(paths, pattern)


def embedding_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}_norm.npy"


def summary_filename(repr_name: str, pooling: str) -> str:
    if pooling == "mean":
        return f"{repr_name}_layer_summary.csv"
    return f"{repr_name}_lasttok_layer_summary.csv"


def threshold_filename(repr_name: str, pooling: str, layer: int) -> str:
    return f"threshold_curve_{repr_name}_{pooling}_layer{layer}.csv"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = pd.read_csv(run_dir / "meta.csv")
    folder_ids = meta["folder_id"].tolist()
    is_winnable = meta["is_winnable"].to_numpy(dtype=bool)
    if args.query_mask_col:
        if args.query_mask_col not in meta.columns:
            raise ValueError(f"Missing query mask column: {args.query_mask_col}")
        query_mask = meta[args.query_mask_col].to_numpy(dtype=bool)
    else:
        query_mask = np.ones(len(meta), dtype=bool)
    n_all = int(np.sum(query_mask))
    n_winnable = int(np.sum(query_mask & is_winnable))
    winnable_fraction = n_winnable / n_all if n_all else 0.0
    singleton_ceiling = winnable_fraction

    layers = parse_layers(args.layers)
    if not layers:
        layers = infer_layers(run_dir, args.repr, args.pooling)
    if not layers:
        raise FileNotFoundError("No embeddings found for given repr/pooling.")

    layer_rows: List[Dict[str, float]] = []
    thresholds = None

    for layer in layers:
        emb_path = embedding_path(run_dir, args.repr, args.pooling, layer)
        emb = np.load(emb_path)
        sim = similarity_matrix(emb)
        checks = sanity_checks(sim)

        acc1_all = accuracy_at_k(sim, folder_ids, query_mask, k=1)
        acc3_all = accuracy_at_k(sim, folder_ids, query_mask, k=3)
        acc5_all = accuracy_at_k(sim, folder_ids, query_mask, k=5)
        acc1_win = accuracy_at_k(sim, folder_ids, query_mask & is_winnable, k=1)
        acc3_win = accuracy_at_k(sim, folder_ids, query_mask & is_winnable, k=3)
        acc5_win = accuracy_at_k(sim, folder_ids, query_mask & is_winnable, k=5)

        sim_upper = upper_triangle(sim)
        labels = upper_triangle_labels(folder_ids)
        if thresholds is None:
            thresholds = np.linspace(sim_upper.min(), sim_upper.max(), args.threshold_bins)
        curve = sweep_thresholds(sim_upper, labels, thresholds)
        curve["layer"] = layer
        curve.to_csv(run_dir / threshold_filename(args.repr, args.pooling, layer), index=False)

        same_avg = float(sim_upper[labels].mean()) if labels.any() else float("nan")
        diff_avg = float(sim_upper[~labels].mean()) if (~labels).any() else float("nan")
        gap = same_avg - diff_avg
        best_idx = curve["f1"].idxmax()
        best_row = curve.loc[best_idx]

        layer_rows.append(
            {
                "layer": layer,
                "acc@1_all": acc1_all,
                "acc@3_all": acc3_all,
                "acc@5_all": acc5_all,
                "acc@1_winnable": acc1_win,
                "acc@3_winnable": acc3_win,
                "acc@5_winnable": acc5_win,
                "winnable_fraction": winnable_fraction,
                "n_all": n_all,
                "n_winnable": n_winnable,
                "singleton_ceiling": singleton_ceiling,
                "query_mask_col": args.query_mask_col or "",
                "same_avg": same_avg,
                "diff_avg": diff_avg,
                "gap": gap,
                "best_threshold": float(best_row["threshold"]),
                "best_precision": float(best_row["precision"]),
                "best_recall": float(best_row["recall"]),
                "best_f1": float(best_row["f1"]),
                "diag_mean": checks["diag_mean"],
                "off_diag_mean": checks["off_diag_mean"],
            }
        )

    summary = pd.DataFrame(layer_rows)
    summary_path = run_dir / summary_filename(args.repr, args.pooling)
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
