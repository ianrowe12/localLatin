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
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics by token-length buckets.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing embeddings.")
    parser.add_argument("--repr", choices=["ff1", "hidden"], required=True)
    parser.add_argument("--pooling", choices=["mean", "lasttok"], default="mean")
    parser.add_argument("--layers", default="", help="Layer list, e.g. 1-12 or 1,3,5.")
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--token_length_col", default="token_length")
    parser.add_argument("--threshold_bins", type=int, default=200)
    parser.add_argument("--write_threshold_curves", action="store_true")
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = pd.read_csv(run_dir / "meta.csv")
    if args.token_length_col not in meta.columns:
        raise ValueError(f"Missing token length column: {args.token_length_col}")

    folder_ids = meta["folder_id"].tolist()
    is_winnable = meta["is_winnable"].to_numpy(dtype=bool)
    if args.query_mask_col:
        if args.query_mask_col not in meta.columns:
            raise ValueError(f"Missing query mask column: {args.query_mask_col}")
        base_query_mask = meta[args.query_mask_col].to_numpy(dtype=bool)
    else:
        base_query_mask = np.ones(len(meta), dtype=bool)
    lengths = meta[args.token_length_col]
    buckets = pd.qcut(lengths, q=args.num_buckets, duplicates="drop")
    meta = meta.assign(bucket=buckets)

    layers = parse_layers(args.layers)
    if not layers:
        layers = infer_layers(run_dir, args.repr, args.pooling)
    if not layers:
        raise FileNotFoundError("No embeddings found for given repr/pooling.")

    bucket_rows: List[Dict[str, float]] = []
    bucket_categories = list(meta["bucket"].cat.categories)

    for layer in layers:
        emb = np.load(embedding_path(run_dir, args.repr, args.pooling, layer))
        sim = similarity_matrix(emb)
        checks = sanity_checks(sim)
        sim_upper = upper_triangle(sim)
        labels = upper_triangle_labels(folder_ids)
        upper_i, upper_j = np.triu_indices(sim.shape[0], k=1)

        for bucket_id, bucket_range in enumerate(bucket_categories):
            bucket_mask = (meta["bucket"] == bucket_range).to_numpy()
            query_mask_all = bucket_mask & base_query_mask
            query_mask_win = bucket_mask & base_query_mask & is_winnable

            acc1_all = accuracy_at_k(sim, folder_ids, query_mask_all, k=1)
            acc3_all = accuracy_at_k(sim, folder_ids, query_mask_all, k=3)
            acc5_all = accuracy_at_k(sim, folder_ids, query_mask_all, k=5)
            acc1_win = accuracy_at_k(sim, folder_ids, query_mask_win, k=1)
            acc3_win = accuracy_at_k(sim, folder_ids, query_mask_win, k=3)
            acc5_win = accuracy_at_k(sim, folder_ids, query_mask_win, k=5)

            pair_mask = bucket_mask[upper_i] & bucket_mask[upper_j]
            sim_bucket = sim_upper[pair_mask]
            labels_bucket = labels[pair_mask]
            same_avg = float(sim_bucket[labels_bucket].mean()) if labels_bucket.any() else float("nan")
            diff_avg = float(sim_bucket[~labels_bucket].mean()) if (~labels_bucket).any() else float("nan")
            gap = same_avg - diff_avg
            off_diag_mean = float(sim_bucket.mean()) if len(sim_bucket) else float("nan")

            best_threshold = float("nan")
            best_precision = float("nan")
            best_recall = float("nan")
            best_f1 = float("nan")

            if args.write_threshold_curves and len(sim_bucket) > 0:
                thresholds = np.linspace(sim_bucket.min(), sim_bucket.max(), args.threshold_bins)
                curve = sweep_thresholds(sim_bucket, labels_bucket, thresholds)
                curve["layer"] = layer
                curve["bucket_id"] = bucket_id
                curve.to_csv(
                    run_dir / f"threshold_curve_{args.repr}_{args.pooling}_layer{layer}_bucket{bucket_id}.csv",
                    index=False,
                )
                best_idx = curve["f1"].idxmax()
                best_row = curve.loc[best_idx]
                best_threshold = float(best_row["threshold"])
                best_precision = float(best_row["precision"])
                best_recall = float(best_row["recall"])
                best_f1 = float(best_row["f1"])

            bucket_rows.append(
                {
                    "layer": layer,
                    "bucket_id": bucket_id,
                    "bucket_range": str(bucket_range),
                    "bucket_count": int(bucket_mask.sum()),
                    "bucket_winnable": int(query_mask_win.sum()),
                    "acc@1_all": acc1_all,
                    "acc@3_all": acc3_all,
                    "acc@5_all": acc5_all,
                    "acc@1_winnable": acc1_win,
                    "acc@3_winnable": acc3_win,
                    "acc@5_winnable": acc5_win,
                    "query_mask_col": args.query_mask_col or "",
                    "same_avg": same_avg,
                    "diff_avg": diff_avg,
                    "gap": gap,
                    "best_threshold": best_threshold,
                    "best_precision": best_precision,
                    "best_recall": best_recall,
                    "best_f1": best_f1,
                    "diag_mean": checks["diag_mean"],
                    "off_diag_mean": off_diag_mean,
                }
            )

    summary = pd.DataFrame(bucket_rows)
    summary_path = run_dir / f"bucket_summary_{args.repr}_{args.pooling}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved bucket summary: {summary_path}")


if __name__ == "__main__":
    main()
