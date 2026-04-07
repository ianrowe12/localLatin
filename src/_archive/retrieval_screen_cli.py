from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd

from canon_retrieval import accuracy_at_k, sanity_checks, similarity_matrix, upper_triangle, upper_triangle_labels
from cli_utils import extract_layer_numbers, parse_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast retrieval screening metrics (no threshold sweep)."
    )
    parser.add_argument("--run_dir", required=True, help="Run directory with embeddings.")
    parser.add_argument("--repr", choices=["ff1", "hidden"], required=True)
    parser.add_argument("--pooling", choices=["mean", "lasttok", "sif"], default="mean")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--query_mask_col",
        default="",
        help="Optional boolean column in meta.csv to restrict queries.",
    )
    parser.add_argument(
        "--compute_margins",
        action="store_true",
        help="Compute margin summaries (best_same - best_diff).",
    )
    parser.add_argument(
        "--output_json",
        default="retrieval_screen.json",
        help="Output JSON filename (written under run_dir).",
    )
    return parser.parse_args()


def infer_layers(run_dir: Path, repr_name: str, pooling: str) -> List[int]:
    if pooling == "mean":
        suffix = ""
    elif pooling == "lasttok":
        suffix = "_lasttok"
    else:
        suffix = "_sif"
    pattern = rf"{repr_name}_layer(\\d+)_embeddings{suffix}_norm\\.npy$"
    paths = [p.as_posix() for p in run_dir.glob(f"{repr_name}_layer*_embeddings{suffix}_norm.npy")]
    return extract_layer_numbers(paths, pattern)


def embedding_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    if pooling == "mean":
        suffix = ""
    elif pooling == "lasttok":
        suffix = "_lasttok"
    else:
        suffix = "_sif"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}_norm.npy"


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
    }


def compute_margins(
    sim: np.ndarray, folder_ids: np.ndarray, query_mask: np.ndarray
) -> np.ndarray:
    n = sim.shape[0]
    margins = np.full(n, np.nan)
    for i in range(n):
        if not query_mask[i]:
            continue
        scores = sim[i].copy()
        scores[i] = -np.inf
        same_mask = folder_ids == folder_ids[i]
        same_mask[i] = False
        diff_mask = ~same_mask
        diff_mask[i] = False
        best_same = float(np.max(scores[same_mask])) if np.any(same_mask) else float("nan")
        best_diff = float(np.max(scores[diff_mask])) if np.any(diff_mask) else float("nan")
        if np.isfinite(best_same) and np.isfinite(best_diff):
            margins[i] = best_same - best_diff
    return margins


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = pd.read_csv(run_dir / "meta.csv")
    folder_ids = meta["folder_id"].to_numpy()
    is_winnable = meta["is_winnable"].to_numpy(dtype=bool)

    if args.query_mask_col:
        if args.query_mask_col not in meta.columns:
            raise ValueError(f"Missing query mask column: {args.query_mask_col}")
        query_mask = meta[args.query_mask_col].to_numpy(dtype=bool)
        query_mask_col = args.query_mask_col
    else:
        query_mask = np.ones(len(meta), dtype=bool)
        query_mask_col = ""

    layers = parse_layers(str(args.layer)) if str(args.layer) else []
    if not layers:
        layers = infer_layers(run_dir, args.repr, args.pooling)
    if not layers:
        raise FileNotFoundError("No embeddings found for given repr/pooling.")
    layer = layers[0]

    emb_path = embedding_path(run_dir, args.repr, args.pooling, layer)
    emb = np.load(emb_path)

    start = perf_counter()
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
    same_avg = float(sim_upper[labels].mean()) if labels.any() else float("nan")
    diff_avg = float(sim_upper[~labels].mean()) if (~labels).any() else float("nan")
    gap = same_avg - diff_avg

    margin_summary = {}
    if args.compute_margins:
        margin_query_mask = query_mask & is_winnable
        margins = compute_margins(sim, folder_ids, margin_query_mask)
        margin_summary = {
            "margin_query_mask": "query_mask & is_winnable",
            "margin_summary": summarize_array(margins),
        }

    elapsed = perf_counter() - start

    n_all = int(np.sum(query_mask))
    n_winnable = int(np.sum(query_mask & is_winnable))
    winnable_fraction = n_winnable / n_all if n_all else 0.0

    payload = {
        "run_dir": str(run_dir),
        "repr": args.repr,
        "pooling": args.pooling,
        "layer": int(layer),
        "embeddings_path": emb_path.name,
        "num_files": int(emb.shape[0]),
        "num_dims": int(emb.shape[1]),
        "query_mask_col": query_mask_col,
        "n_all": n_all,
        "n_winnable": n_winnable,
        "winnable_fraction": winnable_fraction,
        "acc@1_all": acc1_all,
        "acc@3_all": acc3_all,
        "acc@5_all": acc5_all,
        "acc@1_winnable": acc1_win,
        "acc@3_winnable": acc3_win,
        "acc@5_winnable": acc5_win,
        "same_avg": same_avg,
        "diff_avg": diff_avg,
        "gap": gap,
        "diag_mean": checks["diag_mean"],
        "off_diag_mean": checks["off_diag_mean"],
        "elapsed_seconds": elapsed,
        **margin_summary,
    }

    out_path = run_dir / args.output_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved retrieval screen: {out_path}")


if __name__ == "__main__":
    main()
