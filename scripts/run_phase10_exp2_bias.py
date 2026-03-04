"""Phase 10 Experiment 2: Bias / Residual Analysis.

For each (model, language, method) at the best layer, load cached pair_sims
from the evaluate step, compute residuals (predicted cosine - GT/5), and
aggregate by resource level (high vs low).

Output: bias_results.csv
  columns: model, language, resource_level, method, layer,
           mean_residual, std_residual, frac_overestimated, n_pairs
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────

RESOURCE = {
    "english": "high",
    "french": "high",
    "spanish": "high",
    "sinhala": "low",
    "tamil": "low",
    "serbian": "low",
}

# method → stem of the pair_sims filename (without layer prefix and _pair_sims.npy suffix)
METHOD_SIMS = {
    "baseline_mean": "baseline_mean",
    "sif_abtt_optimal": "sif_abtt_optimal",
    "whitening": "whitening",
}


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10 Exp 2 bias analysis.")
    parser.add_argument("--results_csv", required=True,
                        help="Phase 10 results.csv (from evaluate step).")
    parser.add_argument("--data_dir", required=True,
                        help="Dir with {lang}_split.csv files.")
    parser.add_argument("--cache_dir", required=True,
                        help="Embedding cache root (contains {slug}/{lang}/ subdirs).")
    parser.add_argument("--out_dir", required=True,
                        help="Output dir for bias_results.csv.")
    parser.add_argument(
        "--languages",
        default="english,french,spanish,sinhala,tamil,serbian",
        help="Comma-separated language names.",
    )
    parser.add_argument(
        "--methods",
        default="baseline_mean,sif_abtt_optimal,whitening",
        help="Comma-separated method names.",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────

def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_test_labels(data_dir: Path, language: str) -> np.ndarray:
    """Load similarity scores for test split (raw 0-5 scale)."""
    df = pd.read_csv(data_dir / f"{language}_split.csv")
    test = df[df["split"] == "test"]
    return test["similarity"].values.astype(np.float32)


def append_row(path: Path, row: Dict) -> None:
    fieldnames = [
        "model", "language", "resource_level", "method", "layer",
        "mean_residual", "std_residual", "frac_overestimated", "n_pairs",
    ]
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    results_df = pd.read_csv(args.results_csv)
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    bias_path = out_dir / "bias_results.csv"

    for model_name in results_df["model"].unique():
        slug = model_slug(model_name)
        print(f"\nModel: {model_name}")

        for language in languages:
            if language not in results_df["language"].values:
                print(f"  [{language}] not in results, skipping.")
                continue

            try:
                test_labels = load_test_labels(data_dir, language)
            except FileNotFoundError:
                print(f"  WARNING: {language}_split.csv not found, skipping.")
                continue

            for method in methods:
                if method not in METHOD_SIMS:
                    print(f"  WARNING: unknown method {method}, skipping.")
                    continue

                sub = results_df[
                    (results_df["model"] == model_name) &
                    (results_df["language"] == language) &
                    (results_df["method"] == method)
                ]
                if sub.empty:
                    print(f"  [{language}/{method}] no rows in results, skipping.")
                    continue

                best_layer = int(sub.loc[sub["spearman_test"].idxmax(), "layer"])

                # Load pair sims from evaluate cache
                sims_path = (
                    cache_dir / slug / language
                    / f"layer{best_layer}_{METHOD_SIMS[method]}_pair_sims.npy"
                )
                if not sims_path.exists():
                    print(f"  WARNING: {sims_path} not found, skipping.")
                    continue

                pair_sims = np.load(sims_path)
                n = min(len(pair_sims), len(test_labels))
                gt_norm = test_labels[:n] / 5.0
                residuals = pair_sims[:n] - gt_norm

                resource_level = RESOURCE.get(language, "unknown")

                append_row(bias_path, {
                    "model": model_name,
                    "language": language,
                    "resource_level": resource_level,
                    "method": method,
                    "layer": best_layer,
                    "mean_residual": float(residuals.mean()),
                    "std_residual": float(residuals.std()),
                    "frac_overestimated": float((residuals > 0).mean()),
                    "n_pairs": n,
                })
                print(
                    f"  [{language}/{method}] layer={best_layer}  "
                    f"mean_res={residuals.mean():.4f}  "
                    f"frac_over={float((residuals > 0).mean()):.3f}  "
                    f"n={n}"
                )

    print(f"\nBias results saved: {bias_path}")


if __name__ == "__main__":
    main()
