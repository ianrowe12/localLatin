"""Render the paper-facing Phase 11 release figures.

This release view keeps only the four methods discussed in the revised paper:
baseline, ABTT, SIF, and whitening.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "google/mt5-base": "mt5-base",
}

MAX_LAYERS = {
    "bowphs/LaTa": 12,
    "bowphs/PhilTa": 12,
    "sentence-transformers/LaBSE": 12,
    "Qwen/Qwen3-Embedding-0.6B": 28,
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": 24,
    "google/mt5-base": 12,
}

METHOD_ORDER = ["baseline", "abtt_optimal", "sif_only", "whitening"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "abtt_optimal": "ABTT",
    "sif_only": "SIF",
    "whitening": "Whitening",
}
METHOD_COLORS = {
    "baseline": "#8a8a8a",
    "abtt_optimal": "#1f77b4",
    "sif_only": "#e67e22",
    "whitening": "#2ca6a4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Phase 11 release figures.")
    parser.add_argument("--results_csv", required=True, help="Phase 11 results CSV.")
    parser.add_argument("--out_dir", required=True, help="Figure output directory.")
    parser.add_argument(
        "--repr_name",
        default="hidden",
        help="Representation to plot (paper default: hidden).",
    )
    return parser.parse_args()


def normalize_layer_depth(model_name: str, layer: pd.Series) -> pd.Series:
    max_layers = MAX_LAYERS.get(model_name, int(layer.max()))
    return layer.astype(float) / float(max_layers) * 100.0


def filter_release_rows(results: pd.DataFrame, repr_name: str) -> pd.DataFrame:
    keep_rows = []
    for method in METHOD_ORDER:
        sub = results[(results["repr"] == repr_name) & (results["method"] == method)].copy()
        if method in {"baseline", "abtt_optimal", "whitening"}:
            sub = sub[sub["pooling"] == "mean"]
        elif method == "sif_only":
            sub = sub[sub["pooling"] == "sif"]
        keep_rows.append(sub)
    if not keep_rows:
        return pd.DataFrame()
    return pd.concat(keep_rows, ignore_index=True)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric(results: pd.DataFrame, metric: str, ylabel: str, out_dir: Path, stem: str) -> None:
    model_order = list(SHORT.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=False, sharey=(metric == "aucroc"))
    axes = axes.ravel()

    if metric == "gap":
        finite_values = results[metric].replace([np.inf, -np.inf], np.nan).dropna()
        y_min = float(finite_values.min()) if not finite_values.empty else -0.1
        y_max = float(finite_values.max()) if not finite_values.empty else 0.8
        pad = max(0.03, (y_max - y_min) * 0.08)
        y_limits = (y_min - pad, y_max + pad)
    else:
        y_limits = (0.3, 1.02)

    handles = []
    labels = []
    for ax, model_name in zip(axes, model_order):
        model_rows = results[results["model"] == model_name].copy()
        for method in METHOD_ORDER:
            method_rows = model_rows[model_rows["method"] == method].sort_values("layer")
            if method_rows.empty:
                continue
            x = normalize_layer_depth(model_name, method_rows["layer"])
            line = ax.plot(
                x,
                method_rows[metric],
                color=METHOD_COLORS[method],
                marker="o",
                markersize=4.5,
                linewidth=2.3,
                label=METHOD_LABELS[method],
            )[0]
            if METHOD_LABELS[method] not in labels:
                handles.append(line)
                labels.append(METHOD_LABELS[method])

        ax.set_title(SHORT.get(model_name, model_name), fontsize=15, fontweight="bold")
        ax.set_xlabel("Layer Depth (%)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(0, 100)
        ax.set_ylim(*y_limits)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.25, linewidth=0.8)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results_csv)
    release_rows = filter_release_rows(results, args.repr_name)
    if release_rows.empty:
        raise SystemExit("No release rows matched the requested representation.")

    plot_metric(
        release_rows,
        metric="aucroc",
        ylabel="AUCROC",
        out_dir=out_dir,
        stem="fig_release_aucroc_per_model",
    )
    plot_metric(
        release_rows,
        metric="gap",
        ylabel="Cosine Gap",
        out_dir=out_dir,
        stem="fig_release_gap_per_model",
    )
    print(f"Saved release figures to {out_dir}")


if __name__ == "__main__":
    main()
