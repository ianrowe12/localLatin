"""Render resubmit figures: AUCROC per model, Gap per model, density 2x2.

Combines the release line plots from visualize_phase11_release.py with the
density histogram from package_paper_release_assets.py.
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
    parser = argparse.ArgumentParser(description="Visualize resubmit figures.")
    parser.add_argument("--results_csv", required=True, help="Phase resubmit results CSV.")
    parser.add_argument("--dist_dir", required=True, help="Distribution .npz directory.")
    parser.add_argument("--out_dir", required=True, help="Figure output directory.")
    parser.add_argument(
        "--repr_name",
        default="hidden",
        help="Representation to plot (paper default: hidden).",
    )
    parser.add_argument(
        "--feature_model",
        default="bowphs/PhilTa",
        help="Featured model for density 2x2 figure.",
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


def plot_density_2x2(feature_model: str, dist_dir: Path, out_dir: Path) -> None:
    slug = feature_model.replace("/", "_")
    conds = [
        ("baseline_last", "Baseline - Last Layer"),
        ("baseline_middle", "Baseline - Dip Layer"),
        ("abtt_last", "ABTT - Last Layer"),
        ("abtt_middle", "ABTT - Dip Layer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    bins = np.linspace(-0.2, 1.0, 80)
    row_ymax = [0.0, 0.0]

    for idx, (suffix, title) in enumerate(conds):
        ax = axes[idx]
        npz_path = dist_dir / f"{slug}_{suffix}.npz"
        if not npz_path.exists():
            ax.axis("off")
            ax.text(0.5, 0.5, f"Missing {npz_path.name}", ha="center", va="center")
            continue

        data = np.load(npz_path)
        same = data["same_sims"]
        diff = data["diff_sims"]
        ax.hist(
            diff,
            bins=bins,
            alpha=0.5,
            color="#c0392b",
            density=True,
            edgecolor="none",
            label="Different",
        )
        ax.hist(
            same,
            bins=bins,
            alpha=0.5,
            color="#2471a3",
            density=True,
            edgecolor="none",
            label="Same",
        )
        same_mean = float(np.mean(same))
        diff_mean = float(np.mean(diff))
        ax.axvline(same_mean, color="#2471a3", linestyle="--", linewidth=1.3)
        ax.axvline(diff_mean, color="#c0392b", linestyle="--", linewidth=1.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.text(
            0.98,
            0.95,
            f"gap={same_mean - diff_mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        row = idx // 2
        row_ymax[row] = max(row_ymax[row], ax.get_ylim()[1])

    for row in range(2):
        ymax = row_ymax[row] * 1.05 if row_ymax[row] > 0 else 1.0
        for col in range(2):
            axes[row * 2 + col].set_ylim(0, ymax)

    axes[0].set_ylabel("Density", fontsize=12)
    axes[2].set_ylabel("Density", fontsize=12)
    axes[2].set_xlabel("Cosine Similarity", fontsize=12)
    axes[3].set_xlabel("Cosine Similarity", fontsize=12)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    save(fig, out_dir, "paper_fig_density_2x2")


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

    dist_dir = Path(args.dist_dir)
    if dist_dir.exists():
        plot_density_2x2(args.feature_model, dist_dir, out_dir)
        print(f"Saved density 2x2 to {out_dir}")
    else:
        print(f"Warning: dist_dir {dist_dir} not found, skipping density figure.")

    print(f"Saved release figures to {out_dir}")


if __name__ == "__main__":
    main()
