from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "bowphs/LaTa",
    "bowphs/PhilTa",
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    "google/mt5-base",
]

SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "google/mt5-base": "mt5-base",
}

CONDITION_ORDER = ["baseline", "filter_no_empty", "abtt_D10"]
CONDITION_LABELS = {
    "baseline": "Mean pooling",
    "filter_no_empty": "Standard preprocessing",
    "abtt_D10": "ABTT",
}
CONDITION_COLORS = {
    "baseline": "#bdbdbd",
    "filter_no_empty": "#fdae6b",
    "abtt_D10": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 12d filtering results.")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharey=True)
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for idx_ax, (ax, model_name) in enumerate(zip(axes, models)):
        sub = df[df["model"] == model_name].copy()
        layers = sorted(sub["layer"].unique())
        x = np.arange(len(layers))
        width = 0.22

        for idx, condition in enumerate(CONDITION_ORDER):
            vals = []
            for layer in layers:
                row = sub[(sub["layer"] == layer) & (sub["condition"] == condition)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            offset = (idx - 1.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=CONDITION_LABELS[condition],
                color=CONDITION_COLORS[condition],
                edgecolor="white",
                linewidth=0.5,
            )
            if idx_ax == 0:
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                legend_handles.append(bars[0])
                legend_labels.append(CONDITION_LABELS[condition])

        ax.set_title(SHORT.get(model_name, model_name), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{layer}" for layer in layers], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Layer")
        if metric == "aucroc":
            ax.set_ylim(0.4, 1.0)
        else:
            ax.set_ylim(0.0, 1.0)

    ylabel = "AUCROC" if metric == "aucroc" else "Acc@1"
    axes[0].set_ylabel(ylabel)
    axes[3].set_ylabel(ylabel)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(df, "aucroc", out_dir / "fig_filtering_vs_abtt.png")
    plot_metric(df, "acc_at_1", out_dir / "fig_filtering_vs_abtt_acc1.png")


if __name__ == "__main__":
    main()
