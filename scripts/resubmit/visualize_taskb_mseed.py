"""Visualize M-seed Task B results: tables, bar charts, and breakdowns.

Reads aggregated_results.csv from run_taskb_mseed.py and produces:
  1. LaTeX table + PNG/PDF with mean +/- std for Top-K accuracy
  2. Grouped bar chart comparing models at K=1,2,3,5
  3. Breakdown of existing vs new vs overall assignment accuracy
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
    "google/mt5-base": "mT5-base",
}

MODEL_ORDER = list(SHORT.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize M-seed Task B results.")
    parser.add_argument("--agg_csv", required=True, help="aggregated_results.csv from run_taskb_mseed.py")
    parser.add_argument("--out_dir", required=True, help="Output directory for figures.")
    parser.add_argument("--repr_name", default="hidden", help="Representation to select.")
    parser.add_argument("--top_k", type=int, default=5, help="Max K for top-K reporting.")
    return parser.parse_args()


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def select_best_configs(agg_df: pd.DataFrame, repr_name: str) -> pd.DataFrame:
    """For each model, pick the config with highest dir_acc_at_1_mean."""
    rows = []
    for model in MODEL_ORDER:
        sub = agg_df[(agg_df["model"] == model) & (agg_df["repr"] == repr_name)]
        if sub.empty:
            continue
        best_idx = sub["dir_acc_at_1_mean"].idxmax()
        rows.append(sub.loc[best_idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def write_latex_table(best_df: pd.DataFrame, out_dir: Path, ks: list[int]) -> None:
    col_spec = "l" + "r" * len(ks)
    header_cells = " & ".join(f"\\textbf{{Top-{k}}}" for k in ks)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Model}} & {header_cells} \\\\",
        "\\midrule",
    ]

    # Find best per column for bolding
    best_per_k = {}
    for k in ks:
        col = f"dir_acc_at_{k}_mean"
        if col in best_df.columns:
            best_per_k[k] = best_df[col].max()

    for _, row in best_df.iterrows():
        name = SHORT.get(row["model"], row["model"])
        cells = []
        for k in ks:
            mean_col = f"dir_acc_at_{k}_mean"
            std_col = f"dir_acc_at_{k}_std"
            mean_val = row.get(mean_col, 0) * 100
            std_val = row.get(std_col, 0) * 100
            cell = f"{mean_val:.1f} $\\pm$ {std_val:.1f}"
            if k in best_per_k and abs(row.get(mean_col, 0) - best_per_k[k]) < 1e-9:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{name} & {' & '.join(cells)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    (out_dir / "taskb_mseed_table.tex").write_text("\n".join(lines) + "\n")


def render_table_figure(best_df: pd.DataFrame, out_dir: Path, ks: list[int]) -> None:
    col_labels = ["Model"] + [f"Top-{k}" for k in ks]
    cell_data = []
    means_top1 = []

    for _, row in best_df.iterrows():
        name = SHORT.get(row["model"], row["model"])
        cells = [name]
        for k in ks:
            mean_val = row.get(f"dir_acc_at_{k}_mean", 0) * 100
            std_val = row.get(f"dir_acc_at_{k}_std", 0) * 100
            cells.append(f"{mean_val:.1f} \u00b1 {std_val:.1f}")
        cell_data.append(cells)
        means_top1.append(row.get("dir_acc_at_1_mean", 0))

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1f4e79")
        table[0, j].set_text_props(color="white", fontweight="bold")

    if means_top1:
        best_row = int(np.argmax(means_top1))
        table[best_row + 1, 1].set_facecolor("#d4edda")

    fig.tight_layout()
    save(fig, out_dir, "taskb_mseed_table")


def plot_grouped_bar_topk(best_df: pd.DataFrame, out_dir: Path, ks: list[int]) -> None:
    models = [SHORT.get(row["model"], row["model"]) for _, row in best_df.iterrows()]
    n_models = len(models)
    n_k = len(ks)
    x = np.arange(n_models)
    width = 0.8 / n_k

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(ks):
        means = best_df[f"dir_acc_at_{k}_mean"].values * 100
        stds = best_df[f"dir_acc_at_{k}_std"].values * 100
        offset = (i - n_k / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=f"Top-{k}", color=colors[i % len(colors)], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Task B: Directory Accuracy at Top-K (M=5 seeds)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    ymin = max(0, best_df["dir_acc_at_1_mean"].min() * 100 - 15)
    ax.set_ylim(ymin, 105)

    fig.tight_layout()
    save(fig, out_dir, "taskb_mseed_topk_bar")


def plot_breakdown(best_df: pd.DataFrame, out_dir: Path) -> None:
    models = [SHORT.get(row["model"], row["model"]) for _, row in best_df.iterrows()]
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.25

    metrics = [
        ("existing_acc", "Existing Docs", "#2ca02c"),
        ("new_acc", "New Docs", "#ff7f0e"),
        ("overall_assignment_acc", "Overall", "#1f77b4"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label, color) in enumerate(metrics):
        means = best_df[f"{metric}_mean"].values * 100
        stds = best_df[f"{metric}_std"].values * 100
        offset = (i - 1) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=label, color=color, alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Task B: Assignment Accuracy Breakdown (M=5 seeds)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    all_means = []
    for metric, _, _ in metrics:
        all_means.extend(best_df[f"{metric}_mean"].values * 100)
    ymin = max(0, min(all_means) - 15) if all_means else 0
    ax.set_ylim(ymin, 105)

    fig.tight_layout()
    save(fig, out_dir, "taskb_mseed_breakdown")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_df = pd.read_csv(args.agg_csv)
    best_df = select_best_configs(agg_df, args.repr_name)

    if best_df.empty:
        raise SystemExit("No rows matched the requested representation.")

    best_df.to_csv(out_dir / "taskb_mseed_selected_configs.csv", index=False)
    print(f"Selected best configs for {len(best_df)} models")

    available_ks = [k for k in range(1, args.top_k + 1) if f"dir_acc_at_{k}_mean" in agg_df.columns]
    ks = [k for k in [1, 2, 3, 5] if k in available_ks]

    write_latex_table(best_df, out_dir, ks)
    render_table_figure(best_df, out_dir, ks)
    plot_grouped_bar_topk(best_df, out_dir, ks)
    plot_breakdown(best_df, out_dir)

    print(f"Saved LaTeX table to {out_dir / 'taskb_mseed_table.tex'}")
    print(f"Saved table figure to {out_dir / 'taskb_mseed_table.png'}")
    print(f"Saved bar chart to {out_dir / 'taskb_mseed_topk_bar.png'}")
    print(f"Saved breakdown to {out_dir / 'taskb_mseed_breakdown.png'}")


if __name__ == "__main__":
    main()
