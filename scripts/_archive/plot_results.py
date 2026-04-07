from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_acc_by_layer(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["layer"], df["acc@1_winnable"], label="Acc@1 (winnable)")
    ax.plot(df["layer"], df["acc@3_winnable"], label="Acc@3 (winnable)")
    ax.plot(df["layer"], df["acc@5_winnable"], label="Acc@5 (winnable)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy@K (winnable)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap_offdiag(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["layer"], df["gap"], color="tab:blue", label="Gap (same - diff)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Gap", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["layer"], df["off_diag_mean"], color="tab:orange", label="Off-diag mean")
    ax2.set_ylabel("Off-diag mean", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bucket_heatmap(
    df: pd.DataFrame, value_col: str, title: str, out_path: Path
) -> None:
    pivot = df.pivot_table(
        index="layer", columns="bucket_id", values=value_col, aggfunc="mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Bucket ID (short → long)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label=value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bucket_lines(
    df: pd.DataFrame, value_col: str, title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for layer in sorted(df["layer"].unique()):
        subset = df[df["layer"] == layer].sort_values("bucket_id")
        ax.plot(subset["bucket_id"], subset[value_col], label=f"Layer {layer}")
    ax.set_xlabel("Bucket ID (short → long)")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retrieval summaries.")
    parser.add_argument("--run_dir", required=True, help="Run directory with CSV outputs.")
    parser.add_argument("--out_dir", default="", help="Output directory for plots.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ff1_mean = load_csv(run_dir / "ff1_layer_summary.csv")
    ff1_lasttok = load_csv(run_dir / "ff1_lasttok_layer_summary.csv")
    hidden_mean = load_csv(run_dir / "hidden_layer_summary.csv")
    hidden_lasttok = load_csv(run_dir / "hidden_lasttok_layer_summary.csv")

    if ff1_mean is not None:
        plot_acc_by_layer(ff1_mean, "FF1 mean pooling", out_dir / "ff1_mean_acc.png")
        plot_gap_offdiag(
            ff1_mean, "FF1 mean pooling: gap vs off-diag", out_dir / "ff1_mean_gap.png"
        )

    if ff1_lasttok is not None:
        plot_acc_by_layer(
            ff1_lasttok, "FF1 last-token pooling", out_dir / "ff1_lasttok_acc.png"
        )
        plot_gap_offdiag(
            ff1_lasttok,
            "FF1 last-token pooling: gap vs off-diag",
            out_dir / "ff1_lasttok_gap.png",
        )

    if hidden_mean is not None:
        plot_acc_by_layer(
            hidden_mean, "Hidden states mean pooling", out_dir / "hidden_mean_acc.png"
        )
        plot_gap_offdiag(
            hidden_mean,
            "Hidden states mean pooling: gap vs off-diag",
            out_dir / "hidden_mean_gap.png",
        )

    if hidden_lasttok is not None:
        plot_acc_by_layer(
            hidden_lasttok,
            "Hidden states last-token pooling",
            out_dir / "hidden_lasttok_acc.png",
        )
        plot_gap_offdiag(
            hidden_lasttok,
            "Hidden states last-token pooling: gap vs off-diag",
            out_dir / "hidden_lasttok_gap.png",
        )

    bucket_ff1 = load_csv(run_dir / "bucket_summary_ff1_mean.csv")
    if bucket_ff1 is not None:
        plot_bucket_heatmap(
            bucket_ff1,
            "acc@5_winnable",
            "FF1 mean: acc@5_winnable by length bucket",
            out_dir / "ff1_mean_bucket_acc5_heatmap.png",
        )

    bucket_hidden = load_csv(run_dir / "bucket_summary_hidden_mean.csv")
    if bucket_hidden is not None:
        plot_bucket_heatmap(
            bucket_hidden,
            "acc@5_winnable",
            "Hidden mean: acc@5_winnable by length bucket",
            out_dir / "hidden_mean_bucket_acc5_heatmap.png",
        )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
