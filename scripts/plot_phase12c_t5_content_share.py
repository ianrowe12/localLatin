from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TARGETS = [
    ("LaTa", 8, "LaTa\nL8 mid dip"),
    ("PhilTa", 6, "PhilTa\nL6 worst"),
]

COLORS = {
    "Baseline": "#bdbdbd",
    "ABTT": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a simple T5 content-token share figure for the paper."
    )
    parser.add_argument("--csv", required=True, help="Path to analysis4_importance_share.csv")
    parser.add_argument("--out_dir", required=True, help="Directory for output figure files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, layer, display_label in TARGETS:
        sub = df[
            (df["model"] == model)
            & (df["layer"] == layer)
            & (df["category"] == "content")
        ]
        if sub.empty:
            raise ValueError(f"Missing content-share row for {model} layer {layer}")
        row = sub.iloc[0]
        rows.append(
            {
                "label": display_label,
                "baseline": float(row["share_baseline"]),
                "abtt": float(row["share_abtt"]),
            }
        )

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = range(len(plot_df))
    width = 0.34

    baseline_bars = ax.bar(
        [i - width / 2 for i in x],
        plot_df["baseline"],
        width=width,
        color=COLORS["Baseline"],
        label="Baseline",
    )
    abtt_bars = ax.bar(
        [i + width / 2 for i in x],
        plot_df["abtt"],
        width=width,
        color=COLORS["ABTT"],
        label="ABTT",
    )

    for bars in (baseline_bars, abtt_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Content-token share")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylim(0.0, 0.70)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = out_dir / "fig_t5_content_share.png"
    pdf_path = out_dir / "fig_t5_content_share.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
