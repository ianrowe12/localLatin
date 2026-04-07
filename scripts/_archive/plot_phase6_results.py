from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRANSFORM_ORDER = ["baseline", "variance_drop_25", "best_phase5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 6 summary figures.")
    parser.add_argument("--scoreboard_csv", required=True)
    parser.add_argument("--out_dir", default="")
    return parser.parse_args()


def normalize_transform(name: str) -> str:
    if name.startswith("variance_drop_"):
        return "variance_drop_25"
    return name


def prep_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["transform"] = df["transform"].map(normalize_transform)
    df["model"] = df["model"].str.replace("bowphs/", "", regex=False)
    df["target"] = df["model"] + " " + df["repr"]
    return df


def plot_acc_by_layer(df: pd.DataFrame, out_dir: Path) -> None:
    targets = [
        ("LaTa", "hidden"),
        ("LaTa", "ff1"),
        ("PhilTa", "hidden"),
        ("PhilTa", "ff1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (model, repr_name) in zip(axes.flatten(), targets):
        subset = df[(df["model"] == model) & (df["repr"] == repr_name)]
        if subset.empty:
            ax.axis("off")
            continue
        for transform in TRANSFORM_ORDER:
            sub = subset[subset["transform"] == transform].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(
                sub["layer"],
                sub["acc@5_winnable"],
                marker="o",
                label=transform.replace("_", " "),
            )
        ax.set_title(f"{model} {repr_name}")
        ax.set_xlabel("layer")
        ax.set_ylabel("acc@5_winnable")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase6_acc5_winnable_by_layer.png", dpi=160)
    plt.close(fig)


def plot_delta_bars(df: pd.DataFrame, out_dir: Path) -> None:
    base = df[df["transform"] == "baseline"][
        ["model", "repr", "layer", "acc@5_winnable"]
    ].rename(columns={"acc@5_winnable": "baseline"})
    merged = df.merge(base, on=["model", "repr", "layer"], how="left")
    merged["delta_acc@5_winnable"] = merged["acc@5_winnable"] - merged["baseline"]
    merged = merged[merged["transform"] != "baseline"]
    merged["label"] = (
        merged["model"] + "-" + merged["repr"] + "-L" + merged["layer"].astype(str)
    )
    merged = merged.sort_values(["model", "repr", "layer", "transform"])

    fig, ax = plt.subplots(figsize=(12, 5))
    for transform in ["variance_drop_25", "best_phase5"]:
        sub = merged[merged["transform"] == transform]
        ax.plot(
            sub["label"],
            sub["delta_acc@5_winnable"],
            marker="o",
            linestyle="-",
            label=transform.replace("_", " "),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Delta acc@5_winnable vs baseline")
    ax.set_ylabel("delta acc@5_winnable")
    ax.set_xlabel("target")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase6_delta_acc5_winnable.png", dpi=160)
    plt.close(fig)


def plot_gap_offdiag(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for transform in TRANSFORM_ORDER:
        sub = df[df["transform"] == transform]
        if sub.empty:
            continue
        ax.scatter(
            sub["off_diag_mean"],
            sub["gap"],
            label=transform.replace("_", " "),
            alpha=0.8,
        )
    ax.set_title("GAP vs off-diagonal mean")
    ax.set_xlabel("off_diag_mean")
    ax.set_ylabel("gap")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase6_gap_vs_offdiag.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scoreboard_path = Path(args.scoreboard_csv)
    out_dir = Path(args.out_dir) if args.out_dir else scoreboard_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prep_df(scoreboard_path)
    plot_acc_by_layer(df, out_dir)
    plot_delta_bars(df, out_dir)
    plot_gap_offdiag(df, out_dir)
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
