from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Phase 3 screening + full eval summaries."
    )
    parser.add_argument("--run_dir", required=True, help="Phase 3 run directory.")
    return parser.parse_args()


def parse_drop_percent(param: str) -> float:
    match = re.search(r"drop_lowest_percent=([0-9.]+)", param)
    return float(match.group(1)) if match else float("nan")


def parse_pca_k(param: str) -> int:
    match = re.search(r"pca_components=([0-9]+)", param)
    return int(match.group(1)) if match else -1


def ensure_out_dir(run_dir: Path) -> Path:
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_baseline(run_dir: Path) -> Dict[str, float]:
    baseline_path = run_dir / "phase3_baseline_summary.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["row"]


def plot_variance_vs_random(
    df: pd.DataFrame, baseline: Dict[str, float], out_dir: Path
) -> None:
    variance = df[df["method"] == "variance"].copy()
    random = df[df["method"] == "random"].copy()

    variance["drop"] = variance["params"].apply(parse_drop_percent)
    random["drop"] = random["params"].apply(parse_drop_percent)

    rand_group = (
        random.groupby("drop")
        .agg(
            dims_kept=("dims_kept", "first"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
            off_mean=("off_diag_mean", "mean"),
            off_std=("off_diag_mean", "std"),
        )
        .reset_index()
        .sort_values("dims_kept")
    )

    variance = variance.sort_values("dims_kept")

    # GAP vs dims
    plt.figure(figsize=(7, 4.2))
    plt.plot(variance["dims_kept"], variance["gap"], marker="o", label="variance")
    plt.errorbar(
        rand_group["dims_kept"],
        rand_group["gap_mean"],
        yerr=rand_group["gap_std"],
        fmt="o",
        capsize=3,
        label="random mean ± std",
    )
    plt.axhline(baseline["gap"], color="black", linestyle="--", label="baseline gap")
    plt.xlabel("dims kept")
    plt.ylabel("gap (same_avg - diff_avg)")
    plt.title("Variance vs Random: GAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gap_vs_dims_variance_random.png", dpi=160)
    plt.close()

    # off-diagonal mean vs dims
    plt.figure(figsize=(7, 4.2))
    plt.plot(
        variance["dims_kept"], variance["off_diag_mean"], marker="o", label="variance"
    )
    plt.errorbar(
        rand_group["dims_kept"],
        rand_group["off_mean"],
        yerr=rand_group["off_std"],
        fmt="o",
        capsize=3,
        label="random mean ± std",
    )
    plt.axhline(
        baseline["off_diag_mean"],
        color="black",
        linestyle="--",
        label="baseline off_diag",
    )
    plt.xlabel("dims kept")
    plt.ylabel("off-diagonal mean cosine")
    plt.title("Variance vs Random: off-diagonal mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "offdiag_vs_dims_variance_random.png", dpi=160)
    plt.close()


def plot_pca_full_eval(
    full_eval: pd.DataFrame, baseline: Dict[str, float], out_dir: Path
) -> None:
    pca = full_eval[full_eval["method"] == "pca"].copy()
    if pca.empty:
        return
    pca["k"] = pca["params"].apply(parse_pca_k)
    pca = pca.sort_values("k")

    # Acc@5_winnable vs k
    plt.figure(figsize=(7, 4.2))
    plt.plot(pca["k"], pca["acc@5_winnable"], marker="o", label="PCA acc@5_winnable")
    plt.axhline(
        baseline["acc@5_winnable"],
        color="black",
        linestyle="--",
        label="baseline acc@5_winnable",
    )
    plt.xlabel("PCA components (k)")
    plt.ylabel("acc@5_winnable")
    plt.title("PCA full eval: acc@5_winnable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_acc5_winnable.png", dpi=160)
    plt.close()

    # GAP vs k (from full eval)
    plt.figure(figsize=(7, 4.2))
    plt.plot(pca["k"], pca["gap"], marker="o", label="PCA gap")
    plt.axhline(
        baseline["gap"], color="black", linestyle="--", label="baseline gap"
    )
    plt.xlabel("PCA components (k)")
    plt.ylabel("gap (same_avg - diff_avg)")
    plt.title("PCA full eval: GAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_gap.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = ensure_out_dir(run_dir)

    screening = pd.read_csv(run_dir / "phase3_screening_scoreboard.csv")
    baseline = load_baseline(run_dir)
    full_eval_path = run_dir / "phase3_full_eval_scoreboard.csv"
    full_eval = pd.read_csv(full_eval_path) if full_eval_path.exists() else pd.DataFrame()

    plot_variance_vs_random(screening, baseline, out_dir)
    plot_pca_full_eval(full_eval, baseline, out_dir)

    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
