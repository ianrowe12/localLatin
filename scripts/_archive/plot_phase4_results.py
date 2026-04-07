from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 4 hard-query results.")
    parser.add_argument("--phase4_run_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional output dir (default: <phase4_run_dir>/figures).",
    )
    return parser.parse_args()


def bar_positions(n_groups: int, n_series: int, width: float = 0.2) -> List[np.ndarray]:
    base = np.arange(n_groups)
    offsets = np.linspace(
        -width * (n_series - 1) / 2, width * (n_series - 1) / 2, n_series
    )
    return [base + off for off in offsets]


def plot_recovery_bars(pivot: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    k_cols = [c for c in pivot.columns if c.startswith("net_gain_rate@")]
    ks = [int(c.split("@")[1]) for c in k_cols]
    ks_sorted = [k for _, k in sorted(zip(ks, ks))]
    k_cols = [f"net_gain_rate@{k}" for k in ks_sorted]

    series_labels = pivot["compare_label"].tolist()
    positions = bar_positions(len(k_cols), len(series_labels), width=0.22)

    plt.figure(figsize=(7, 4))
    for idx, label in enumerate(series_labels):
        values = pivot.loc[pivot["compare_label"] == label, k_cols].iloc[0].to_numpy()
        plt.bar(positions[idx], values, width=0.22, label=label)
    plt.xticks(np.arange(len(k_cols)), [f"@{k}" for k in ks_sorted])
    plt.ylabel("net gain rate (recovered - regressed)")
    plt.xlabel("K")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "net_gain_rate_by_k.png", dpi=160)
    plt.close()


def plot_hit_rates(pivot: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    k_cols = [c for c in pivot.columns if c.startswith("baseline_hit_rate@")]
    ks = [int(c.split("@")[1]) for c in k_cols]
    ks_sorted = [k for _, k in sorted(zip(ks, ks))]

    plt.figure(figsize=(7, 4))
    for _, row in pivot.iterrows():
        compare = row["compare_label"]
        baseline = [row[f"baseline_hit_rate@{k}"] for k in ks_sorted]
        compare_rates = [row[f"compare_hit_rate@{k}"] for k in ks_sorted]
        plt.plot(ks_sorted, baseline, marker="o", label="baseline")
        plt.plot(ks_sorted, compare_rates, marker="o", label=compare)
        plt.xlabel("K")
        plt.ylabel("hit rate")
        plt.xticks(ks_sorted)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"hit_rate_{compare}.png", dpi=160)
        plt.close()


def plot_margin_delta_hist(hard: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    delta_cols = [c for c in hard.columns if c.endswith("_margin_delta")]
    if not delta_cols:
        return
    for col in delta_cols:
        label = col.replace("_margin_delta", "")
        values = hard[col].to_numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=60, alpha=0.8)
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.xlabel("margin delta (compare - baseline)")
        plt.ylabel("count")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(out_dir / f"margin_delta_hist_{label}.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.phase4_run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Phase 4 run dir not found: {run_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pivot_path = run_dir / "recovery_regression_pivot.csv"
    hard_path = run_dir / "hard_queries.csv"

    if not pivot_path.exists():
        raise FileNotFoundError(f"Missing pivot CSV: {pivot_path}")
    pivot = pd.read_csv(pivot_path)
    plot_recovery_bars(pivot, out_dir)
    plot_hit_rates(pivot, out_dir)

    if hard_path.exists():
        hard = pd.read_csv(hard_path)
        plot_margin_delta_hist(hard, out_dir)

    print(f"Phase 4 plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
