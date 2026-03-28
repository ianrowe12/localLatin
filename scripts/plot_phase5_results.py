from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Phase 5 screening + full eval summaries."
    )
    parser.add_argument("--run_dir", required=True, help="Phase 5 run directory.")
    return parser.parse_args()


def ensure_out_dir(run_dir: Path) -> Path:
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_baseline(run_dir: Path) -> Dict[str, float]:
    baseline_path = run_dir / "phase5_baseline_summary.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["row"]


def plot_screen_scatter(
    screening: pd.DataFrame, baseline: Dict[str, float], out_dir: Path
) -> None:
    if screening.empty:
        return

    plt.figure(figsize=(7, 4.2))
    plt.scatter(screening["gap"], screening["acc@5_winnable"], alpha=0.7)
    plt.scatter(
        [baseline["gap"]],
        [baseline["acc@5_winnable"]],
        marker="*",
        s=150,
        label="baseline",
    )
    plt.xlabel("gap (same_avg - diff_avg)")
    plt.ylabel("acc@5_winnable")
    plt.title("Phase 5 screening: GAP vs acc@5_winnable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "screen_gap_vs_acc5.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.scatter(screening["off_diag_mean"], screening["acc@5_winnable"], alpha=0.7)
    plt.scatter(
        [baseline["off_diag_mean"]],
        [baseline["acc@5_winnable"]],
        marker="*",
        s=150,
        label="baseline",
    )
    plt.xlabel("off_diag_mean")
    plt.ylabel("acc@5_winnable")
    plt.title("Phase 5 screening: off_diag_mean vs acc@5_winnable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "screen_offdiag_vs_acc5.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.scatter(screening["dims_kept"], screening["acc@5_winnable"], alpha=0.7)
    plt.axhline(
        baseline["acc@5_winnable"], color="black", linestyle="--", label="baseline"
    )
    plt.xlabel("dims kept")
    plt.ylabel("acc@5_winnable")
    plt.title("Phase 5 screening: dims kept vs acc@5_winnable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "screen_dims_vs_acc5.png", dpi=160)
    plt.close()


def plot_full_eval_bars(
    full_eval: pd.DataFrame, baseline: Dict[str, float], out_dir: Path
) -> None:
    if full_eval.empty:
        return

    full_eval = full_eval.copy()
    full_eval["label"] = (
        full_eval["method"].astype(str) + ":" + full_eval["params"].fillna("")
    )
    full_eval = full_eval.sort_values("acc@5_winnable", ascending=True)

    plt.figure(figsize=(9, 4.8))
    plt.barh(full_eval["label"], full_eval["acc@5_winnable"])
    plt.axvline(
        baseline["acc@5_winnable"], color="black", linestyle="--", label="baseline"
    )
    plt.xlabel("acc@5_winnable")
    plt.title("Phase 5 full eval: acc@5_winnable by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "full_eval_acc5_winnable.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.plot(
        ["baseline", "best"],
        [baseline["acc@1_winnable"], full_eval.iloc[-1]["acc@1_winnable"]],
        marker="o",
        label="acc@1_winnable",
    )
    plt.plot(
        ["baseline", "best"],
        [baseline["acc@3_winnable"], full_eval.iloc[-1]["acc@3_winnable"]],
        marker="o",
        label="acc@3_winnable",
    )
    plt.plot(
        ["baseline", "best"],
        [baseline["acc@5_winnable"], full_eval.iloc[-1]["acc@5_winnable"]],
        marker="o",
        label="acc@5_winnable",
    )
    plt.ylabel("accuracy (winnable)")
    plt.title("Baseline vs best Phase 5 candidate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_vs_best_acc.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = ensure_out_dir(run_dir)

    screening_path = run_dir / "phase5_screening_scoreboard.csv"
    full_eval_path = run_dir / "phase5_full_eval_scoreboard.csv"

    screening = (
        pd.read_csv(screening_path) if screening_path.exists() else pd.DataFrame()
    )
    full_eval = pd.read_csv(full_eval_path) if full_eval_path.exists() else pd.DataFrame()
    baseline = load_baseline(run_dir)

    plot_screen_scatter(screening, baseline, out_dir)
    plot_full_eval_bars(full_eval, baseline, out_dir)

    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
