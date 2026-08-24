"""Build a per-model best-layer Task A summary table from phase_resubmit_results.csv.

For each model, selects the (method, layer, D) row that maximises a chosen
selection metric and emits both a CSV and a LaTeX table for the paper.

Default selection metric is `overall_assignment_acc` on the test split, which
matches the Phase 10+ primary metric described in CLAUDE.md and what the prior
pipeline already optimised. Pass `--select_on train_dir_acc_at_1` to instead
pick the layer using a train metric (avoids test-set selection bias) and still
report the test numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "google/mt5-base": "mT5-base",
}
MODEL_ORDER = list(MODEL_DISPLAY)

REPORT_COLS = [
    "model_display",
    "best_layer",
    "method",
    "D",
    "dir_acc_at_1",
    "overall_assignment_acc",
    "aucroc",
    "existing_acc",
    "new_acc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_csv",
        default="runs/active/resubmit/results/phase_resubmit_results.csv",
    )
    p.add_argument(
        "--out_csv",
        default="runs/active/resubmit/results/taskA_per_model_summary.csv",
    )
    p.add_argument(
        "--out_tex",
        default="overleaf_drafts/tables/taskA_per_model.tex",
    )
    p.add_argument(
        "--select_on",
        default="overall_assignment_acc",
        choices=["overall_assignment_acc", "dir_acc_at_1", "train_dir_acc_at_1", "train_aucroc"],
        help="Column used to pick the best layer per model.",
    )
    return p.parse_args()


def build_table(df: pd.DataFrame, select_on: str) -> pd.DataFrame:
    if select_on not in df.columns:
        raise SystemExit(f"--select_on={select_on!r} not in CSV columns: {list(df.columns)}")
    keep = df[df[select_on].notna()].copy()
    idx = keep.groupby("model")[select_on].idxmax()
    best = keep.loc[idx].copy()
    best = best.rename(columns={"layer": "best_layer"})
    best["model_display"] = best["model"].map(MODEL_DISPLAY).fillna(best["model"])
    # Stable model ordering: known models first in MODEL_ORDER, then any extras alphabetically.
    known = [m for m in MODEL_ORDER if m in best["model"].values]
    extras = sorted(set(best["model"]) - set(MODEL_ORDER))
    order = {m: i for i, m in enumerate(known + extras)}
    best = best.sort_values("model", key=lambda s: s.map(order))
    return best[REPORT_COLS].reset_index(drop=True)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def write_outputs(table: pd.DataFrame, out_csv: Path, out_tex: Path, select_on: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    display = table.rename(
        columns={
            "model_display": "Model",
            "best_layer": "Layer",
            "method": "Method",
            "D": "D",
            "dir_acc_at_1": "Acc@1",
            "overall_assignment_acc": "AssignAcc",
            "aucroc": "AUROC",
            "existing_acc": "Exist.Acc",
            "new_acc": "New.Acc",
        }
    )
    # Escape underscores in cells where pandas won't (escape=False so we control everything).
    display = display.copy()
    display["Method"] = display["Method"].map(_tex_escape)
    caption = (
        "Per-model best Task~A configuration on the refreshed split "
        f"(best layer chosen by \\texttt{{{_tex_escape(select_on)}}})."
    )
    display.to_latex(
        out_tex,
        index=False,
        float_format="%.3f",
        column_format="l" + "c" * (len(display.columns) - 1),
        caption=caption,
        label="tab:taskA_per_model",
        escape=False,
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    table = build_table(df, args.select_on)
    write_outputs(table, Path(args.out_csv), Path(args.out_tex), args.select_on)
    print(f"selection metric: {args.select_on}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_tex}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
