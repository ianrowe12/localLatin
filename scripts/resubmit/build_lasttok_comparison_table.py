"""Build the mean-vs-last-token pooling comparison table for the appendix.

Reads two result CSVs (one for mean pooling, one for last-token pooling,
both restricted to ABTT methods) and writes ``tables/appendix_lasttok_comparison.tex``.

Selection is train-only (issue #184). For each (model, pooling) pair the
reported layer is the one with the highest training-set directory accuracy at
rank 1 under ``--select_method`` (default ``abtt_optimal``, ABTT with $D$ tuned
per layer on train), through the same ``train_selected_layers`` helper the
headline and per-layer tables use. Before #184 the default was the test-set
argmax of ``overall_assignment_acc``, which bolded LaTa mean layer 1 (0.909)
while Table~tab:taskB_headline reports layer 8. ``--select_on`` refuses any
column that is not a ``train_`` metric, so the sbatch cannot regress.

Only the six paper models are reported. The last-token run also covered
Qwen3-8B, which the paper does not otherwise use, so its rows are dropped;
mT5-base was not part of that run, so the table has five models. The script
tolerates missing CSVs: if either/both inputs are absent, it emits a
placeholder .tex file with TODO comments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from taskb_mseed_selection import train_selected_layers

MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}
MODEL_ORDER = list(MODEL_DISPLAY.values())  # canonical display-name ordering

POOL_DISPLAY = {"mean": "Mean", "lasttok": "Last token"}
POOL_ORDER = ["Mean", "Last token"]

ALLOWED_METHODS = {"abtt_fixed", "abtt_optimal"}
ALLOWED_POOLINGS = {"mean", "lasttok"}

DEFAULT_SELECT_ON = "train_dir_acc_at_1"
DEFAULT_SELECT_METHOD = "abtt_optimal"

CAPTION = (
    "Mean versus last-token pooling under ABTT with $D$ tuned per layer on the train "
    "split, for the five paper models whose last-token embeddings were extracted "
    "(mT5-base was not part of this run). For each model and pooling the layer is the "
    "one with the highest training-set directory accuracy at rank~1, the rule behind the "
    "ABTT cells of Table~\\ref{tab:taskB_headline} (their layers are listed in "
    "Table~\\ref{tab:selected_layers}); $D$ is the number of principal "
    "components removed at that layer. The three score columns are test-set Task~B "
    "assignment accuracy, Task~B directory accuracy at rank~1, and Task~A cosine gap at "
    "that layer."
)
LABEL = "tab:lasttok_comparison"
COLUMN_FORMAT = "llrrccc"
HEADER = (
    "Model & Pooling & Layer & $D$ & Assignment acc. & Dir.\\ acc.\\ @1 & Cosine gap \\\\"
)
DISPLAY_COLS = ["Model", "Pool", "Layer", "D", "Assign Acc", "Acc@1", "Gap"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mean_csv",
        default="runs/active/resubmit/results/lasttok/phase_resubmit_results_mean_abttonly.csv",
    )
    p.add_argument(
        "--lasttok_csv",
        default="runs/active/resubmit/results/lasttok/phase_resubmit_results_lasttok.csv",
    )
    p.add_argument(
        "--out_tex",
        default="overleaf_drafts/tables/appendix_lasttok_comparison.tex",
    )
    p.add_argument(
        "--select_on",
        default=DEFAULT_SELECT_ON,
        help="Train-split column used to pick the layer per (model, pooling) pair. "
        "Must start with 'train_'.",
    )
    p.add_argument(
        "--select_method",
        default=DEFAULT_SELECT_METHOD,
        choices=sorted(ALLOWED_METHODS),
        help="ABTT method whose rows are selected and reported.",
    )
    return p.parse_args()


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"WARN: {path} not found, skipping.", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - tolerate malformed inputs
        print(f"WARN: failed to read {path}: {exc}", file=sys.stderr)
        return None


def _format_d(row: pd.Series) -> str:
    try:
        return str(int(row["D"])) if pd.notna(row["D"]) else "--"
    except (TypeError, ValueError):
        return "--"


def build_table(
    df: pd.DataFrame,
    select_on: str = DEFAULT_SELECT_ON,
    select_method: str = DEFAULT_SELECT_METHOD,
) -> pd.DataFrame:
    """One row per (paper model, pooling) at the train-selected layer."""
    if not select_on.startswith("train_"):
        raise SystemExit(
            f"--select_on={select_on!r} is not a train-split column; the table "
            "selects layers on train only (issue #184)"
        )
    if select_on not in df.columns:
        raise SystemExit(
            f"--select_on={select_on!r} not in CSV columns: {list(df.columns)}"
        )
    if select_method not in ALLOWED_METHODS:
        raise SystemExit(f"--select_method must be one of {sorted(ALLOWED_METHODS)}")

    dropped = sorted(set(df["model"]) - set(MODEL_DISPLAY))
    if dropped:
        print(f"note: dropping models outside the paper set: {dropped}", file=sys.stderr)
    keep = df[
        df["model"].isin(MODEL_DISPLAY)
        & (df["method"] == select_method)
        & df["pooling"].isin(ALLOWED_POOLINGS)
        & df[select_on].notna()
    ].copy()
    if keep.empty:
        return pd.DataFrame(columns=DISPLAY_COLS)

    picked = []
    for pooling in sorted(keep["pooling"].unique()):
        sub = keep[keep["pooling"] == pooling]
        models = [m for m in MODEL_DISPLAY if m in set(sub["model"])]
        layers = train_selected_layers(sub, models, method=select_method, metric=select_on)
        for model, layer in layers.items():
            row = sub[(sub["model"] == model) & (sub["layer"] == layer)]
            if len(row) != 1:
                raise SystemExit(
                    f"expected one row for model={model!r} pooling={pooling!r} "
                    f"layer={layer}, found {len(row)}"
                )
            picked.append(row.iloc[0])
    best = pd.DataFrame(picked).reset_index(drop=True)

    best["Model"] = best["model"].map(MODEL_DISPLAY)
    best["Pool"] = best["pooling"].map(POOL_DISPLAY)
    best["Layer"] = best["layer"].astype(int)
    best["D"] = best.apply(_format_d, axis=1)
    best["Assign Acc"] = best["overall_assignment_acc"].map(lambda v: f"{v:.3f}")
    best["Acc@1"] = best["dir_acc_at_1"].map(lambda v: f"{v:.3f}")
    best["Gap"] = best["gap"].map(lambda v: f"{v:.3f}")

    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    pool_rank = {p: i for i, p in enumerate(POOL_ORDER)}
    best["_model_rank"] = best["Model"].map(model_rank)
    best["_pool_rank"] = best["Pool"].map(pool_rank)
    best = best.sort_values(["_model_rank", "_pool_rank"]).reset_index(drop=True)
    return best[DISPLAY_COLS]


def _missing_pairs(table: pd.DataFrame) -> list[tuple[str, str]]:
    """(model, pooling) pairs with no row, restricted to models the run covered.

    A model absent from both poolings (mT5-base) is not a missing pair: the
    caption says the run did not include it.
    """
    present = set(zip(table["Model"].tolist(), table["Pool"].tolist())) if not table.empty else set()
    covered = {m for m, _ in present}
    missing = []
    for model in MODEL_ORDER:
        if model not in covered:
            continue
        for pool in POOL_ORDER:
            if (model, pool) not in present:
                missing.append((model, pool))
    return missing


def _render_body(table: pd.DataFrame) -> str:
    if table.empty:
        return "% (no rows: both inputs empty after filtering)"
    lines = []
    prev_model = None
    for _, row in table.iterrows():
        if prev_model is not None and row["Model"] != prev_model:
            lines.append("\\midrule")
        cells = [
            row["Model"] if row["Model"] != prev_model else "",
            row["Pool"],
            str(row["Layer"]),
            row["D"],
            row["Assign Acc"],
            row["Acc@1"],
            row["Gap"],
        ]
        lines.append(" & ".join(cells) + " \\\\")
        prev_model = row["Model"]
    return "\n".join(lines)


def _render_tex(table: pd.DataFrame, todo_comments: list[str]) -> str:
    header_comments = "\n".join(todo_comments)
    if header_comments:
        header_comments += "\n"
    body = _render_body(table)
    return (
        f"{header_comments}\\begin{{table*}}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        f"\\caption{{{CAPTION}}}\n"
        f"\\label{{{LABEL}}}\n"
        f"\\begin{{tabular}}{{{COLUMN_FORMAT}}}\n"
        "\\toprule\n"
        f"{HEADER}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def _slug_for_sbatch(model_display: str) -> str:
    return model_display.lower().replace(".", "").replace("-", "_")


def main() -> None:
    args = parse_args()
    mean_path = Path(args.mean_csv)
    lasttok_path = Path(args.lasttok_csv)
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    mean_df = _load_csv(mean_path)
    lasttok_df = _load_csv(lasttok_path)

    todo_comments: list[str] = []

    if mean_df is None and lasttok_df is None:
        todo_comments.append(
            "% TODO: both CSVs missing, run slurm/resubmit/lasttok_evaluate.sbatch first"
        )
        tex = _render_tex(pd.DataFrame(columns=DISPLAY_COLS), todo_comments)
        out_tex.write_text(tex)
        print(f"wrote placeholder {out_tex} (both CSVs missing)")
        return

    frames = [df for df in (mean_df, lasttok_df) if df is not None]
    joined = pd.concat(frames, ignore_index=True)
    table = build_table(joined, args.select_on, args.select_method)

    for model, pool in _missing_pairs(table):
        slug = _slug_for_sbatch(model)
        pool_slug = "mean" if pool == "Mean" else "lasttok"
        todo_comments.append(
            f"% TODO: {model} {pool_slug} missing, rerun sbatch lasttok_extract_{slug}.sbatch"
        )

    tex = _render_tex(table, todo_comments)
    out_tex.write_text(tex)
    print(f"selection: {args.select_method} at the argmax of {args.select_on} (train)")
    print(f"wrote {out_tex}")
    if not table.empty:
        print(table.to_string(index=False))
    if todo_comments:
        print("TODO comments emitted:")
        for c in todo_comments:
            print(f"  {c}")


if __name__ == "__main__":
    main()
