"""Build the paper's two headline tables from the Task A/B results CSV.

``tab:taskA_headline`` and ``tab:taskB_headline`` carry most of the numbers the
abstract quotes, and until now they were hand-maintained (issue #81). Both are
pure functions of ``phase_resubmit_results.csv``, so this recovers them.

Layout, shared by both tables: six models down the side, four post-processing
settings across, twice over for two metrics. One layer is chosen per (model,
setting) cell by the training-set criterion for that task, and printed as a
subscript, so the test numbers are never selected on test.

    python scripts/resubmit/build_headline_tables.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

# Ordering is shared with the per-layer appendix tables.
MODELS: List[Tuple[str, str]] = [
    ("bowphs/LaTa", "LaTa"),
    ("bowphs/PhilTa", "PhilTa"),
    ("google/mt5-base", "mT5-base"),
    ("sentence-transformers/LaBSE", "LaBSE"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B"),
    (
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "KaLM-mini",
    ),
]

# The ABTT columns use the train-tuned D, not the fixed one.
METHODS: List[Tuple[str, str]] = [
    ("baseline", "Base"),
    ("sif_only", "SIF"),
    ("abtt_optimal", "ABTT"),
    ("sif_abtt_optimal", "SIF+ABTT"),
]

# Overleaf receives these files, so the header says nothing about the repo.
HEADER = "% generated table\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--results_csv",
        default="runs/active/resubmit/results/phase_resubmit_results.csv",
    )
    p.add_argument("--out_dir", default="overleaf_drafts/tables")
    p.add_argument("--repr_name", default="hidden")
    return p.parse_args()


def best_rows(
    results: pd.DataFrame, repr_name: str, select_on: str
) -> pd.DataFrame:
    """One row per (model, method): the layer that maximises `select_on` on train."""
    rows = []
    for model_id, _ in MODELS:
        for method, _ in METHODS:
            sub = results[
                (results["model"] == model_id)
                & (results["repr"] == repr_name)
                & (results["method"] == method)
            ]
            if sub.empty:
                raise SystemExit(
                    f"no rows for model={model_id!r} method={method!r} "
                    f"repr={repr_name!r} in the results CSV"
                )
            best = sub.loc[sub[select_on].idxmax()].copy()
            best["_model_id"] = model_id
            best["_method"] = method
            rows.append(best)
    return pd.DataFrame(rows)


def format_cells(
    values: Sequence[float], layers: Sequence[int], fmt: str
) -> List[str]:
    """Format one metric block of a row, bolding every cell at the block maximum.

    Ties are bolded together. mT5-base reaches exactly the same assignment
    accuracy under ABTT and SIF+ABTT (identical integer counts), so an idxmax
    would drop one of the two bolds.
    """
    best = max(values)
    cells = []
    for value, layer in zip(values, layers):
        text = format(value, fmt)
        if value == best:
            text = f"\\textbf{{{text}}}"
        cells.append(f"{text}\\,\\textsubscript{{{int(layer)}}}")
    return cells


def render_table(
    best: pd.DataFrame,
    left_banner: str,
    right_banner: str,
    left_col: str,
    right_col: str,
    fmt: str,
    scale: float,
    caption: str,
    label: str,
) -> str:
    lines = [
        HEADER.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        f"& \\multicolumn{{4}}{{c}}{{\\textbf{{{left_banner}}}}} "
        f"& \\multicolumn{{4}}{{c}}{{\\textbf{{{right_banner}}}}} \\\\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"\textbf{Model} & "
        + " & ".join(label for _, label in METHODS)
        + " & "
        + " & ".join(label for _, label in METHODS)
        + r" \\",
        r"\midrule",
    ]

    for model_id, display in MODELS:
        rows = [
            best[(best["_model_id"] == model_id) & (best["_method"] == method)].iloc[0]
            for method, _ in METHODS
        ]
        layers = [r["layer"] for r in rows]
        left = format_cells([r[left_col] * scale for r in rows], layers, fmt)
        right = format_cells([r[right_col] * scale for r in rows], layers, fmt)
        lines.append(" & ".join([display] + left + right) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def task_a_caption(best: pd.DataFrame) -> str:
    base = best[best["_method"] == "baseline"]["aucroc"]
    abtt = best[best["_method"] == "abtt_optimal"]["aucroc"]
    return (
        "Task A pairwise duplicate detection for all six models under four "
        "post-processing settings. Each cell is a test-set score at the layer "
        "chosen by training-set AUROC, given as the subscript; ABTT uses $D$ "
        "tuned on train. Baseline AUROC spans "
        f"{base.min():.3f} to {base.max():.3f}, and ABTT lifts every model into a "
        f"{abtt.min():.3f} to {abtt.max():.3f} band. Cosine gap is the mean "
        "same-directory minus the mean different-directory cosine. Per-layer "
        "grids: Appendix Tables~\\ref{tab:taskA_main}, \\ref{tab:taskA_appendix}, "
        "and~\\ref{tab:taskA_appendix_sif}."
    )


def task_b_caption(results: pd.DataFrame) -> str:
    row = results.iloc[0]
    prior = 100.0 * float(row["n_existing"]) / float(row["n_test"])
    return (
        "Task B autonomous routing for all six models under four post-processing "
        "settings, in percent. Each cell is a test-set score at the layer chosen "
        "by training-set directory accuracy at rank 1, given as the subscript. "
        "Assignment accuracy scores the existing-versus-new decision alone, "
        "comparing each file's maximum cosine against the train-fit threshold "
        "$\\tau$; a degenerate threshold that routes everything as existing "
        f"already attains the {prior:.1f} percent class prior. Directory accuracy "
        "at rank 1 requires the correct directory, so read the two columns "
        "together. Per-layer grids: Appendix "
        "Tables~\\ref{tab:taskB_routing_main}, \\ref{tab:taskB_routing_appendix}, "
        "and~\\ref{tab:taskB_routing_appendix_sif}."
    )


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_a = best_rows(results, args.repr_name, "train_aucroc")
    task_a = render_table(
        best_a,
        left_banner="Task A AUROC",
        right_banner="Task A cosine gap",
        left_col="aucroc",
        right_col="gap",
        fmt=".3f",
        scale=1.0,
        caption=task_a_caption(best_a),
        label="tab:taskA_headline",
    )
    (out_dir / "taskA_headline.tex").write_text(task_a)
    print(f"Wrote {out_dir / 'taskA_headline.tex'}")

    best_b = best_rows(results, args.repr_name, "train_dir_acc_at_1")
    task_b = render_table(
        best_b,
        left_banner="Assignment accuracy",
        right_banner="Directory accuracy @1",
        left_col="overall_assignment_acc",
        right_col="dir_acc_at_1",
        fmt=".1f",
        scale=100.0,
        caption=task_b_caption(results),
        label="tab:taskB_headline",
    )
    (out_dir / "taskB_headline.tex").write_text(task_b)
    print(f"Wrote {out_dir / 'taskB_headline.tex'}")


if __name__ == "__main__":
    main()
