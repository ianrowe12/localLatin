"""Build the paper's two headline tables from the Task A/B results CSV.

``tab:taskA_headline`` and ``tab:taskB_headline`` carry most of the numbers the
abstract quotes, and until now they were hand-maintained (issue #81). Both are
pure functions of ``phase_resubmit_results.csv``, so this recovers them.

Layout, shared by both tables: six models down the side, four post-processing
settings across, twice over for two metrics. One layer is chosen per (model,
setting) cell by the training-set criterion for that task, and printed as a
subscript, so the test numbers are never selected on test.

Below the six model rows sits a reference block (issue #118). It holds the
supervised fine-tuning ceiling and the three lexical baselines, scored on the
same split, the same held-out test set and the same evaluation code, so a
reader does not have to leave the headline table to see what surface overlap
and what supervision achieve. The lexical rows have no post-processing axis, so
each spans its metric block; the fine-tuned row fills only the Base and ABTT
columns.

    python scripts/resubmit/build_headline_tables.py
"""
from __future__ import annotations

import argparse
import math
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

# Order of the reference block, top to bottom.
LEXICAL_SYSTEMS: List[Tuple[str, str]] = [
    ("BM25 (word)", "BM25 (word)"),
    ("TF-IDF char 3-5", "TF-IDF char 3--5"),
    ("Levenshtein", "Levenshtein"),
]

# The fine-tuning ceiling was run with a baseline and an ABTT variant and
# nothing else, so its two CSV rows collapse into one table row that fills the
# Base and ABTT columns and leaves the two SIF columns empty.
FINETUNE_ROW_LABEL = "LaTa (fine-tuned)"
FINETUNE_VARIANTS: List[Tuple[str, str]] = [
    ("LaTa (fine-tuned)", "baseline"),
    ("LaTa (fine-tuned) + ABTT", "abtt_optimal"),
]

# Overleaf receives these files, so the header says nothing about the repo.
HEADER = "% generated table\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--results_csv",
        default="runs/active/resubmit/results/phase_resubmit_results.csv",
    )
    p.add_argument(
        "--lexical_csv",
        default="runs/active/resubmit/results/lexical_baselines.csv",
    )
    p.add_argument(
        "--finetune_csv",
        default=(
            "runs/active/resubmit/results/finetune/"
            "finetune_lata_ceiling_comparison.csv"
        ),
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


def finetune_cells(
    values: Sequence[float], layers: Sequence[float], fmt: str
) -> List[str]:
    """Format a fine-tuning ceiling block, with empty cells where SIF has no run."""
    present = [v for v in values if not math.isnan(v)]
    best = max(present) if present else float("nan")
    cells = []
    for value, layer in zip(values, layers):
        if math.isnan(value):
            cells.append("--")
            continue
        text = format(value, fmt)
        if value == best:
            text = f"\\textbf{{{text}}}"
        cells.append(f"{text}\\,\\textsubscript{{{int(layer)}}}")
    return cells


NAN = float("nan")


def _finetune_row(
    finetune: pd.DataFrame,
    left_col: str,
    right_col: str,
    layer_col: str,
    fmt: str,
    scale: float,
) -> str:
    """One table row for the ceiling, its base value under Base and ABTT under ABTT."""
    left_values, right_values, layers = [], [], []
    for csv_label, method in FINETUNE_VARIANTS:
        rows = finetune[finetune["system"] == csv_label]
        if rows.empty:
            raise SystemExit(f"no {csv_label!r} row in the fine-tuning ceiling CSV")
        row = rows.iloc[0]
        if row["method"] != method:
            raise SystemExit(
                f"{csv_label!r} is method {row['method']!r}, expected {method!r}"
            )
        left_values.append(float(row[left_col]) * scale)
        right_values.append(float(row[right_col]) * scale)
        layers.append(float(row[layer_col]))

    # Column order is Base, SIF, ABTT, SIF+ABTT; the ceiling has no SIF run.
    order = [left_values[0], NAN, left_values[1], NAN]
    order_right = [right_values[0], NAN, right_values[1], NAN]
    cell_layers = [layers[0], NAN, layers[1], NAN]
    left = finetune_cells(order, cell_layers, fmt)
    right = finetune_cells(order_right, cell_layers, fmt)
    return " & ".join([FINETUNE_ROW_LABEL] + left + right) + r" \\"


def reference_rows(
    lexical: pd.DataFrame,
    finetune: pd.DataFrame,
    lexical_left_col: str,
    lexical_right_col: str,
    finetune_left_col: str,
    finetune_right_col: str,
    finetune_layer_col: str,
    fmt: str,
    scale: float,
) -> List[str]:
    """The reference block printed under the six model rows.

    Two kinds of row, shaped differently on purpose. A fine-tuned encoder still
    has a Base and an ABTT variant, so its numbers sit in those two columns and
    the SIF columns stay empty. A lexical baseline has no post-processing axis
    at all, so one value spans its whole metric block rather than being repeated
    four times, which would read as four separate runs.
    """
    lines = [r"\midrule"]
    lines.append(
        _finetune_row(
            finetune,
            finetune_left_col,
            finetune_right_col,
            finetune_layer_col,
            fmt,
            scale,
        )
    )

    for key, display in LEXICAL_SYSTEMS:
        rows = lexical[lexical["model"] == key]
        if rows.empty:
            raise SystemExit(f"no {key!r} row in the lexical baselines CSV")
        row = rows.iloc[0]
        left = format(float(row[lexical_left_col]) * scale, fmt)
        right = format(float(row[lexical_right_col]) * scale, fmt)
        lines.append(
            f"{display} & \\multicolumn{{4}}{{c}}{{{left}}} "
            f"& \\multicolumn{{4}}{{c}}{{{right}}} \\\\"
        )

    return lines


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
    reference_lines: Sequence[str],
) -> str:
    lines = [
        HEADER.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        # \footnotesize, not \small: the reference block adds four rows and the
        # table has to stay inside the page budget alongside five other floats.
        r"\footnotesize",
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

    lines += list(reference_lines)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def reference_caption() -> str:
    """The two sentences that explain the reference block.

    It says what the rows are and how to read them, and nothing else. The
    comparison itself is a row-against-row read of the table, so restating the
    cells in prose would only cost lines. The second sentence is the framing
    constraint from issue #119: no caption may imply that the embeddings beat
    surface matching on this corpus.
    """
    return (
        " Below the rule, reference systems on the same split with the same "
        "evaluation code: LaTa fine-tuned contrastively on the 565 positive "
        "train pairs, and three lexical baselines fitted on train files. "
        "Surface overlap is the practitioner's operating point on this corpus, "
        "and the embedding rows diagnose representation geometry rather than "
        "beat it."
    )


def task_a_caption(best: pd.DataFrame) -> str:
    base = best[best["_method"] == "baseline"]["aucroc"]
    abtt = best[best["_method"] == "abtt_optimal"]["aucroc"]
    return (
        "Task A pairwise duplicate detection for all six models under four "
        "post-processing settings. Each cell is a test-set score at the layer "
        "chosen by training-set AUROC, given as the subscript; ABTT uses $D$ "
        "tuned on train. Baseline AUROC spans "
        f"{base.min():.3f} to {base.max():.3f}, and ABTT lifts every model into a "
        f"{abtt.min():.3f} to {abtt.max():.3f} band. Cosine gap is defined in "
        "Figure~\\ref{fig:gap}. Per-layer "
        "grids: Appendix Tables~\\ref{tab:taskA_main}, \\ref{tab:taskA_appendix}, "
        "and~\\ref{tab:taskA_appendix_sif}."
        + reference_caption()
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
        + reference_caption()
    )


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results_csv)
    lexical = pd.read_csv(args.lexical_csv)
    finetune = pd.read_csv(args.finetune_csv)
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
        reference_lines=reference_rows(
            lexical,
            finetune,
            lexical_left_col="aucroc",
            lexical_right_col="gap",
            finetune_left_col="taskA_aucroc",
            finetune_right_col="taskA_cosine_gap",
            finetune_layer_col="taskA_layer",
            fmt=".3f",
            scale=1.0,
        ),
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
        reference_lines=reference_rows(
            lexical,
            finetune,
            lexical_left_col="overall_assignment_acc",
            lexical_right_col="dir_acc_at_1",
            finetune_left_col="taskB_assignment_acc",
            finetune_right_col="taskB_dir_acc_at_1",
            finetune_layer_col="taskB_layer",
            fmt=".1f",
            scale=100.0,
        ),
    )
    (out_dir / "taskB_headline.tex").write_text(task_b)
    print(f"Wrote {out_dir / 'taskB_headline.tex'}")


if __name__ == "__main__":
    main()
