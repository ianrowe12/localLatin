"""Build the paper's two headline tables from the Task A/B results CSV.

``tab:taskA_headline`` and ``tab:taskB_headline`` carry most of the numbers the
abstract quotes, and until now they were hand-maintained (issue #81). Both are
pure functions of ``phase_resubmit_results.csv``, so this recovers them.

Layout, shared by both tables: six models down the side, four post-processing
settings across, twice over for two metrics. One layer is chosen per (model,
setting) cell by the training-set criterion for that task, and printed as a
subscript, so the test numbers are never selected on test.

Below the six model rows sits a reference block (issue #118). By default it
holds one row, the supervised fine-tuning ceiling, scored on the same split,
the same held-out test set and the same evaluation code, so a reader does not
have to leave the headline table to see what supervision achieves. The
fine-tuned row fills only the Base and ABTT columns.

The three lexical baselines (BM25, character 3-5-gram TF-IDF, Levenshtein)
left the paper by decision (issue #197, 2026-09-15) and are rebuttal material.
Passing ``--lexical_csv`` adds them back as rows that span their metric block
and adds their caption clauses; without it the tables carry no lexical row and
no lexical sentence.

``--finetune_csv`` is repeatable (issue #194): one ceiling per model, one row
per ceiling, and every caption claim about a ceiling is derived from that
model's own cells. A second model that beat a zero-shot cell where the first
did not has to change the sentence, not inherit it.

    python scripts/resubmit/build_headline_tables.py \
        --finetune_csv .../finetune_lata_ceiling_comparison.csv \
        --finetune_run_info .../finetune/run_info.json \
        --finetune_csv .../finetune_qwen3_0.6b_ceiling_comparison.csv \
        --finetune_run_info .../finetune/qwen3_0.6b/run_info.json
    python scripts/resubmit/build_headline_tables.py \
        --lexical_csv runs/active/resubmit/results/lexical_baselines.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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

# Each fine-tuning ceiling was run with a baseline and an ABTT variant and
# nothing else, so a model's two CSV rows collapse into one table row that fills
# the Base and ABTT columns and leaves the two SIF columns empty. Issue #194
# added a second ceiling (Qwen3-0.6B), so the labels are read off the comparison
# CSVs rather than listed here: whichever models were run get a row.
FINETUNE_SUFFIX = " (fine-tuned)"
ABTT_SUFFIX = " + ABTT"


def finetune_labels(finetune: pd.DataFrame) -> List[str]:
    """The fine-tuned base labels, in the order the comparison CSVs supplied them."""
    labels: List[str] = []
    for system in finetune["system"]:
        if str(system).endswith(FINETUNE_SUFFIX) and system not in labels:
            labels.append(str(system))
    if not labels:
        raise SystemExit(
            "no '<model> (fine-tuned)' row in the fine-tuning ceiling CSV(s)"
        )
    return labels


def finetune_variants(label: str) -> List[Tuple[str, str]]:
    return [(label, "baseline"), (label + ABTT_SUFFIX, "abtt_optimal")]


def short_name(label: str) -> str:
    return label[: -len(FINETUNE_SUFFIX)] if label.endswith(FINETUNE_SUFFIX) else label


def _and_list(parts: Sequence[str]) -> str:
    parts = list(parts)
    if len(parts) < 2:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + (" and " if len(parts) == 2 else ", and ") + parts[-1]


DEFAULT_FINETUNE_CSV = (
    "runs/active/resubmit/results/finetune/finetune_lata_ceiling_comparison.csv"
)
DEFAULT_FINETUNE_RUN_INFO = "runs/active/resubmit/finetune/run_info.json"

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
        default=None,
        help=(
            "Opt-in: the lexical baselines CSV (BM25, TF-IDF char 3-5, "
            "Levenshtein). Off by default since issue #197 removed the lexical "
            "rows from the paper; pass it only to rebuild the rebuttal variant."
        ),
    )
    p.add_argument(
        "--finetune_csv",
        action="append",
        default=None,
        help=(
            "Comparison CSV of a fine-tuning ceiling. Repeat once per model; "
            "the reference block gets a row per model in the order given."
        ),
    )
    p.add_argument(
        "--finetune_run_info",
        action="append",
        default=None,
        help=(
            "run_info.json written by a fine-tuning run; its caption_facts "
            "give the pair and dev-carve counts the reference caption quotes. "
            "Repeat alongside --finetune_csv. If absent, or if the runs "
            "disagree, the caption states the carve without numbers."
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
    label: str,
    left_col: str,
    right_col: str,
    layer_col: str,
    fmt: str,
    scale: float,
) -> str:
    """One table row for the ceiling, its base value under Base and ABTT under ABTT."""
    left_values, right_values, layers = [], [], []
    for csv_label, method in finetune_variants(label):
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
    return " & ".join([label] + left + right) + r" \\"


def reference_rows(
    lexical: Optional[pd.DataFrame],
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
    four times, which would read as four separate runs. ``lexical`` is None
    unless ``--lexical_csv`` was given (issue #197), and then the block holds
    the fine-tuned row alone.
    """
    lines = [r"\midrule"]
    for label in finetune_labels(finetune):
        lines.append(
            _finetune_row(
                finetune,
                label,
                finetune_left_col,
                finetune_right_col,
                finetune_layer_col,
                fmt,
                scale,
            )
        )

    if lexical is None:
        return lines

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
        # \footnotesize, not \small: the reference block adds rows (one by
        # default, four with --lexical_csv) and the table has to stay inside
        # the page budget alongside five other floats.
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


def load_finetune_facts(path: str) -> Optional[dict]:
    """The ``caption_facts`` block of the fine-tuning run's ``run_info.json``.

    Returns None when the file is absent so the caption can still be built,
    without the numbers, from the comparison CSV alone.
    """
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("caption_facts")


def finetune_pairs_clause(facts: Sequence[Optional[dict]]) -> str:
    """How many pairs the fine-tuned rows were trained on, read from the runs.

    The caption used to say "the 565 positive train pairs", which is the number
    available; a run trains on what is left after the directory-level dev carve
    (issue #185, review item 6). Quoting the runs' own counts keeps the caption
    from drifting when the carve or the split changes.

    Every ceiling is carved from one split with one seed, so the counts agree
    and the caption states them once. If a future run breaks that, the clause
    falls back to the number-free phrasing rather than quoting one model's
    counts for all of them.
    """
    known = [f for f in facts if f]
    if not known:
        return "the positive train pairs left by a directory-level dev carve"
    keys = ("n_fit_pairs", "n_all_train_pairs", "n_dev_dirs")
    shared = {k: {f[k] for f in known} for k in keys}
    if any(len(v) != 1 for v in shared.values()):
        return "the positive train pairs left by a directory-level dev carve"
    return (
        f"{shared['n_fit_pairs'].pop()} of the {shared['n_all_train_pairs'].pop()} "
        f"positive train pairs (a {shared['n_dev_dirs'].pop()}-directory dev carve "
        f"takes the rest)"
    )


def reference_caption(
    comparison: str,
    facts: Sequence[Optional[dict]] = (),
    with_lexical: bool = False,
    model_names: Sequence[str] = ("LaTa",),
) -> str:
    """The sentences that explain the reference block.

    The first says what the rows are, with the fine-tuning pair count taken
    from the run (``facts``) and a pointer to the appendix that gives the
    setup. ``comparison`` is one sentence, derived from the cells by the
    caller, that states how the reference rows sit against the model rows, so
    a re-run cannot ship new numbers under old prose. The fine-tuning ceiling
    is a finding at this training budget, not a bar cleared (issue #176).

    With ``with_lexical`` (the rebuttal variant, issue #197) the caption also
    names the three lexical rows and closes with the framing constraint from
    issues #119 and #176: no caption may imply that the embeddings beat
    surface matching on this corpus. Without it the paper carries no
    comparison against surface matching, so it says nothing either way.
    """
    names = _and_list(list(model_names))
    if not with_lexical:
        # "a reference system" for one ceiling, "reference systems" for more:
        # the article has to go with the plural, not just the noun.
        opening = (
            "a reference system" if len(model_names) == 1 else "reference systems"
        )
        return (
            f" Below the rule, {opening} on the same split with "
            "the same evaluation code: " + names + " fine-tuned contrastively on "
            + finetune_pairs_clause(facts)
            + ", a ceiling at this training budget (setup: "
            "Appendix~\\ref{app:reference_systems}). "
            + comparison
        )
    return (
        " Below the rule, reference systems on the same split with the same "
        "evaluation code: " + names + " fine-tuned contrastively on "
        + finetune_pairs_clause(facts)
        + ", a ceiling at this training budget, and three lexical baselines "
        "fitted on train files (both setups: Appendix~\\ref{app:reference_systems}). "
        + comparison
        + " Surface overlap is the practitioner's operating point on this "
        "corpus, and the embedding rows diagnose representation geometry "
        "rather than beat it."
    )


def _lexical_value(lexical: pd.DataFrame, key: str, col: str) -> float:
    rows = lexical[lexical["model"] == key]
    if rows.empty:
        raise SystemExit(f"no {key!r} row in the lexical baselines CSV")
    return float(rows.iloc[0][col])


def _finetune_value(finetune: pd.DataFrame, csv_label: str, col: str) -> float:
    rows = finetune[finetune["system"] == csv_label]
    if rows.empty:
        raise SystemExit(f"no {csv_label!r} row in the fine-tuning ceiling CSV")
    return float(rows.iloc[0][col])


def level_word(diff: float, tolerance: float) -> str:
    """'level with' inside the seed spread, otherwise 'above' or 'below'.

    Issue #176: a 0.1-point single-seed difference against seed standard
    deviations of up to 1.0 is not a lead, so the caption may only say
    'above' or 'below' when the difference clears ``tolerance``.
    """
    if abs(diff) <= tolerance:
        return "level with"
    return "above" if diff > 0 else "below"


def task_a_comparison(
    best: pd.DataFrame, lexical: Optional[pd.DataFrame], finetune: pd.DataFrame
) -> str:
    """One sentence on the reference rows of the Task A table, from its cells.

    Without ``lexical`` the sentence covers the fine-tuned row alone.
    """
    abtt_best = float(best[best["_method"] == "abtt_optimal"]["aucroc"].max())
    labels = finetune_labels(finetune)

    def move(label: str) -> Tuple[float, float, float, float]:
        return (
            _finetune_value(finetune, label, "taskA_aucroc"),
            _finetune_value(finetune, label + ABTT_SUFFIX, "taskA_aucroc"),
            _finetune_value(finetune, label, "taskA_cosine_gap"),
            _finetune_value(finetune, label + ABTT_SUFFIX, "taskA_cosine_gap"),
        )

    if len(labels) == 1:
        # One ceiling keeps the published wording exactly, so adding the
        # machinery for a second model does not rewrite a shipped caption.
        ft_base, ft_abtt, ft_base_gap, ft_abtt_gap = move(labels[0])
        finetune_clause = (
            f"ABTT moves the fine-tuned encoder's AUROC from {ft_base:.3f} to "
            f"{ft_abtt:.3f} while moving its gap from {ft_base_gap:.3f} to "
            f"{ft_abtt_gap:.3f}."
        )
    else:
        moves = []
        for label in labels:
            ft_base, ft_abtt, ft_base_gap, ft_abtt_gap = move(label)
            moves.append(
                f"{ft_base:.3f} to {ft_abtt:.3f} for {short_name(label)} "
                f"(gap {ft_base_gap:.3f} to {ft_abtt_gap:.3f})"
            )
        finetune_clause = (
            "ABTT moves the fine-tuned encoders' AUROC " + _and_list(moves) + "."
        )
    if lexical is None:
        return finetune_clause
    tfidf = _lexical_value(lexical, "TF-IDF char 3-5", "aucroc")
    return (
        f"TF-IDF char 3--5 is {level_word(tfidf - abtt_best, 0.001)} the best "
        f"ABTT AUROC ({tfidf:.3f} against {abtt_best:.3f}), and " + finetune_clause
    )


def task_b_comparison(
    best: pd.DataFrame, lexical: Optional[pd.DataFrame], finetune: pd.DataFrame
) -> str:
    """One sentence on the reference rows of the Task B table, from its cells.

    ``tolerance`` is one point: the five-seed standard deviations of the
    SIF+ABTT cells reach 1.0, so a smaller single-seed difference is 'level'.
    Without ``lexical`` the sentence covers the fine-tuned row alone.
    """
    abtt = best[best["_method"] == "abtt_optimal"]
    assign = 100.0 * abtt["overall_assignment_acc"]
    dir1 = 100.0 * abtt["dir_acc_at_1"]
    labels = finetune_labels(finetune)

    def placement(label: str) -> Tuple[float, float, str]:
        """Where one ceiling sits against the zero-shot ABTT cells.

        Derived per model from the cells, so a second ceiling that clears a
        cell the first one did not must change the sentence rather than
        inherit "below everything".
        """
        ft_assign = 100.0 * _finetune_value(
            finetune, label + ABTT_SUFFIX, "taskB_assignment_acc"
        )
        ft_dir1 = 100.0 * _finetune_value(
            finetune, label + ABTT_SUFFIX, "taskB_dir_acc_at_1"
        )
        if ft_assign < assign.min() and ft_dir1 < dir1.min():
            where = "below every zero-shot ABTT cell"
        elif ft_assign > assign.max() and ft_dir1 > dir1.max():
            where = "above every zero-shot ABTT cell"
        else:
            where = "inside the zero-shot ABTT range"
        return ft_assign, ft_dir1, where

    if len(labels) == 1:
        # One ceiling keeps the published wording exactly.
        ft_assign, ft_dir1, where = placement(labels[0])
        finetune_clause = (
            f"the fine-tuned encoder with ABTT ({ft_assign:.1f} and {ft_dir1:.1f}) "
            f"is {where} ({assign.min():.1f} to {assign.max():.1f} and "
            f"{dir1.min():.1f} to {dir1.max():.1f})."
        )
    else:
        placements = []
        for label in labels:
            ft_assign, ft_dir1, where = placement(label)
            placements.append(
                f"{where} for {short_name(label)} ({ft_assign:.1f} and {ft_dir1:.1f})"
            )
        finetune_clause = (
            "the fine-tuned encoders with ABTT sit " + _and_list(placements)
            + f", against cells spanning {assign.min():.1f} to {assign.max():.1f} "
            f"and {dir1.min():.1f} to {dir1.max():.1f}."
        )
    if lexical is None:
        return finetune_clause[0].upper() + finetune_clause[1:]
    tf_assign = 100.0 * _lexical_value(lexical, "TF-IDF char 3-5", "overall_assignment_acc")
    tf_dir1 = 100.0 * _lexical_value(lexical, "TF-IDF char 3-5", "dir_acc_at_1")
    return (
        f"TF-IDF char 3--5 is {level_word(tf_assign - assign.max(), 1.0)} the "
        f"best ABTT cell ({tf_assign:.1f} against {assign.max():.1f} assignment "
        f"accuracy, {tf_dir1:.1f} against {dir1.max():.1f} directory accuracy at "
        f"rank 1), and " + finetune_clause
    )


def task_a_caption(
    best: pd.DataFrame,
    lexical: Optional[pd.DataFrame],
    finetune: pd.DataFrame,
    facts: Sequence[Optional[dict]] = (),
) -> str:
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
        + reference_caption(
            task_a_comparison(best, lexical, finetune),
            facts,
            with_lexical=lexical is not None,
            model_names=[short_name(l) for l in finetune_labels(finetune)],
        )
    )


def task_b_caption(
    results: pd.DataFrame,
    best: pd.DataFrame,
    lexical: Optional[pd.DataFrame],
    finetune: pd.DataFrame,
    facts: Sequence[Optional[dict]] = (),
) -> str:
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
        + reference_caption(
            task_b_comparison(best, lexical, finetune),
            facts,
            with_lexical=lexical is not None,
            model_names=[short_name(l) for l in finetune_labels(finetune)],
        )
    )


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results_csv)
    lexical = pd.read_csv(args.lexical_csv) if args.lexical_csv else None
    finetune_csvs = args.finetune_csv or [DEFAULT_FINETUNE_CSV]
    finetune = pd.concat(
        [pd.read_csv(path) for path in finetune_csvs], ignore_index=True
    )
    facts = [
        load_finetune_facts(path)
        for path in (args.finetune_run_info or [DEFAULT_FINETUNE_RUN_INFO])
    ]
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
        caption=task_a_caption(best_a, lexical, finetune, facts),
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
        caption=task_b_caption(results, best_b, lexical, finetune, facts),
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
