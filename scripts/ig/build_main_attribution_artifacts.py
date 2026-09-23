"""Build main-text attribution reporting artifacts for the three paper models.

The input is the attribution summary of the run of record, the benchmark v1
re-sample of issue #187 scored with a 20-draw random-order reference (issue
#206; ``RUN_OF_RECORD`` and ``METRICS_DIR_OF_RECORD`` in
``attribution_run_of_record.py``):

    runs/active/ig_examples_200pos_v1/attribution_metrics_draws20/summary_v2.csv

together with its gitignored per-pair cache ``v2_hidden/`` for the caption's tie
clause. Every table written here carries a ``% source run:`` stamp naming the
run and the metrics directory, and the generator refuses to overwrite a table
stamped with a different source unless ``--allow_run_change`` is passed (issue
#201), so a bare rerun cannot rewrite the paper's numbers from an older sample
or from the 5-draw pass kept beside this one.

Outputs:

    overleaf_drafts/tables/attribution_metrics_main.tex
    overleaf_drafts/tables/attribution_metrics_secondary.tex
    overleaf_drafts/tables/attribution_shuffle_control.tex
    overleaf_drafts/figures/fig_attribution_rho_loo_main.{pdf,png,tex}

Selection (issue #120, from ``docs/research/attribution_metrics_decision.md``
part B). The main table carries two columns and nothing else: ``rho_LOO``,
which ABTT wins 5/6 on the v1 sample, and ``DelAUC gap``, which it wins 4/6
(6/6 and 3/6 on the run 3 sample the memo was written on). Both are
threshold-free and both have a calibrated zero. They are printed as paired
base/ABTT columns rather than ``base -> ABTT`` arrow cells, which are not a
table convention readers expect.

Everything the old four-metric table used to carry in the main text moves to
the secondary appendix table: ``tau_LOO`` (the tie-corrected twin of
``rho_LOO``, with which it correlates 0.9995, so it is not a second witness),
``InsAUC gap`` (which fails the shuffled-attribution control in a baseline
cell: two on the run 3 sample, one on v1) and the three ERASER-style headline
cells.

The third table (issue #226) is the shuffled-attribution control itself: the
per-cell gap between each metric and its mean over ``SHUFFLE_DRAWS_OF_RECORD``
permutations of the same attribution vector, with its standard error, for the
six candidate metrics over the twelve cells. The main text quotes the control
as a validity statement; this table is where a reader checks it. DelAUC gap
and AOPC-Comprehensiveness have the same shuffle gap to machine precision, as
do InsAUC gap and AOPC-Sufficiency (memo A3: the random-order reference and the
constant trapezoid offset cancel in the difference), so each pair shares one
column and the generator verifies the identity before it claims it.

All three tables read one summary, so all describe the same erasure operator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# matplotlib is imported inside render_rho_figure, not here. Building the two
# tables needs pandas and nothing else, and the table path is what the tests and
# a table-only re-run exercise; a module-level plotting import would make both
# depend on a backend they never use.


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from attribution_run_of_record import (  # noqa: E402
    DEFAULT_SUMMARY_CSV,
    SHUFFLE_DRAWS_OF_RECORD,
    refuse_run_change,
    run_name,
    stamp_line,
)

DEFAULT_SUMMARY = DEFAULT_SUMMARY_CSV
DEFAULT_TABLE_OUT = REPO_ROOT / "overleaf_drafts/tables/attribution_metrics_main.tex"
DEFAULT_SECONDARY_OUT = (
    REPO_ROOT / "overleaf_drafts/tables/attribution_metrics_secondary.tex"
)
DEFAULT_SHUFFLE_OUT = (
    REPO_ROOT / "overleaf_drafts/tables/attribution_shuffle_control.tex"
)
DEFAULT_FIG_OUT = REPO_ROOT / "overleaf_drafts/figures/fig_attribution_rho_loo_main"

MODELS = (
    ("bowphs/LaTa", "LaTa"),
    ("bowphs/PhilTa", "PhilTa"),
    ("google/mt5-base", "mT5-base"),
)
METHODS = (
    ("ig", "IG"),
    ("retrieval_mark", "MaRC"),
)

RHO_KEY = "loo_rho"
DEL_GAP_KEY = "del_auc_gap"
DEL_RANDOM_KEY = "del_auc_random"
TAU_KEY = "loo_tau"
INS_GAP_KEY = "ins_auc_gap"
SUFF_KEY = "suff@0.25_ratio"
COMP_KEY = "comp@0.25_ratio"
MINFRAC_KEY = "compactness@0.80"

_NUMBER_WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four",
                 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                 10: "ten", 11: "eleven", 12: "twelve"}

MAIN_METRIC_KEYS = (RHO_KEY, DEL_GAP_KEY, DEL_RANDOM_KEY)
SECONDARY_METRIC_KEYS = (TAU_KEY, INS_GAP_KEY, SUFF_KEY, COMP_KEY, MINFRAC_KEY)
METRIC_KEYS = MAIN_METRIC_KEYS + SECONDARY_METRIC_KEYS

# The shuffled-attribution control (memo A3): one column per metric or per pair
# of metrics whose shuffle gaps are identical by construction. Each entry is
# (summary keys sharing the column, column label, short label for the caption).
AOPC_COMP_KEY = "aopc_comp_ratio"
AOPC_SUFF_KEY = "aopc_suff_ratio"
SHUFFLE_COLUMNS = (
    ((RHO_KEY,), r"$\rho_{\text{LOO}}$", r"$\rho_{\text{LOO}}$"),
    ((TAU_KEY,), r"$\tau_{\text{LOO}}$", r"$\tau_{\text{LOO}}$"),
    ((DEL_GAP_KEY, AOPC_COMP_KEY), "DelAUC gap, AOPC-Comp", "DelAUC gap"),
    ((INS_GAP_KEY, AOPC_SUFF_KEY), "InsAUC gap, AOPC-Suff", "InsAUC gap"),
)
SHUFFLE_KEYS = tuple(key for keys, _, _ in SHUFFLE_COLUMNS for key in keys)
# Two shuffle gaps count as the same number at this tolerance; the memo
# measured 2.2e-16 between the members of each pair.
SHUFFLE_IDENTITY_TOL = 1e-9


def _shuffle_gap_col(metric_key: str, stat: str) -> str:
    return f"rand_{metric_key}_gap_{stat}"


# Overleaf receives these files, so the header says nothing about the repo.
HEADER = "% generated table"
REGEN_NOTE = (
    "% Selection and wording follow the part B memo behind issue #120; the numbers "
    "come from the\n% run stamped above (issue #187 re-sample). Regenerate from "
    "that summary rather than\n% editing the numbers here."
)


def _header_lines(source_run: Optional[str]) -> list[str]:
    lines = [HEADER]
    if source_run:
        lines.append(stamp_line(source_run))
    lines.append(REGEN_NOTE)
    return lines


def _mean_col(metric_key: str) -> str:
    return f"{metric_key}_mean"


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}"


def _pair_cells(base: float, abtt: float, *, lower_is_better: bool = False) -> list[str]:
    """Two independent cells, the better one bolded. No arrow between them."""
    if pd.isna(base) or pd.isna(abtt):
        return [_fmt(base), _fmt(abtt)]
    abtt_wins = abtt < base if lower_is_better else abtt > base
    base_text = _fmt(base)
    abtt_text = _fmt(abtt)
    if abtt_wins:
        abtt_text = rf"\textbf{{{abtt_text}}}"
    else:
        base_text = rf"\textbf{{{base_text}}}"
    return [base_text, abtt_text]


def _load_main_rows(summary_csv: Path) -> pd.DataFrame:
    return select_main_rows(pd.read_csv(summary_csv), source=str(summary_csv))


def select_main_rows(df: pd.DataFrame, *, source: str = "summary") -> pd.DataFrame:
    """Keep the three models and two views the paper reports, and check the columns."""
    required_cols = {"model", "method", "variant", "n", "full_cos_mean"}
    required_cols.update(_mean_col(k) for k in METRIC_KEYS)
    required_cols.update(f"{k}_n" for k in (DEL_GAP_KEY, INS_GAP_KEY))
    required_cols.update(
        _shuffle_gap_col(k, stat) for k in SHUFFLE_KEYS for stat in ("mean", "se", "n")
    )
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    wanted_models = {m for m, _ in MODELS}
    wanted_methods = {m for m, _ in METHODS}
    sub = df[df["model"].isin(wanted_models) & df["method"].isin(wanted_methods)].copy()

    expected = len(MODELS) * len(METHODS) * 2
    if len(sub) != expected:
        found = sub[["model", "method", "variant"]].sort_values(
            ["model", "method", "variant"]
        )
        raise ValueError(
            f"expected {expected} model/method/variant rows, found {len(sub)}:\n"
            + found.to_string(index=False)
        )
    return sub


def _get(summary: pd.DataFrame, model: str, method: str, variant: str, col: str) -> float:
    row = summary[
        (summary["model"] == model)
        & (summary["method"] == method)
        & (summary["variant"] == variant)
    ]
    if row.empty:
        raise KeyError((model, method, variant, col))
    return float(row.iloc[0][col])


def _wins(summary: pd.DataFrame, metric_key: str, *, lower_is_better: bool = False) -> int:
    won = 0
    for model, _ in MODELS:
        for method, _ in METHODS:
            base = _get(summary, model, method, "baseline", _mean_col(metric_key))
            abtt = _get(summary, model, method, "abtt", _mean_col(metric_key))
            won += int(abtt < base if lower_is_better else abtt > base)
    return won


def _abtt_pair_count_range(summary: pd.DataFrame, metric_key: str) -> tuple[int, int]:
    counts = [
        int(_get(summary, model, method, "abtt", f"{metric_key}_n"))
        for model, _ in MODELS
        for method, _ in METHODS
    ]
    return min(counts), max(counts)


def _baseline_pair_count_range(summary: pd.DataFrame, metric_key: str) -> tuple[int, int]:
    """Baseline pair counts, as a range.

    The ratio metrics are undefined below ``FULL_COS_FLOOR``. On the canon
    sample the baseline cleared that floor for all 200 pairs in every cell, so
    the caption could quote one number. It is not a property of the protocol:
    at a layer whose *uncorrected* pair cosines sit near zero, the baseline
    loses pairs too. Reported as a range, which collapses to one number printed
    once when every cell agrees.
    """
    counts = [
        int(_get(summary, model, method, "baseline", f"{metric_key}_n"))
        for model, _ in MODELS
        for method, _ in METHODS
    ]
    return min(counts), max(counts)


def _count_phrase(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo} to {hi}"


def paired_cell_stats(pairs_root: Path, metric_key: str) -> dict[tuple[str, str], tuple[float, float]]:
    """Paired ABTT-minus-baseline mean and standard error per (model, method).

    The summary carries per-variant standard errors, which do not give the
    standard error of the *difference*: the two variants score the same pairs,
    so the difference is paired and its error is smaller than the unpaired
    combination. The per-pair JSON cache written by
    ``run_attribution_metrics.py`` is what makes the paired form computable, so
    a caption claim about a cell being a tie is read off the same statistic the
    selection memo used.
    """
    import json
    from collections import defaultdict

    per_cell: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for path in sorted(pairs_root.rglob("*.json")):
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in rows:
            key = metric_key if metric_key in row else None
            if key is None or row.get("method") is None:
                continue
            cell = (str(row.get("model")), str(row["method"]))
            slot = per_cell[cell].setdefault(path.stem, {})
            slot[str(row["variant"])] = row[key]

    out: dict[tuple[str, str], tuple[float, float]] = {}
    for cell, examples in per_cell.items():
        diffs = [
            v["abtt"] - v["baseline"]
            for v in examples.values()
            if "abtt" in v and "baseline" in v
            and v["abtt"] is not None and v["baseline"] is not None
            and np.isfinite(v["abtt"]) and np.isfinite(v["baseline"])
        ]
        if len(diffs) < 2:
            continue
        arr = np.asarray(diffs, dtype=float)
        out[cell] = (float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))))
    return out


def _ties_for(pairs_root: Optional[Path], metric_key: str,
              tie_se: float = 2.0) -> list[str]:
    """Cells whose paired ABTT-minus-baseline difference is inside the noise.

    The published canon caption hardcoded "the LaTa MaRC DelAUC win is a tie at
    about 1.2 standard errors". That was a property of one sample; issue #141
    re-sampled and the narrow cells moved. Reading them off the per-pair cache
    keeps the claim true of whatever summary is passed in.
    """
    if pairs_root is None:
        return []
    if not pairs_root.exists():
        raise FileNotFoundError(
            f"per-pair metric cache not found at {pairs_root}. The caption's tie "
            "clause is computed from paired per-pair differences, so without it "
            "the committed table cannot be reproduced. Re-run "
            "scripts/ig/run_attribution_metrics.py to rebuild the cache (it is "
            "gitignored and regenerable from the tracked NPZs), or pass "
            "--no_tie_clause to render the caption without it."
        )
    stats = paired_cell_stats(pairs_root, metric_key)
    out = []
    for model, model_label in MODELS:
        for method, method_label in METHODS:
            entry = stats.get((model, method))
            if entry is None:
                continue
            mean, se = entry
            if se > 0 and abs(mean / se) < tie_se:
                out.append(f"{model_label} {method_label}")
    return out


def tie_sentence(pairs_root: Optional[Path], summary: pd.DataFrame,
                 metric_key: str, label: str, tie_se: float = 2.0) -> str:
    """One clause naming this metric's tie cells, or the empty string."""
    ties = _ties_for(pairs_root, metric_key, tie_se)
    if not ties:
        return ""
    return f" {label} is a tie for {', '.join(ties)}."


def tie_clause(pairs_root: Optional[Path], tie_se: float = 2.0) -> str:
    """One compact clause covering both main columns.

    Kept to a single sentence: the caption is already long, and two parallel
    sentences saying the same thing about different columns cost a line of the
    page budget for no extra information.
    """
    parts = []
    for key, label in ((RHO_KEY, r"$\rho$"), (DEL_GAP_KEY, "DelAUC gap")):
        ties = _ties_for(pairs_root, key, tie_se)
        if ties:
            parts.append(f"{label} for {' and '.join(ties)}")
    if not parts:
        return ""
    return (f" Ties, within {tie_se:.0f} standard errors of zero: "
            f"{', '.join(parts)}.")


def _random_floor_range(summary: pd.DataFrame) -> tuple[float, float]:
    floors = [
        _get(summary, model, method, variant, _mean_col(DEL_RANDOM_KEY))
        for model, _ in MODELS
        for method, _ in METHODS
        for variant in ("baseline", "abtt")
    ]
    return min(floors), max(floors)


def main_caption(summary: pd.DataFrame, pairs_root: Optional[Path] = None) -> str:
    rho_wins = _wins(summary, RHO_KEY)
    del_wins = _wins(summary, DEL_GAP_KEY)
    lo_n, hi_n = _abtt_pair_count_range(summary, DEL_GAP_KEY)
    base_lo, base_hi = _baseline_pair_count_range(summary, DEL_GAP_KEY)
    lo_floor, hi_floor = _random_floor_range(summary)

    # Which cells are ties is read off the paired per-pair differences, not
    # asserted. See tie_sentence: the published canon caption named LaTa MaRC
    # because that was the narrow cell on that sample, and a different sample
    # has different narrow cells or none.
    ties = tie_clause(pairs_root)

    return (
        r"\caption{Attribution faithfulness at the predeclared operational "
        r"layers, 200 positive pairs per model, for integrated gradients (IG) "
        r"and retrieval-adapted MaRC. $\rho_{\text{LOO}}$ correlates "
        r"attribution magnitude with the leave-one-out change in the cosine. "
        r"DelAUC gap is the random-order minus attribution-order "
        r"deletion-curve area, positive when attribution beats chance; the "
        rf"reference runs from {lo_floor:.3f} to {hi_floor:.3f} here, so zero "
        r"is chance. Higher is better in both; "
        rf"boldface marks the better variant. ABTT wins {rho_wins}/6 and "
        rf"{del_wins}/6.{ties} Ratio metrics are undefined below a full-query "
        rf"cosine of 0.05, so the DelAUC columns average {_count_phrase(lo_n, hi_n)} ABTT "
        rf"pairs against {_count_phrase(base_lo, base_hi)} baseline pairs. Cross-variant comparisons are "
        r"descriptive. Secondary metrics: "
        r"Table~\ref{tab:attribution_metrics_secondary}.}"
    )


def render_table(summary: pd.DataFrame, out_path: Path,
                 pairs_root: Optional[Path] = None, *,
                 source_run: Optional[str] = None) -> None:
    lines: list[str] = [
        *_header_lines(source_run),
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{$\rho_{\text{LOO}}$} "
        r"& \multicolumn{2}{c}{DelAUC gap} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Model & Method & base & ABTT & base & ABTT \\",
        r"\midrule",
    ]

    for model, model_label in MODELS:
        first_model_row = True
        for method, method_label in METHODS:
            cells = [model_label if first_model_row else "", method_label]
            cells += _pair_cells(
                _get(summary, model, method, "baseline", _mean_col(RHO_KEY)),
                _get(summary, model, method, "abtt", _mean_col(RHO_KEY)),
            )
            cells += _pair_cells(
                _get(summary, model, method, "baseline", _mean_col(DEL_GAP_KEY)),
                _get(summary, model, method, "abtt", _mean_col(DEL_GAP_KEY)),
            )
            lines.append(" & ".join(cells) + r" \\")
            first_model_row = False
        if model != MODELS[-1][0]:
            lines.append(r"\addlinespace[2pt]")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        main_caption(summary, pairs_root),
        r"\label{tab:attribution_metrics_main}",
        r"\end{table}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


SECONDARY_COLUMNS = (
    (TAU_KEY, r"$\tau_{\text{LOO}}$", False),
    (INS_GAP_KEY, r"InsAUC gap", False),
    (SUFF_KEY, r"Suff@25\%", False),
    (COMP_KEY, r"Comp@25\%", False),
    (MINFRAC_KEY, r"MinFrac@0.80", True),
)


SHUFFLE_GAP_KEY = "rand_ins_auc_gap_gap"


def _shuffle_failures(summary: pd.DataFrame, gap_key: str) -> list[str]:
    """Cells where the real attribution does not beat a shuffle of its own scores.

    Criterion 5 of the selection memo, read off the summary instead of quoted
    from it. The published caption said "two of the twelve cells, both of them
    baseline cells"; both halves of that are properties of one sample.
    """
    failures = []
    for model, model_label in MODELS:
        for method, method_label in METHODS:
            for variant in ("baseline", "abtt"):
                value = _get(summary, model, method, variant, _mean_col(gap_key))
                if value <= 0:
                    failures.append(f"{model_label} {method_label} {variant}")
    return failures


def secondary_caption(summary: pd.DataFrame) -> str:
    tau_wins = _wins(summary, TAU_KEY)
    ins_wins = _wins(summary, INS_GAP_KEY)
    lo_n, hi_n = _abtt_pair_count_range(summary, INS_GAP_KEY)
    base_lo, base_hi = _baseline_pair_count_range(summary, INS_GAP_KEY)
    failures = _shuffle_failures(summary, SHUFFLE_GAP_KEY)
    n_fail = len(failures)
    all_baseline = failures and all(f.endswith("baseline") for f in failures)
    fail_clause = (
        f"in {_NUMBER_WORDS.get(n_fail, str(n_fail))} of the twelve cells"
        + (", all of them baseline cells," if all_baseline and n_fail > 1
           else ", a baseline cell," if all_baseline else ",")
    )
    return (
        r"\caption{Secondary attribution metrics, on the same pairs, the same "
        r"layers and the same erasure operator as "
        r"Table~\ref{tab:attribution_metrics_main}. Boldface marks the better "
        r"variant within a pair; higher is better everywhere except MinFrac. "
        r"None of these columns is in the main table, and each is out for its "
        r"own reason. Kendall $\tau_b$ agrees with $\rho_{\text{LOO}}$ in "
        rf"{tau_wins}/6 cells, but it is the tie-corrected twin of the same "
        r"statistic rather than a second witness. Chance-corrected insertion "
        rf"faithfulness favours ABTT in {ins_wins}/6 cells, and we do not "
        rf"report it in the main table because {fail_clause} the real "
        r"attribution does not beat a "
        r"permutation of its own scores "
        r"(Table~\ref{tab:attribution_shuffle_control}), so the measurement "
        r"does not meet the validity bar we set for a headline column. The threshold-based "
        r"ERASER metrics are reported for completeness: their "
        r"baseline-versus-ABTT verdict depends on the threshold and on the "
        r"erasure operator, which is why the main table uses threshold-free, "
        r"chance-corrected metrics instead. The InsAUC columns average "
        rf"{_count_phrase(lo_n, hi_n)} ABTT pairs against the baseline's "
        rf"{_count_phrase(base_lo, base_hi)}, for the "
        r"same small-denominator reason. Full threshold sweeps are in "
        r"Tables~\ref{tab:attribution_sweep_main_methods} "
        r"and~\ref{tab:attribution_sweep_supplemental_methods}.}"
    )


def render_secondary_table(summary: pd.DataFrame, out_path: Path, *,
                           source_run: Optional[str] = None) -> None:
    n_metrics = len(SECONDARY_COLUMNS)
    banner = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in SECONDARY_COLUMNS
    )
    rules = "".join(
        rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(n_metrics)
    )
    lines: list[str] = [
        *_header_lines(source_run),
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{ll" + "rr" * n_metrics + "}",
        r"\toprule",
        r"& & " + banner + r" \\",
        rules,
        r"Model & Method & " + " & ".join(["base", "ABTT"] * n_metrics) + r" \\",
        r"\midrule",
    ]

    for model, model_label in MODELS:
        first_model_row = True
        for method, method_label in METHODS:
            cells = [model_label if first_model_row else "", method_label]
            for key, _, lower_is_better in SECONDARY_COLUMNS:
                cells += _pair_cells(
                    _get(summary, model, method, "baseline", _mean_col(key)),
                    _get(summary, model, method, "abtt", _mean_col(key)),
                    lower_is_better=lower_is_better,
                )
            lines.append(" & ".join(cells) + r" \\")
            first_model_row = False
        if model != MODELS[-1][0]:
            lines.append(r"\addlinespace[2pt]")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        secondary_caption(summary),
        r"\label{tab:attribution_metrics_secondary}",
        r"\end{table*}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _cell_label(model_label: str, method_label: str, variant: str) -> str:
    return f"{model_label} {method_label} {variant}"


def check_shuffle_identities(summary: pd.DataFrame,
                             tol: float = SHUFFLE_IDENTITY_TOL) -> None:
    """Fail if two metrics that share a control column do not share its numbers.

    The caption says DelAUC gap and AOPC-Comp, and InsAUC gap and AOPC-Suff,
    have identical shuffle gaps by construction (memo A3). That is a property
    of the implementation in ``src/attribution_metrics.py``, so it is checked
    on every summary rather than asserted once.
    """
    # The table prints the first key's mean and SE, so both must agree.
    for keys, label, _ in SHUFFLE_COLUMNS:
        if len(keys) < 2:
            continue
        first = keys[0]
        for other in keys[1:]:
            for model, model_label in MODELS:
                for method, method_label in METHODS:
                    for variant in ("baseline", "abtt"):
                        for stat in ("mean", "se"):
                            a = _get(summary, model, method, variant, _shuffle_gap_col(first, stat))
                            b = _get(summary, model, method, variant, _shuffle_gap_col(other, stat))
                            if abs(a - b) > tol:
                                raise ValueError(
                                    f"shuffle gap {stat} for {first} and {other} differ by "
                                    f"{abs(a - b):.3g} in "
                                    f"{_cell_label(model_label, method_label, variant)}; the "
                                    f"'{label}' column claims they are identical by construction"
                                )


def _shuffle_cells(summary: pd.DataFrame, metric_key: str):
    """(cell label, gap mean, gap se) over the twelve cells, in table order."""
    for model, model_label in MODELS:
        for method, method_label in METHODS:
            for variant in ("baseline", "abtt"):
                yield (
                    _cell_label(model_label, method_label, variant),
                    _get(summary, model, method, variant, _shuffle_gap_col(metric_key, "mean")),
                    _get(summary, model, method, variant, _shuffle_gap_col(metric_key, "se")),
                )


def _shuffle_count_range(summary: pd.DataFrame, metric_key: str,
                         variant: str) -> tuple[int, int]:
    counts = [
        int(_get(summary, model, method, variant, _shuffle_gap_col(metric_key, "n")))
        for model, _ in MODELS
        for method, _ in METHODS
    ]
    return min(counts), max(counts)


def _fmt_signed(value: float) -> str:
    return f"{value:+.3f}"


def _join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def shuffle_control_caption(summary: pd.DataFrame, shuffle_draws: int) -> str:
    draws_word = _NUMBER_WORDS.get(shuffle_draws, str(shuffle_draws))
    failing: list[str] = []
    marginal: list[str] = []
    for keys, _, short in SHUFFLE_COLUMNS:
        for cell, mean, se in _shuffle_cells(summary, keys[0]):
            if mean <= 0:
                failing.append(f"{cell} on {short}, at {_fmt_signed(mean)}")
            elif se > 0 and mean / se < 2.0:
                marginal.append(f"{cell} on {short}")
    if failing:
        n_fail = len(failing)
        fail_text = (
            rf" Boldface marks the {_NUMBER_WORDS.get(n_fail, str(n_fail))} "
            rf"cell{'s' if n_fail > 1 else ''} at or below zero: {_join_names(failing)}."
        )
    else:
        fail_text = " Every cell is positive."
    if marginal:
        n_marg = len(marginal)
        marginal_text = (
            rf" {_NUMBER_WORDS.get(n_marg, str(n_marg)).capitalize()} further "
            rf"cell{'s are' if n_marg > 1 else ' is'} positive but within two "
            rf"standard errors of zero: {_join_names(marginal)}."
        )
    else:
        marginal_text = ""
    rho_lo, rho_hi = _shuffle_count_range(summary, RHO_KEY, "baseline")
    rho_alo, rho_ahi = _shuffle_count_range(summary, RHO_KEY, "abtt")
    auc_blo, auc_bhi = _shuffle_count_range(summary, DEL_GAP_KEY, "baseline")
    auc_alo, auc_ahi = _shuffle_count_range(summary, DEL_GAP_KEY, "abtt")
    rank_count = _count_phrase(min(rho_lo, rho_alo), max(rho_hi, rho_ahi))
    return (
        r"\caption{Shuffled-attribution control for the six candidate metrics, "
        r"on the same pairs, layers and erasure operator as "
        r"Table~\ref{tab:attribution_metrics_main}. Each entry is the real "
        rf"metric minus its mean over {draws_word} permutations of the same "
        r"attribution's scores across the query tokens, with the standard "
        r"error over pairs in parentheses: a positive gap means the real "
        r"attribution beats a fake one drawn from its own score distribution, "
        r"and a metric passes the control only when every cell is positive. "
        r"DelAUC gap and AOPC-Comprehensiveness share one column, as do InsAUC "
        r"gap and AOPC-Sufficiency, because the random-order reference and the "
        r"constant trapezoid offset cancel in the difference, so each pair has "
        rf"the same gap to machine precision and one verdict.{fail_text}"
        rf"{marginal_text} The rank columns use {rank_count} pairs per cell; "
        r"the AUC columns, undefined below a full-query cosine of 0.05, "
        rf"average {_count_phrase(auc_alo, auc_ahi)} ABTT pairs against "
        rf"{_count_phrase(auc_blo, auc_bhi)} baseline pairs.}}"
    )


def render_shuffle_control_table(summary: pd.DataFrame, out_path: Path, *,
                                 source_run: Optional[str] = None,
                                 shuffle_draws: int = SHUFFLE_DRAWS_OF_RECORD) -> None:
    check_shuffle_identities(summary)
    n_cols = len(SHUFFLE_COLUMNS)
    lines: list[str] = [
        *_header_lines(source_run),
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lll" + "r" * n_cols + "}",
        r"\toprule",
        r"Model & Method & Variant & " + " & ".join(label for _, label, _ in SHUFFLE_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for model, model_label in MODELS:
        first_model_row = True
        for method, method_label in METHODS:
            first_method_row = True
            for variant, variant_label in (("baseline", "base"), ("abtt", "ABTT")):
                cells = [
                    model_label if first_model_row else "",
                    method_label if first_method_row else "",
                    variant_label,
                ]
                for keys, _, _ in SHUFFLE_COLUMNS:
                    mean = _get(summary, model, method, variant, _shuffle_gap_col(keys[0], "mean"))
                    se = _get(summary, model, method, variant, _shuffle_gap_col(keys[0], "se"))
                    text = f"{_fmt_signed(mean)} ({se:.3f})"
                    cells.append(rf"\textbf{{{text}}}" if mean <= 0 else text)
                lines.append(" & ".join(cells) + r" \\")
                first_model_row = False
                first_method_row = False
        if model != MODELS[-1][0]:
            lines.append(r"\addlinespace[2pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        shuffle_control_caption(summary, shuffle_draws),
        r"\label{tab:attribution_shuffle_control}",
        r"\end{table*}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")



def rho_figure_caption(summary: pd.DataFrame,
                       pairs_root: Optional[Path] = None) -> str:
    """The figure's win count and tie cell, read off the same statistics as
    the table caption.

    The published figure caption hardcoded "in all six cells" and survived the
    #187 re-sample while the table caption, computed from ``_wins`` and
    ``_ties_for``, said five with PhilTa MaRC a tie. Both captions now come
    from the same two functions, so they cannot disagree again.
    """
    wins = _wins(summary, RHO_KEY)
    tie_labels = set(_ties_for(pairs_root, RHO_KEY))
    not_won = []
    for model, model_label in MODELS:
        for method, method_label in METHODS:
            base = _get(summary, model, method, "baseline", _mean_col(RHO_KEY))
            abtt = _get(summary, model, method, "abtt", _mean_col(RHO_KEY))
            if not abtt > base:
                not_won.append((f"{model_label} {method_label}", base, abtt))
    n_cells = len(MODELS) * len(METHODS)

    lead = (
        r"\caption{Leave-one-out rank correlation $\rho_{\text{LOO}}$ at the "
        r"predeclared operational attribution layers, for integrated gradients "
        r"(IG) and retrieval-adapted MaRC. Each line connects the baseline and "
        r"ABTT variants of one model-method cell. "
    )
    if wins == n_cells:
        wins_text = rf"ABTT raises $\rho_{{\text{{LOO}}}}$ in all {_NUMBER_WORDS[n_cells]} cells."
    else:
        parts = []
        for label, base, abtt in not_won:
            verdict = ""
            if pairs_root is not None:
                verdict = (", a tie within two standard errors" if label in tie_labels
                           else ", a loss")
            parts.append(f"{label}, moves from {_fmt(base)} to {_fmt(abtt)}{verdict}")
        if len(not_won) == 1:
            rest = f"the {_NUMBER_WORDS[n_cells]}th, {parts[0]}"
        else:
            rest = "the others: " + "; ".join(parts)
        wins_text = (
            rf"ABTT raises $\rho_{{\text{{LOO}}}}$ in {_NUMBER_WORDS[wins]} of the "
            rf"{_NUMBER_WORDS[n_cells]} cells; {rest} "
            r"(Table~\ref{tab:attribution_metrics_main})."
        )
    return (lead + wins_text
            + r" Secondary metrics: Table~\ref{tab:attribution_metrics_secondary}.}")


def render_rho_figure(summary: pd.DataFrame, out_base: Path,
                      pairs_root: Optional[Path] = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows: list[dict[str, object]] = []
    for model, model_label in MODELS:
        for method, method_label in METHODS:
            rows.append(
                {
                    "label": f"{model_label} / {method_label}",
                    "method": method,
                    "base": _get(summary, model, method, "baseline", _mean_col(RHO_KEY)),
                    "abtt": _get(summary, model, method, "abtt", _mean_col(RHO_KEY)),
                }
            )

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    y = np.arange(len(rows))[::-1]
    colors = {"ig": "#0072B2", "retrieval_mark": "#D55E00"}

    for yi, row in zip(y, rows):
        color = colors[str(row["method"])]
        base = float(row["base"])
        abtt = float(row["abtt"])
        ax.plot([base, abtt], [yi, yi], color=color, linewidth=1.4, alpha=0.85)
        ax.scatter(base, yi, s=28, facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)
        ax.scatter(abtt, yi, s=34, marker="D", facecolor=color, edgecolor=color, zorder=3)

    ax.axvline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["label"]) for row in rows], fontsize=8)
    ax.set_xlabel(r"$\rho_{\mathrm{LOO}}$ (higher is better)", fontsize=9)
    ax.set_xlim(-0.08, 0.68)
    ax.grid(axis="x", color="0.86", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.25",
            markerfacecolor="white",
            markeredgecolor="0.25",
            linewidth=0,
            label="baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="0.25",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            linewidth=0,
            label="ABTT",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.4)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
        dpi=300,
        metadata={"CreationDate": None},
    )
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    tex = "\n".join(
        [
            "% generated figure",
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{figures/{out_base.name}.pdf}}",
            rho_figure_caption(summary, pairs_root),
            r"\label{fig:attribution_rho_loo_main}",
            r"\end{figure}",
            "",
        ]
    )
    out_base.with_suffix(".tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--no_tie_clause", action="store_true",
        help="Render the caption without the tie clause. Only for a run with no "
             "per-pair cache; the committed table is built with the clause.",
    )
    parser.add_argument(
        "--pairs_root", type=Path, default=None,
        help="Per-pair metric JSON cache from run_attribution_metrics.py. Used "
             "for the caption's paired standard errors. Defaults to "
             "<summary_csv parent>/v2_hidden.",
    )
    parser.add_argument("--table_out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--secondary_table_out", type=Path, default=DEFAULT_SECONDARY_OUT)
    parser.add_argument("--shuffle_table_out", type=Path, default=DEFAULT_SHUFFLE_OUT)
    parser.add_argument("--fig_out_base", type=Path, default=DEFAULT_FIG_OUT)
    parser.add_argument(
        "--allow_run_change", action="store_true",
        help="Overwrite a table stamped with a different source run. Only when "
             "the run of record is changing on purpose; see attribution_run_of_record.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pairs_root = None if args.no_tie_clause else (
        args.pairs_root or args.summary_csv.parent / "v2_hidden"
    )
    run = run_name(args.summary_csv)
    for out in (args.table_out, args.secondary_table_out, args.shuffle_table_out):
        refuse_run_change(out, run, allow=args.allow_run_change)
    summary = _load_main_rows(args.summary_csv)
    render_table(summary, args.table_out, args.pairs_root, source_run=run)
    render_secondary_table(summary, args.secondary_table_out, source_run=run)
    render_shuffle_control_table(summary, args.shuffle_table_out, source_run=run)
    render_rho_figure(summary, args.fig_out_base, args.pairs_root)
    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.secondary_table_out}")
    print(f"Wrote {args.shuffle_table_out}")
    print(f"Wrote {args.fig_out_base.with_suffix('.pdf')}")
    print(f"Wrote {args.fig_out_base.with_suffix('.png')}")
    print(f"Wrote {args.fig_out_base.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
