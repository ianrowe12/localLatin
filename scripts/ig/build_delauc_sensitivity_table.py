"""Appendix table: DelAUC-gap win/tie/loss under every knob setting.

Issue #195 (G9). Reads the two CSVs written by
``scripts/ig/run_delauc_sensitivity.py`` and renders one LaTeX table: a row per
configuration, the knob it moves, the win/tie/loss count over the six
model-view cells, and the range the baseline and ABTT cell gaps span under that
setting.

The table is generated, never hand-edited, and it is deliberately not
``\\input`` anywhere yet: issue #195 is an analysis, and whether the appendix
carries it is a separate decision.

Usage:

    python scripts/ig/build_delauc_sensitivity_table.py \\
        --configs_csv runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity/configs.csv \\
        --cells_csv   runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity/cells.csv \\
        --out overleaf_drafts/tables/attribution_delauc_sensitivity.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

# Human-readable labels for the knob axis, in the order the table prints them.
KNOB_LABELS = {
    "-": "Predeclared",
    "schedule": "Deletion step schedule",
    "erasure": "Deleted token becomes",
    "draws": "Random orderings",
    "seed": "Random-order seed",
    "token_filter": "Token filter",
    "side": "Side erased",
    "mixed": "Combined",
}

SETTING_LABELS = {
    "predeclared": "every token, drop, 5 draws, query only",
    "sched_frac0.05": "every 5\\% of the query",
    "sched_frac0.10": "every 10\\% of the query",
    "sched_frac0.20": "every 20\\% of the query",
    "erase_zero": "zero vector, denominator kept",
    "erase_centroid": "corpus mean vector",
    "draws1": "1 draw",
    "draws20": "20 draws",
    "draws50": "50 draws",
    "seed20260101": "seed 20260101",
    "seed7": "seed 7",
    "filter_all": "none (mismatches the pooling)",
    "filter_no_empty": "no\\_empty (mismatches the pooling)",
    "side_both": "query and candidate",
    "sched0.10_zero": "10\\% grid + zero vector",
    "sched0.10_centroid": "10\\% grid + corpus mean",
    "sched0.10_both": "10\\% grid + both sides",
    "zero_both": "zero vector + both sides",
    "centroid_both": "corpus mean + both sides",
    "sched0.10_zero_both": "10\\% grid + zero + both sides",
}


def fmt(value: float, places: int = 3) -> str:
    """Fixed-point, with negative zero printed as zero.

    A cell gap of -0.0001 rounds to "-0.00", which reads as a sign rather than
    as the rounding it is.
    """
    if pd.isna(value):
        return "--"
    text = f"{value:.{places}f}"
    if float(text) == 0.0:
        return f"{0.0:.{places}f}"
    return text


def range_cell(lo: float, hi: float) -> str:
    return f"{fmt(lo, 2)} to {fmt(hi, 2)}"


def setting_label(name: str) -> str:
    return SETTING_LABELS.get(name, name.replace("_", "\\_"))


def build_rows(configs: pd.DataFrame) -> list[str]:
    """One LaTeX row per configuration, with a rule between knob blocks."""
    rows: list[str] = []
    previous_knob: Optional[str] = None
    for cfg in configs.itertuples():
        if previous_knob is not None and cfg.knob != previous_knob:
            rows.append("\\addlinespace[2pt]")
        knob_cell = KNOB_LABELS.get(cfg.knob, cfg.knob) if cfg.knob != previous_knob else ""
        wtl = f"{cfg.wins}/{cfg.ties}/{cfg.losses}"
        if cfg.config == "predeclared":
            wtl = f"\\textbf{{{wtl}}}"
        rows.append(
            f"{knob_cell} & {setting_label(cfg.config)} & {wtl} & "
            f"{range_cell(cfg.gap_base_min, cfg.gap_base_max)} & "
            f"{range_cell(cfg.gap_abtt_min, cfg.gap_abtt_max)} & "
            f"{fmt(cfg.mean_abs_paired)} \\\\"
        )
        previous_knob = cfg.knob
    return rows


def caption(configs: pd.DataFrame, cells: pd.DataFrame) -> str:
    base = configs[configs["config"] == "predeclared"].iloc[0]
    valid = configs[configs["pooling_matches_generator"]]
    wins = sorted(valid["wins"].tolist())
    n_filter = int((~configs["pooling_matches_generator"]).sum())
    min_pairs = int(cells["n_pairs"].min())
    max_pairs = int(cells["n_pairs"].max())
    return (
        "Sensitivity of the chance-corrected deletion AUC gap to the choices "
        "behind it, over the same 200 positive pairs per model and the same "
        "stored hidden states as Table~\\ref{tab:attribution_metrics_main}. "
        "Each row recomputes the metric with one knob moved and everything "
        "else at the predeclared setting in the first row. W/T/L counts the "
        "six model-view cells in which ABTT beats, ties or loses to the "
        "uncorrected baseline, a tie being a paired difference within two "
        "standard errors of zero. The two range columns give the smallest and "
        "largest cell gap that setting produces for each variant, which is "
        "the spread the main table reports cell by cell. The last column is "
        "the mean absolute paired difference over the six cells. Across the "
        f"{len(valid)} settings that pool the same vectors the components were "
        f"fitted on, ABTT wins {wins[0]} to {wins[-1]} of six cells and the "
        f"predeclared setting gives {int(base.wins)}. The {n_filter} token-filter "
        "rows are diagnostics, not candidate settings: the artifacts were "
        "generated by pooling one filtered token set, so any other filter "
        "scores a vector the ABTT components were not fitted on. Cells average "
        f"{min_pairs} to {max_pairs} pairs, the rest falling below the "
        "full-query cosine floor of 0.05 at which the ratio is undefined."
    )


def render(configs: pd.DataFrame, cells: pd.DataFrame) -> str:
    body = "\n".join(build_rows(configs))
    return "\n".join([
        "% generated by scripts/ig/build_delauc_sensitivity_table.py",
        "% Source: runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity/",
        "% Analysis memo: docs/research/delauc_sensitivity.md (issue #195).",
        "% Not \\input anywhere yet: see the memo before wiring it into the appendix.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Knob & Setting & W/T/L & base gap & ABTT gap & $|\\Delta|$ \\\\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption(configs, cells)}}}",
        "\\label{tab:attribution_delauc_sensitivity}",
        "\\end{table*}",
        "",
    ])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = "runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity"
    p.add_argument("--configs_csv", default=f"{root}/configs.csv")
    p.add_argument("--cells_csv", default=f"{root}/cells.csv")
    p.add_argument("--out", default="overleaf_drafts/tables/attribution_delauc_sensitivity.tex")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configs = pd.read_csv(args.configs_csv)
    cells = pd.read_csv(args.cells_csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(configs, cells))
    print(f"Wrote {out} ({len(configs)} configurations, {len(cells)} cells)")


if __name__ == "__main__":
    main()
