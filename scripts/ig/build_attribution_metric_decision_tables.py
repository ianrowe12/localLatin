"""Render the tables behind docs/research/attribution_metrics_decision.md.

Reads the wide summary produced by ``scripts/ig/run_attribution_metrics.py``
(one row per model x method x variant) and emits, as Markdown on stdout:

  1. a full per-cell table for every metric, old and new;
  2. a baseline-vs-ABTT wins table, one row per metric;
  3. the ``random`` / ``inverse`` control rows;
  4. the randomization-check gaps.

Nothing here decides anything: the selection argument lives in the memo. This
script exists so every number in that memo is regenerable from one command and
traceable to one CSV.

Usage:
  python scripts/ig/build_attribution_metric_decision_tables.py \\
      --summary runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv \\
      --wins_csv runs/active/ig_examples_200pos_run3_operational/attribution_metrics/metric_wins_v2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUMMARY = (
    REPO_ROOT
    / "runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv"
)

MODELS: Tuple[Tuple[str, str], ...] = (
    ("bowphs/LaTa", "LaTa"),
    ("bowphs/PhilTa", "PhilTa"),
    ("google/mt5-base", "mT5-base"),
)
VIEWS: Tuple[Tuple[str, str], ...] = (
    ("ig", "IG"),
    ("retrieval_mark", "MaRC"),
)
CONTROLS: Tuple[Tuple[str, str], ...] = (
    ("random", "random"),
    ("inverse", "inverse"),
)


class MetricSpec(NamedTuple):
    label: str
    key: str
    higher_is_better: bool
    family: str


# Ordered old-first so the memo's "everything we have" table reads as an
# extension of the published one rather than a replacement.
METRIC_SPECS: Tuple[MetricSpec, ...] = (
    MetricSpec("rho_LOO", "loo_rho", True, "existing"),
    MetricSpec("Suff@10%", "suff@0.10_ratio", True, "existing"),
    MetricSpec("Suff@25%", "suff@0.25_ratio", True, "existing"),
    MetricSpec("Suff@50%", "suff@0.50_ratio", True, "existing"),
    MetricSpec("Comp@10%", "comp@0.10_ratio", True, "existing"),
    MetricSpec("Comp@25%", "comp@0.25_ratio", True, "existing"),
    MetricSpec("Comp@50%", "comp@0.50_ratio", True, "existing"),
    MetricSpec("MinFrac@0.70", "compactness@0.70", False, "existing"),
    MetricSpec("MinFrac@0.80", "compactness@0.80", False, "existing"),
    MetricSpec("MinFrac@0.90", "compactness@0.90", False, "existing"),
    MetricSpec("MinFrac@0.95", "compactness@0.95", False, "existing"),
    MetricSpec("tau_LOO", "loo_tau", True, "new"),
    MetricSpec("AOPC-Suff", "aopc_suff_ratio", True, "new"),
    MetricSpec("AOPC-Comp", "aopc_comp_ratio", True, "new"),
    MetricSpec("DelAUC", "del_auc", False, "new"),
    MetricSpec("DelAUC gap", "del_auc_gap", True, "new"),
    MetricSpec("InsAUC", "ins_auc", True, "new"),
    MetricSpec("InsAUC gap", "ins_auc_gap", True, "new"),
    MetricSpec("Rand gap (rho)", "rand_loo_rho_gap", True, "new"),
    MetricSpec("Rand gap (tau)", "rand_loo_tau_gap", True, "new"),
    MetricSpec("Rand gap (AOPC-S)", "rand_aopc_suff_ratio_gap", True, "new"),
    MetricSpec("Rand gap (AOPC-C)", "rand_aopc_comp_ratio_gap", True, "new"),
)


def _mean(df: pd.DataFrame, model: str, method: str, variant: str, key: str) -> float:
    col = f"{key}_mean"
    if col not in df.columns:
        return float("nan")
    row = df[(df["model"] == model) & (df["method"] == method) & (df["variant"] == variant)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def _fmt(v: float) -> str:
    return "--" if pd.isna(v) else f"{v:.3f}"


def _winner(base: float, abtt: float, higher_is_better: bool) -> Optional[str]:
    if pd.isna(base) or pd.isna(abtt):
        return None
    if base == abtt:
        return "tie"
    wins = abtt > base if higher_is_better else abtt < base
    return "ABTT" if wins else "baseline"


def render_full_table(df: pd.DataFrame) -> List[str]:
    lines = [
        "| Metric | Dir | Model | View | baseline | ABTT | Winner |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for spec in METRIC_SPECS:
        arrow = "up" if spec.higher_is_better else "down"
        for model, model_label in MODELS:
            for method, view_label in VIEWS:
                base = _mean(df, model, method, "baseline", spec.key)
                abtt = _mean(df, model, method, "abtt", spec.key)
                win = _winner(base, abtt, spec.higher_is_better) or "--"
                lines.append(
                    f"| {spec.label} | {arrow} | {model_label} | {view_label} | "
                    f"{_fmt(base)} | {_fmt(abtt)} | {win} |"
                )
    return lines


def wins_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for spec in METRIC_SPECS:
        record = {
            "metric": spec.label,
            "key": spec.key,
            "family": spec.family,
            "direction": "higher" if spec.higher_is_better else "lower",
        }
        abtt_wins = 0
        cells = 0
        for model, model_label in MODELS:
            for method, view_label in VIEWS:
                base = _mean(df, model, method, "baseline", spec.key)
                abtt = _mean(df, model, method, "abtt", spec.key)
                win = _winner(base, abtt, spec.higher_is_better)
                record[f"{model_label}/{view_label}"] = (
                    "" if win is None else ("A" if win == "ABTT" else "b")
                )
                if win is None:
                    continue
                cells += 1
                abtt_wins += int(win == "ABTT")
        record["abtt_wins"] = abtt_wins
        record["cells"] = cells
        rows.append(record)
    return pd.DataFrame(rows)


def render_wins_table(wins: pd.DataFrame) -> List[str]:
    cell_cols = [f"{m}/{v}" for _, m in MODELS for _, v in VIEWS]
    header = "| Metric | Dir | Family | " + " | ".join(cell_cols) + " | ABTT wins |"
    sep = "|---|---|---|" + "|".join([":-:"] * len(cell_cols)) + "|---|"
    lines = [header, sep]
    for _, r in wins.iterrows():
        cells = " | ".join(str(r.get(c, "")) for c in cell_cols)
        lines.append(
            f"| {r['metric']} | {r['direction']} | {r['family']} | {cells} | "
            f"{int(r['abtt_wins'])}/{int(r['cells'])} |"
        )
    return lines


def render_controls_table(df: pd.DataFrame, keys: Sequence[str]) -> List[str]:
    specs = [s for s in METRIC_SPECS if s.key in keys]
    lines = [
        "| Model | Row | Variant | " + " | ".join(s.label for s in specs) + " |",
        "|---|---|---|" + "|".join(["--:"] * len(specs)) + "|",
    ]
    for model, model_label in MODELS:
        for method, label in tuple(VIEWS) + CONTROLS:
            for variant in ("baseline", "abtt"):
                vals = " | ".join(_fmt(_mean(df, model, method, variant, s.key)) for s in specs)
                lines.append(f"| {model_label} | {label} | {variant} | {vals} |")
    return lines


def render_drift_note(df: pd.DataFrame) -> List[str]:
    if "full_cos_drift_mean" not in df.columns:
        return []
    worst = float(df["full_cos_drift_mean"].max())
    return [
        "",
        f"Largest mean |full cosine - stored `cos_orig_*`| over all cells: {worst:.3e}.",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    p.add_argument("--wins_csv", default=None,
                   help="Optional path to also write the wins table as CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)

    print("## Per-cell results (all metrics)\n")
    print("\n".join(render_full_table(df)))

    wins = wins_frame(df)
    print("\n## Baseline vs ABTT wins (A = ABTT wins the cell, b = baseline wins)\n")
    print("\n".join(render_wins_table(wins)))

    print("\n## Controls\n")
    print("\n".join(render_controls_table(
        df,
        keys=("loo_rho", "loo_tau", "aopc_suff_ratio", "aopc_comp_ratio",
              "del_auc_gap", "ins_auc_gap"),
    )))
    print("\n".join(render_drift_note(df)))

    if args.wins_csv:
        out = Path(args.wins_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        wins.to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
