"""Diff two attribution-metric summaries cell by cell.

Issue #141 re-sampled the 200-positive-pair attribution set on benchmark v1
after #113 found that the published sample came from the legacy phase-9 split
over ``data/canon``. The question the paper needs answered is not "did the
numbers move" (they must, the pairs are different files) but "did any
**verdict** move": in how many of the six model-view cells does ``rho_LOO``
favour ABTT, likewise ``DelAUC gap``, and do the two sufficiency-side metrics
still fail the shuffled-attribution control. On the benchmark v1 sample the
answers are 5/6 (was 6/6), 4/6 (was 3/6) and yes.

Emits a markdown table per metric, with the old and new baseline -> ABTT means
and the win marker, so the memo can be written straight from the output.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS: Tuple[Tuple[str, str], ...] = (
    ("bowphs/LaTa", "LaTa"),
    ("bowphs/PhilTa", "PhilTa"),
    ("google/mt5-base", "mT5-base"),
)
VIEWS: Tuple[Tuple[str, str], ...] = (("ig", "IG"), ("retrieval_mark", "MaRC"))

# (summary key, display name, higher_is_better)
METRICS: Tuple[Tuple[str, str, bool], ...] = (
    ("loo_rho_mean", "rho_LOO", True),
    ("del_auc_gap_mean", "DelAUC gap", True),
    ("ins_auc_gap_mean", "InsAUC gap", True),
    ("loo_tau_mean", "tau_LOO", True),
    ("aopc_suff_ratio_mean", "AOPC-Suff", True),
    ("aopc_comp_ratio_mean", "AOPC-Comp", True),
    ("del_auc_mean", "DelAUC", False),
    ("ins_auc_mean", "InsAUC", True),
    ("del_auc_random_mean", "DelAUC random floor", True),
    ("suff@0.25_ratio_mean", "Suff@25%", True),
    ("comp@0.25_ratio_mean", "Comp@25%", True),
    ("compactness@0.80_mean", "MinFrac@0.80", False),
)

# The shuffled-attribution control: criterion 5 of the decision memo.
SHUFFLE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("rand_loo_rho_gap_mean", "rho_LOO"),
    ("rand_loo_tau_gap_mean", "tau_LOO"),
    ("rand_del_auc_gap_gap_mean", "DelAUC gap"),
    ("rand_ins_auc_gap_gap_mean", "InsAUC gap"),
    ("rand_aopc_suff_ratio_gap_mean", "AOPC-Suff"),
    ("rand_aopc_comp_ratio_gap_mean", "AOPC-Comp"),
)


def cell(summary: pd.DataFrame, model: str, method: str, variant: str, key: str) -> float:
    rows = summary[
        (summary["model"] == model)
        & (summary["method"] == method)
        & (summary["variant"] == variant)
    ]
    if len(rows) != 1:
        raise SystemExit(
            f"expected 1 row for ({model}, {method}, {variant}), found {len(rows)}"
        )
    if key not in rows.columns:
        raise SystemExit(f"summary has no column {key!r}")
    return float(rows.iloc[0][key])


def win(base: float, abtt: float, higher_is_better: bool) -> str:
    better = abtt > base if higher_is_better else abtt < base
    return "A" if better else "b"


def metric_table(old: pd.DataFrame, new: pd.DataFrame) -> List[str]:
    header = ["| Metric | Dir | Cell | old base -> ABTT | new base -> ABTT | old | new |",
              "|---|---|---|---|---|:-:|:-:|"]
    lines = list(header)
    for key, name, higher in METRICS:
        for model, model_label in MODELS:
            for method, view_label in VIEWS:
                ob = cell(old, model, method, "baseline", key)
                oa = cell(old, model, method, "abtt", key)
                nb = cell(new, model, method, "baseline", key)
                na = cell(new, model, method, "abtt", key)
                ow, nw = win(ob, oa, higher), win(nb, na, higher)
                flip = " **flip**" if ow != nw else ""
                lines.append(
                    f"| {name} | {'up' if higher else 'down'} "
                    f"| {model_label}/{view_label} "
                    f"| {ob:.3f} -> {oa:.3f} | {nb:.3f} -> {na:.3f} "
                    f"| {ow} | {nw}{flip} |"
                )
    return lines


def wins_table(old: pd.DataFrame, new: pd.DataFrame) -> List[str]:
    lines = ["| Metric | Dir | old ABTT wins | new ABTT wins | cells that flipped |",
             "|---|---|---|---|---|"]
    for key, name, higher in METRICS:
        old_wins = new_wins = 0
        flipped: List[str] = []
        for model, model_label in MODELS:
            for method, view_label in VIEWS:
                ow = win(
                    cell(old, model, method, "baseline", key),
                    cell(old, model, method, "abtt", key),
                    higher,
                )
                nw = win(
                    cell(new, model, method, "baseline", key),
                    cell(new, model, method, "abtt", key),
                    higher,
                )
                old_wins += ow == "A"
                new_wins += nw == "A"
                if ow != nw:
                    flipped.append(f"{model_label}/{view_label} {ow}->{nw}")
        lines.append(
            f"| {name} | {'higher' if higher else 'lower'} | {old_wins}/6 | "
            f"{new_wins}/6 | {', '.join(flipped) if flipped else '-'} |"
        )
    return lines


def shuffle_table(old: pd.DataFrame, new: pd.DataFrame) -> List[str]:
    """Criterion 5: positive gap in all 12 cells (3 models x 2 views x 2 variants)."""
    lines = ["| Metric | old positive cells | new positive cells | old failures | new failures |",
             "|---|---|---|---|---|"]
    for key, name in SHUFFLE_KEYS:
        counts: Dict[str, int] = {"old": 0, "new": 0}
        failures: Dict[str, List[str]] = {"old": [], "new": []}
        for label, summary in (("old", old), ("new", new)):
            for model, model_label in MODELS:
                for method, view_label in VIEWS:
                    for variant in ("baseline", "abtt"):
                        value = cell(summary, model, method, variant, key)
                        if value > 0:
                            counts[label] += 1
                        else:
                            failures[label].append(
                                f"{model_label}/{view_label} {variant} ({value:+.3f})"
                            )
        lines.append(
            f"| {name} | {counts['old']}/12 | {counts['new']}/12 | "
            f"{'; '.join(failures['old']) or 'none'} | "
            f"{'; '.join(failures['new']) or 'none'} |"
        )
    return lines


def render(old: pd.DataFrame, new: pd.DataFrame) -> str:
    blocks: Sequence[Tuple[str, List[str]]] = (
        ("## Wins per metric, old vs new", wins_table(old, new)),
        ("## Shuffled-attribution control (criterion 5)", shuffle_table(old, new)),
        ("## Every cell", metric_table(old, new)),
    )
    out: List[str] = []
    for title, lines in blocks:
        out.append(title)
        out.append("")
        out.extend(lines)
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old_summary",
        type=Path,
        default=REPO_ROOT
        / "runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv",
    )
    parser.add_argument(
        "--new_summary",
        type=Path,
        default=REPO_ROOT
        / "runs/active/ig_examples_200pos_v1/attribution_metrics/summary_v2.csv",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    text = render(pd.read_csv(args.old_summary), pd.read_csv(args.new_summary))
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
