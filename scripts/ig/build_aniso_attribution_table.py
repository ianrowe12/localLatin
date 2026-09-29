"""Appendix table: attribution faithfulness at the most anisotropic layers (issue #227).

The run of record (``RUN_OF_RECORD`` in ``attribution_run_of_record.py``) scores
integrated gradients and retrieval-adapted MaRC at the operational layers the
train-only retrieval rule selects: LaTa 7, PhilTa 1, mT5-base 1. On PhilTa and
mT5-base layer 1 the baseline already separates equivalent pairs (test AUROC
0.939 and 0.822), so only LaTa is explained inside the collapse. Issue #227 ran
the same 600 pairs, the same methods and the same settings again at each
model's most anisotropic layer, the label-free mechanism-check layers the paper
already names (LaTa 8, PhilTa 6, mT5-base 5):

    runs/active/ig_examples_200pos_v1_aniso/attribution_metrics_draws20/summary_v2.csv

This generator puts the two layer sets side by side, one row per (model, layer,
method), with the two main-table columns (``rho_LOO`` and ``DelAUC gap``) as
base/ABTT pairs plus the paired ABTT-minus-baseline difference and its ratio to
the paired standard error, which is the statistic the main table's 2-SE tie rule
reads. The layers printed are read from each run's examples CSV, not assumed.

Output:

    overleaf_drafts/tables/attribution_metrics_aniso.tex

The table carries a ``% source run:`` stamp naming the anisotropic run and a
``% reference run:`` line naming the run of record it is compared with. Like
the main generator, it refuses to overwrite a table stamped with a different
source unless ``--allow_run_change`` is passed. It does not touch the run of
record or its tables.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from attribution_run_of_record import (  # noqa: E402
    DEFAULT_SUMMARY_CSV,
    refuse_run_change,
    run_name,
    stamp_line,
)
from build_main_attribution_artifacts import (  # noqa: E402
    _NUMBER_WORDS,
    DEL_GAP_KEY,
    METHODS,
    MODELS,
    RHO_KEY,
    _abtt_pair_count_range,
    _shuffle_gap_col,
    _baseline_pair_count_range,
    _count_phrase,
    _get,
    _mean_col,
    paired_cell_stats,
    select_main_rows,
)

ANISO_RUN = "ig_examples_200pos_v1_aniso"
DEFAULT_ANISO_SUMMARY = (
    REPO_ROOT / "runs/active" / ANISO_RUN / "attribution_metrics_draws20" / "summary_v2.csv"
)
DEFAULT_REFERENCE_SUMMARY = DEFAULT_SUMMARY_CSV
DEFAULT_TABLE_OUT = REPO_ROOT / "overleaf_drafts/tables/attribution_metrics_aniso.tex"

REFERENCE_PREFIX = "% reference run: "
TIE_SE = 2.0

HEADER = "% generated table"
REGEN_NOTE = (
    "% Issue #227: the run of record's pairs, methods and settings at each model's "
    "most\n% anisotropic layer, beside the operational layers. Regenerate with "
    "scripts/ig/build_aniso_attribution_table.py\n% rather than editing the numbers here."
)

Cell = Tuple[str, str]
Stats = Dict[Cell, Tuple[float, float]]


@dataclass(frozen=True)
class LayerSet:
    """One attribution run: its summary rows, per-cell paired stats and layers."""

    tag: str  # short row label, e.g. "op." or "anis."
    summary: pd.DataFrame
    rho: Stats
    del_gap: Stats
    layers: Dict[str, int]


def run_layers(examples_csv: Path) -> Dict[str, int]:
    """The single layer each model was attributed at, read from the run's examples CSV."""
    examples = pd.read_csv(examples_csv)
    out: Dict[str, int] = {}
    for model, _ in MODELS:
        values = sorted({int(v) for v in examples.loc[examples["model_name"] == model, "layer"]})
        if len(values) != 1:
            raise ValueError(f"{examples_csv}: expected one layer for {model}, found {values}")
        out[model] = values[0]
    return out


def verdict(entry: Optional[Tuple[float, float]], tie_se: float = TIE_SE) -> str:
    """``win``, ``tie`` or ``loss`` for ABTT under the main table's 2-SE rule."""
    if entry is None:
        return "missing"
    mean, se = entry
    if se > 0 and abs(mean / se) < tie_se:
        return "tie"
    return "win" if mean > 0 else "loss"


def verdict_counts(stats: Stats, tie_se: float = TIE_SE) -> Dict[str, int]:
    counts = {"win": 0, "tie": 0, "loss": 0}
    for model, _ in MODELS:
        for method, _ in METHODS:
            v = verdict(stats.get((model, method)), tie_se)
            if v in counts:
                counts[v] += 1
    return counts


def _wtl(stats: Stats) -> str:
    c = verdict_counts(stats)
    return f"{c['win']}/{c['tie']}/{c['loss']}"


def _delta_cell(entry: Optional[Tuple[float, float]]) -> str:
    if entry is None:
        return "--"
    mean, se = entry
    z = mean / se if se > 0 else float("nan")
    tie = verdict(entry) == "tie"
    text = f"${mean:+.3f}$ (${z:.1f}$)"
    return text + (r"$^\dagger$" if tie else "")


def _num_cell(value: float, bold: bool) -> str:
    """One per-variant mean; a negative value goes in math mode so it prints a minus sign."""
    if pd.isna(value):
        return "--"
    text = f"{value:.3f}"
    if value < 0:
        return rf"$\mathbf{{{text}}}$" if bold else f"${text}$"
    return rf"\textbf{{{text}}}" if bold else text


def _variant_cells(base: float, abtt: float) -> list:
    """Base and ABTT cells, the better one bolded (higher is better), as in the main table."""
    if pd.isna(base) or pd.isna(abtt):
        return [_num_cell(base, False), _num_cell(abtt, False)]
    abtt_wins = abtt > base
    return [_num_cell(base, not abtt_wins), _num_cell(abtt, abtt_wins)]


def _layers_phrase(layers: Dict[str, int]) -> str:
    parts = [f"{label} {layers[model]}" for model, label in MODELS]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def shuffle_failures(summary: pd.DataFrame, metric_key: str) -> list:
    """Cells where the real attribution does not beat a shuffle of its own scores."""
    col = _shuffle_gap_col(metric_key, "mean")
    out = []
    for model, model_label in MODELS:
        for method, method_label in METHODS:
            for variant in ("baseline", "abtt"):
                if _get(summary, model, method, variant, col) <= 0:
                    out.append(f"{model_label} {method_label} {variant}")
    return out


def _shuffle_clause(layer_set: LayerSet, where: str) -> str:
    """One sentence on the rank column's shuffled-attribution control, read off the summary."""
    failures = shuffle_failures(layer_set.summary, RHO_KEY)
    if not failures:
        return (rf" At the {where} layers every $\rho_{{\text{{LOO}}}}$ cell beats "
                r"a shuffle of its own attribution scores.")
    n = len(failures)
    return (
        rf" At the {where} layers the real attribution does not beat a shuffle "
        rf"of its own scores on $\rho_{{\text{{LOO}}}}$ in "
        rf"{_NUMBER_WORDS.get(n, str(n))} of the twelve cells "
        rf"({', '.join(failures)})."
    )


def caption(reference: LayerSet, aniso: LayerSet) -> str:
    lo_n, hi_n = _abtt_pair_count_range(aniso.summary, DEL_GAP_KEY)
    base_lo, base_hi = _baseline_pair_count_range(aniso.summary, DEL_GAP_KEY)
    return (
        r"\caption{Attribution faithfulness at two layer sets on the same 200 "
        r"positive pairs per model: the operational layers of "
        r"Table~\ref{tab:attribution_metrics_main} (op., "
        rf"{_layers_phrase(reference.layers)}, chosen by the train-only retrieval "
        r"rule) and each model's most anisotropic layer (anis., "
        rf"{_layers_phrase(aniso.layers)}, the largest top-PC variance share on "
        r"the test split, Table~\ref{tab:layer_diagnostics_main}). Methods, "
        r"settings, token filter "
        r"and erasure operator are those of Table~\ref{tab:attribution_metrics_main}; "
        r"ABTT removes $D=10$ components fit on training embeddings at each layer. "
        r"$\rho_{\text{LOO}}$ correlates attribution magnitude with the "
        r"leave-one-out change in the cosine; DelAUC gap is the random-order minus "
        r"attribution-order deletion-curve area, positive when attribution beats "
        r"chance. Boldface marks the better variant. $\Delta$ is the paired ABTT minus "
        r"baseline mean over pairs valid under both variants, with its ratio to "
        r"the paired standard error in parentheses; a dagger marks a tie, within "
        r"two standard errors of zero. ABTT wins, ties and loses "
        rf"{_wtl(reference.rho)} cells on $\rho_{{\text{{LOO}}}}$ and "
        rf"{_wtl(reference.del_gap)} on DelAUC gap at the operational layers, and "
        rf"{_wtl(aniso.rho)} and {_wtl(aniso.del_gap)} at the most anisotropic "
        r"layers."
        + _shuffle_clause(aniso, "most anisotropic")
        + r" At the most anisotropic layers the DelAUC columns average "
        rf"{_count_phrase(lo_n, hi_n)} ABTT pairs against "
        rf"{_count_phrase(base_lo, base_hi)} baseline pairs, since ratio metrics "
        r"are undefined below a full-query cosine of 0.05.}"
    )


def render(reference: LayerSet, aniso: LayerSet, *, source_run: Optional[str],
           reference_run: Optional[str]) -> str:
    lines = [HEADER]
    if source_run:
        lines.append(stamp_line(source_run))
    if reference_run:
        lines.append(f"{REFERENCE_PREFIX}{reference_run}")
    lines += [
        REGEN_NOTE,
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{$\rho_{\text{LOO}}$} "
        r"& \multicolumn{3}{c}{DelAUC gap} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Model & Layer & Method & base & ABTT & $\Delta$ & base & ABTT & $\Delta$ \\",
        r"\midrule",
    ]
    for model, model_label in MODELS:
        first_model_row = True
        for layer_set in (reference, aniso):
            first_layer_row = True
            for method, method_label in METHODS:
                cells = [
                    model_label if first_model_row else "",
                    f"{layer_set.tag} {layer_set.layers[model]}" if first_layer_row else "",
                    method_label,
                ]
                cells += _variant_cells(
                    _get(layer_set.summary, model, method, "baseline", _mean_col(RHO_KEY)),
                    _get(layer_set.summary, model, method, "abtt", _mean_col(RHO_KEY)),
                )
                cells.append(_delta_cell(layer_set.rho.get((model, method))))
                cells += _variant_cells(
                    _get(layer_set.summary, model, method, "baseline", _mean_col(DEL_GAP_KEY)),
                    _get(layer_set.summary, model, method, "abtt", _mean_col(DEL_GAP_KEY)),
                )
                cells.append(_delta_cell(layer_set.del_gap.get((model, method))))
                lines.append(" & ".join(cells) + r" \\")
                first_model_row = False
                first_layer_row = False
        if model != MODELS[-1][0]:
            lines.append(r"\addlinespace[3pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        caption(reference, aniso),
        r"\label{tab:attribution_metrics_aniso}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def load_layer_set(summary_csv: Path, tag: str,
                   pairs_root: Optional[Path] = None) -> LayerSet:
    """Summary rows, paired per-pair statistics and layers for one run.

    The paired statistics need the run's per-pair JSON cache
    (``<metrics dir>/v2_hidden``, gitignored and rebuilt by the metrics sbatch);
    without it the tie markers cannot be computed, so its absence is an error.
    """
    summary_csv = Path(summary_csv)
    pairs_root = pairs_root or summary_csv.parent / "v2_hidden"
    if not pairs_root.exists():
        raise FileNotFoundError(
            f"per-pair metric cache not found at {pairs_root}; rebuild it with "
            "scripts/ig/run_attribution_metrics.py (the run's metrics sbatch)."
        )
    summary = select_main_rows(pd.read_csv(summary_csv), source=str(summary_csv))
    layers = run_layers(summary_csv.parent.parent / "positive200_examples.csv")
    return LayerSet(
        tag=tag,
        summary=summary,
        rho=paired_cell_stats(pairs_root, RHO_KEY),
        del_gap=paired_cell_stats(pairs_root, DEL_GAP_KEY),
        layers=layers,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aniso_summary_csv", type=Path, default=DEFAULT_ANISO_SUMMARY)
    parser.add_argument("--reference_summary_csv", type=Path,
                        default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--table_out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument(
        "--allow_run_change", action="store_true",
        help="Overwrite a table stamped with a different source run.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    source = run_name(args.aniso_summary_csv)
    reference_run = run_name(args.reference_summary_csv)
    refuse_run_change(args.table_out, source, allow=args.allow_run_change)
    reference = load_layer_set(args.reference_summary_csv, "op.")
    aniso = load_layer_set(args.aniso_summary_csv, "anis.")
    if reference.layers == aniso.layers:
        raise SystemExit("the two runs were attributed at the same layers")
    text = render(reference, aniso, source_run=source, reference_run=reference_run)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.table_out}")


if __name__ == "__main__":
    main()
