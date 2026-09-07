"""Build main-text attribution reporting artifacts for the three paper models.

The input is the expanded Run 3 operational attribution summary:

    runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv

Outputs:

    overleaf_drafts/tables/attribution_metrics_main.tex
    overleaf_drafts/tables/attribution_metrics_secondary.tex
    overleaf_drafts/figures/fig_attribution_rho_loo_main.{pdf,png,tex}

Selection (issue #120, from ``docs/research/attribution_metrics_decision.md``
part B). The main table carries two columns and nothing else: ``rho_LOO``,
which ABTT wins 6/6, and ``DelAUC gap``, which it wins 3/6. Both are
threshold-free and both have a calibrated zero. They are printed as paired
base/ABTT columns rather than ``base -> ABTT`` arrow cells, which are not a
table convention readers expect.

Everything the old four-metric table used to carry in the main text moves to
the secondary appendix table: ``tau_LOO`` (the tie-corrected twin of
``rho_LOO``, with which it correlates 0.9995, so it is not a second witness),
``InsAUC gap`` (which fails the shuffled-attribution control in two baseline
cells) and the three ERASER-style headline cells.

Both tables read one summary, so both describe the same erasure operator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUMMARY = (
    REPO_ROOT
    / "runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv"
)
DEFAULT_TABLE_OUT = REPO_ROOT / "overleaf_drafts/tables/attribution_metrics_main.tex"
DEFAULT_SECONDARY_OUT = (
    REPO_ROOT / "overleaf_drafts/tables/attribution_metrics_secondary.tex"
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

MAIN_METRIC_KEYS = (RHO_KEY, DEL_GAP_KEY, DEL_RANDOM_KEY)
SECONDARY_METRIC_KEYS = (TAU_KEY, INS_GAP_KEY, SUFF_KEY, COMP_KEY, MINFRAC_KEY)
METRIC_KEYS = MAIN_METRIC_KEYS + SECONDARY_METRIC_KEYS

# Overleaf receives these files, so the header says nothing about the repo.
HEADER = "% generated table"
REGEN_NOTE = (
    "% Selection and wording follow the part B memo behind issue #120. If the "
    "attribution\n% re-sample of issue #141 lands, regenerate this file from "
    "the new summary rather than\n% editing the numbers here."
)


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


def _baseline_pair_count(summary: pd.DataFrame, metric_key: str) -> int:
    counts = {
        int(_get(summary, model, method, "baseline", f"{metric_key}_n"))
        for model, _ in MODELS
        for method, _ in METHODS
    }
    if len(counts) != 1:
        raise ValueError(f"baseline pair counts for {metric_key} disagree: {sorted(counts)}")
    return counts.pop()


def _random_floor_range(summary: pd.DataFrame) -> tuple[float, float]:
    floors = [
        _get(summary, model, method, variant, _mean_col(DEL_RANDOM_KEY))
        for model, _ in MODELS
        for method, _ in METHODS
        for variant in ("baseline", "abtt")
    ]
    return min(floors), max(floors)


def main_caption(summary: pd.DataFrame) -> str:
    rho_wins = _wins(summary, RHO_KEY)
    del_wins = _wins(summary, DEL_GAP_KEY)
    lo_n, hi_n = _abtt_pair_count_range(summary, DEL_GAP_KEY)
    base_n = _baseline_pair_count(summary, DEL_GAP_KEY)
    lo_floor, hi_floor = _random_floor_range(summary)

    # The "about 1.2 standard errors" figure is the paired ABTT-minus-baseline
    # difference for LaTa/MaRC from part A7 of the selection memo, which needs
    # per-pair values the summary does not carry. It is only true while that
    # cell is the narrow DelAUC win, so refuse to print it otherwise: on a new
    # summary, recompute the paired standard errors before restoring the claim.
    lata_marc_base = _get(
        summary, "bowphs/LaTa", "retrieval_mark", "baseline", _mean_col(DEL_GAP_KEY)
    )
    lata_marc_abtt = _get(
        summary, "bowphs/LaTa", "retrieval_mark", "abtt", _mean_col(DEL_GAP_KEY)
    )
    margin = lata_marc_abtt - lata_marc_base
    if not 0.0 < margin < 0.10:
        raise ValueError(
            "the caption's 1.2 standard error claim is about the LaTa/MaRC "
            f"DelAUC gap being a narrow ABTT win; this summary gives {margin:+.3f}. "
            "Recompute the paired standard errors before regenerating."
        )

    return (
        r"\caption{Attribution faithfulness at the predeclared operational "
        r"layers, 200 positive pairs per model, for integrated gradients (IG) "
        r"and retrieval-adapted MaRC. $\rho_{\text{LOO}}$ correlates "
        r"attribution magnitude with the leave-one-out change in the cosine. "
        r"DelAUC gap is the deletion-curve area in attribution order minus the "
        rf"area under a random order, whose reference runs from {lo_floor:.3f} "
        rf"to {hi_floor:.3f} here, so zero is chance. Higher is better in both; "
        rf"boldface marks the better variant. ABTT wins {rho_wins}/6 and "
        rf"{del_wins}/6, and the LaTa MaRC DelAUC win is a tie at about 1.2 "
        r"standard errors. Ratio metrics are undefined below a full-query "
        rf"cosine of 0.05, so the DelAUC columns average {lo_n} to {hi_n} ABTT "
        rf"pairs against {base_n} baseline pairs. Cross-variant comparisons are "
        r"descriptive. Secondary metrics: "
        r"Table~\ref{tab:attribution_metrics_secondary}.}"
    )


def render_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = [
        HEADER,
        REGEN_NOTE,
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
        main_caption(summary),
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


def secondary_caption(summary: pd.DataFrame) -> str:
    tau_wins = _wins(summary, TAU_KEY)
    ins_wins = _wins(summary, INS_GAP_KEY)
    lo_n, hi_n = _abtt_pair_count_range(summary, INS_GAP_KEY)
    base_n = _baseline_pair_count(summary, INS_GAP_KEY)
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
        r"report it in the main table because in two of the twelve cells, both "
        r"of them baseline cells, the real attribution does not beat a "
        r"permutation of its own scores, so the measurement does not meet the "
        r"validity bar we set for a headline column. The threshold-based "
        r"ERASER metrics are reported for completeness: their "
        r"baseline-versus-ABTT verdict depends on the threshold and on the "
        r"erasure operator, which is why the main table uses threshold-free, "
        r"chance-corrected metrics instead. The InsAUC columns average "
        rf"{lo_n} to {hi_n} ABTT pairs against the baseline's {base_n}, for the "
        r"same small-denominator reason. Full threshold sweeps are in "
        r"Tables~\ref{tab:attribution_sweep_main_methods} "
        r"and~\ref{tab:attribution_sweep_supplemental_methods}.}"
    )


def render_secondary_table(summary: pd.DataFrame, out_path: Path) -> None:
    n_metrics = len(SECONDARY_COLUMNS)
    banner = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in SECONDARY_COLUMNS
    )
    rules = "".join(
        rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(n_metrics)
    )
    lines: list[str] = [
        HEADER,
        REGEN_NOTE,
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


def render_rho_figure(summary: pd.DataFrame, out_base: Path) -> None:
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

    from matplotlib.lines import Line2D

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
            (
                r"\caption{\textbf{$\rho_{\text{LOO}}$ foreground view for the main "
                r"candidate-attribution result.} Each line connects the baseline "
                r"and ABTT variants for one model-method cell at the predeclared "
                r"operational attribution layer. ABTT improves the leave-one-out "
                r"rank-correlation signal in all six cells; "
                r"Table~\ref{tab:attribution_metrics_secondary} shows the "
                r"secondary metrics which qualify this narrower faithfulness "
                r"claim.}"
            ),
            r"\label{fig:attribution_rho_loo_main}",
            r"\end{figure}",
            "",
        ]
    )
    out_base.with_suffix(".tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table_out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--secondary_table_out", type=Path, default=DEFAULT_SECONDARY_OUT)
    parser.add_argument("--fig_out_base", type=Path, default=DEFAULT_FIG_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _load_main_rows(args.summary_csv)
    render_table(summary, args.table_out)
    render_secondary_table(summary, args.secondary_table_out)
    render_rho_figure(summary, args.fig_out_base)
    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.secondary_table_out}")
    print(f"Wrote {args.fig_out_base.with_suffix('.pdf')}")
    print(f"Wrote {args.fig_out_base.with_suffix('.png')}")
    print(f"Wrote {args.fig_out_base.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
