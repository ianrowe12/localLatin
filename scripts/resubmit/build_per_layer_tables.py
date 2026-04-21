"""Build per-layer Task A and Task B tables for main paper + appendix.

Post-2026-04-20 layout:
  Main paper (3 models: LaTa, PhilTa, mT5-base) compares only
  `baseline` vs `abtt_optimal` (pure ABTT, not SIF+ABTT).
  Appendix carries the remaining 3 models (LaBSE, Qwen3-0.6B, KaLM-mini)
  under the same 2-method view, plus SIF-variant sweeps across all 6 models.

Inputs:
  - Single-seed per-layer CSV (has all 7 methods incl. `abtt_optimal`).
  - Multi-seed (mseed) CSV (has baseline + three SIF variants only; no
    pure `abtt_optimal`). Used only for the appendix mseed ranking table.

Outputs (tex + audit CSVs):
  tables/taskA_main.tex                      — 3 models, base vs ABTT
  tables/taskA_appendix.tex                  — 3 remaining models, base vs ABTT
  tables/taskA_appendix_sif.tex              — all 6 models, SIF suite (AUROC)
  tables/taskB_routing_main.tex              — 3 models, base vs ABTT
  tables/taskB_routing_appendix.tex          — 3 remaining models, base vs ABTT
  tables/taskB_routing_appendix_sif.tex      — all 6 models, SIF suite (overall)
  tables/taskB_ranking_main.tex              — 3 models, base vs ABTT (single-seed)
  tables/taskB_ranking_appendix.tex          — 3 remaining models (single-seed)
  tables/taskB_ranking_appendix_mseed.tex    — all 6 models, base vs SIF+ABTT (mseed)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd


MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}

MAIN_MODELS = [
    "bowphs/LaTa",
    "bowphs/PhilTa",
    "google/mt5-base",
]
APPENDIX_MODELS = [
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
]
ALL_MODELS = MAIN_MODELS + APPENDIX_MODELS

METHOD_DISPLAY = {
    "baseline": "base",
    "abtt_optimal": "ABTT",
    "abtt_fixed": "ABTT$_{10}$",
    "sif_only": "SIF",
    "sif_abtt_fixed": "SIF+ABTT$_{10}$",
    "sif_abtt_optimal": "SIF+ABTT",
    "whitening": "Whiten",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--taskA_csv",
        default="runs/active/resubmit/results/phase_resubmit_results.csv",
    )
    p.add_argument(
        "--taskB_csv",
        default="runs/active/resubmit/taskb_mseed/aggregated_results.csv",
    )
    p.add_argument(
        "--out_dir_audit",
        default="runs/active/resubmit/results/perlayer_tables",
        help="Where per-layer audit CSVs are written.",
    )
    p.add_argument(
        "--out_tex_dir",
        default="overleaf_drafts/tables",
    )
    return p.parse_args()


# ------------------------------- pivoting helpers ----------------------------


def _filter(
    df: pd.DataFrame,
    models: Sequence[str],
    methods: Sequence[str],
) -> pd.DataFrame:
    unknown = set(df["model"].unique()) - set(MODEL_DISPLAY)
    if unknown:
        raise SystemExit(f"Unknown models in CSV: {unknown}")
    keep = df[df["model"].isin(models) & df["method"].isin(methods)].copy()
    present_methods = set(keep["method"].unique())
    missing_methods = set(methods) - present_methods
    if missing_methods:
        raise SystemExit(
            f"Methods requested but missing for some rows: {missing_methods}. "
            f"Present in filtered frame: {sorted(present_methods)}"
        )
    return keep


def _pivot(
    df: pd.DataFrame,
    models: Sequence[str],
    methods: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    df = _filter(df, models, methods)
    wide = df.pivot_table(
        index=["model", "layer"],
        columns="method",
        values=list(value_cols),
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["model_display"] = wide["model"].map(MODEL_DISPLAY)
    order = {m: i for i, m in enumerate(models)}
    wide = wide.sort_values(
        ["model", "layer"],
        key=lambda s: s.map(order) if s.name == "model" else s,
    ).reset_index(drop=True)
    return wide


# --------------------------------- formatting --------------------------------


def _fmt3(x: float) -> str:
    return "--" if pd.isna(x) else f"{x:.3f}"


def _fmt_pm(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _bold(s: str, is_best: bool) -> str:
    return (r"\textbf{" + s + r"}") if is_best else s


# ----------------------------- tex emitter core ------------------------------


@dataclass
class TableSpec:
    banner: str
    caption: str
    label: str
    col_spec: str
    metric_headers: list[str]
    best_rows: set
    row_cells_fn: Callable[[int, pd.Series], list[str]]


def _write_longtable(wide: pd.DataFrame, spec: TableSpec, out_path: Path) -> None:
    ncols = len(spec.col_spec)
    metric_span = ncols - 2
    banner_row = (
        r"\multicolumn{2}{c}{}"
        + f" & \\multicolumn{{{metric_span}}}{{c}}{{\\textbf{{{spec.banner}}}}}"
        + r" \\"
    )
    header_row = r"\textbf{Model} & \textbf{Layer}"
    for h in spec.metric_headers:
        header_row += " & " + h
    header_row += r" \\"

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{longtable}{" + spec.col_spec + r"}")
    lines.append(r"\caption{" + spec.caption + r"}")
    lines.append(r"\label{" + spec.label + r"} \\")
    lines.append(r"\toprule")
    lines.append(banner_row)
    lines.append(r"\cmidrule(lr){3-" + str(ncols) + "}")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(banner_row)
    lines.append(r"\cmidrule(lr){3-" + str(ncols) + "}")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(ncols) + r"}{r}{\textit{(continued on next page)}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    prev_model = None
    for i, row in wide.iterrows():
        if prev_model is not None and row["model"] != prev_model:
            lines.append(r"\midrule")
        cells = spec.row_cells_fn(i, row)
        if prev_model == row["model"]:
            cells[0] = ""
        lines.append(" & ".join(cells) + r" \\")
        prev_model = row["model"]

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ------------------------------ table emitters -------------------------------


def _best_rows(wide: pd.DataFrame, key_col: str) -> set:
    if key_col not in wide.columns:
        return set()
    idx = wide.groupby("model")[key_col].idxmax()
    return set(idx.values)


def emit_taskA(
    df: pd.DataFrame,
    models: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    out_tex: Path,
    out_audit: Path,
    caption: str,
    label: str,
    banner: str,
    best_method: str,
    best_metric: str = "aucroc",
) -> pd.DataFrame:
    """Emit Task A per-layer table.

    metrics: subset of {"aucroc", "gap"}.
    """
    wide = _pivot(df, models, methods, metrics)
    wide.to_csv(out_audit, index=False)

    best_key = f"{best_metric}__{best_method}"
    best_rows = _best_rows(wide, best_key)

    col_spec = "ll" + "c" * (len(metrics) * len(methods))
    metric_name_display = {"aucroc": "AUROC", "gap": "Cosine gap"}
    metric_headers = []
    for metric in metrics:
        for method in methods:
            metric_headers.append(
                rf"\makecell{{{metric_name_display[metric]}\\({METHOD_DISPLAY[method]})}}"
            )

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows
        cells = [row["model_display"], _bold(str(int(row["layer"])), is_best)]
        for metric in metrics:
            for method in methods:
                cells.append(_bold(_fmt3(row[f"{metric}__{method}"]), is_best))
        return cells

    spec = TableSpec(
        banner=banner,
        caption=caption,
        label=label,
        col_spec=col_spec,
        metric_headers=metric_headers,
        best_rows=best_rows,
        row_cells_fn=row_cells,
    )
    _write_longtable(wide, spec, out_tex)
    return wide


def emit_taskB_routing(
    df: pd.DataFrame,
    models: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    out_tex: Path,
    out_audit: Path,
    caption: str,
    label: str,
    banner: str,
    best_method: str,
    best_metric: str = "overall_assignment_acc",
) -> pd.DataFrame:
    """Emit Task B routing per-layer table (single-seed).

    metrics: subset of {"existing_acc", "new_acc", "overall_assignment_acc"}.
    """
    wide = _pivot(df, models, methods, metrics)
    wide.to_csv(out_audit, index=False)

    best_key = f"{best_metric}__{best_method}"
    best_rows = _best_rows(wide, best_key)

    col_spec = "ll" + "c" * (len(metrics) * len(methods))
    metric_name_display = {
        "existing_acc": "Existing",
        "new_acc": "New",
        "overall_assignment_acc": "Overall",
    }
    metric_headers = []
    for metric in metrics:
        for method in methods:
            metric_headers.append(
                rf"\makecell{{{metric_name_display[metric]}\\({METHOD_DISPLAY[method]})}}"
            )

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows
        cells = [row["model_display"], _bold(str(int(row["layer"])), is_best)]
        for metric in metrics:
            for method in methods:
                cells.append(_bold(_fmt3(row[f"{metric}__{method}"]), is_best))
        return cells

    spec = TableSpec(
        banner=banner,
        caption=caption,
        label=label,
        col_spec=col_spec,
        metric_headers=metric_headers,
        best_rows=best_rows,
        row_cells_fn=row_cells,
    )
    _write_longtable(wide, spec, out_tex)
    return wide


def emit_taskB_ranking_single(
    df: pd.DataFrame,
    models: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    out_tex: Path,
    out_audit: Path,
    caption: str,
    label: str,
    banner: str,
    best_method: str,
    best_metric: str = "dir_acc_at_1",
) -> pd.DataFrame:
    """Emit Task B ranking table from the single-seed Task A CSV (no ±std)."""
    wide = _pivot(df, models, methods, metrics)
    wide.to_csv(out_audit, index=False)

    best_key = f"{best_metric}__{best_method}"
    best_rows = _best_rows(wide, best_key)

    col_spec = "ll" + "c" * (len(metrics) * len(methods))
    metric_name_display = {
        "dir_acc_at_1": "Acc@1",
        "existing_acc": "Existing",
        "new_acc": "New",
    }
    metric_headers = []
    for metric in metrics:
        for method in methods:
            metric_headers.append(
                rf"\makecell{{{metric_name_display[metric]}\\({METHOD_DISPLAY[method]})}}"
            )

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows
        cells = [row["model_display"], _bold(str(int(row["layer"])), is_best)]
        for metric in metrics:
            for method in methods:
                cells.append(_bold(_fmt3(row[f"{metric}__{method}"]), is_best))
        return cells

    spec = TableSpec(
        banner=banner,
        caption=caption,
        label=label,
        col_spec=col_spec,
        metric_headers=metric_headers,
        best_rows=best_rows,
        row_cells_fn=row_cells,
    )
    _write_longtable(wide, spec, out_tex)
    return wide


def emit_taskB_ranking_mseed(
    df: pd.DataFrame,
    models: Sequence[str],
    methods: Sequence[str],
    out_tex: Path,
    out_audit: Path,
    caption: str,
    label: str,
    banner: str,
    best_method: str,
) -> pd.DataFrame:
    """Emit mseed Task B ranking table with mean ± std.

    Reports Acc@1, existing, new.
    """
    value_cols = [
        "dir_acc_at_1_mean",
        "dir_acc_at_1_std",
        "existing_acc_mean",
        "existing_acc_std",
        "new_acc_mean",
        "new_acc_std",
    ]
    wide = _pivot(df, models, methods, value_cols)
    wide.to_csv(out_audit, index=False)

    best_key = f"dir_acc_at_1_mean__{best_method}"
    best_rows = _best_rows(wide, best_key)

    metric_groups = [
        ("Acc@1", "dir_acc_at_1"),
        ("Existing", "existing_acc"),
        ("New", "new_acc"),
    ]
    col_spec = "ll" + "c" * (len(metric_groups) * len(methods))
    metric_headers = [
        rf"\makecell{{{label}\\({METHOD_DISPLAY[m]})}}"
        for (label, _) in metric_groups
        for m in methods
    ]

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows
        cells = [row["model_display"], _bold(str(int(row["layer"])), is_best)]
        for _, stem in metric_groups:
            for m in methods:
                mean = row[f"{stem}_mean__{m}"]
                std = row[f"{stem}_std__{m}"]
                cells.append(_bold(_fmt_pm(mean, std), is_best))
        return cells

    spec = TableSpec(
        banner=banner,
        caption=caption,
        label=label,
        col_spec=col_spec,
        metric_headers=metric_headers,
        best_rows=best_rows,
        row_cells_fn=row_cells,
    )
    _write_longtable(wide, spec, out_tex)
    return wide


# --------------------------------- captions ----------------------------------


CAP_BASE_ABTT = (
    r"\texttt{baseline} is mean pooling without correction; "
    r"\texttt{abtt\_optimal} applies ABTT (no SIF weighting) with $D$ tuned per layer on the train split. "
)
CAP_TASKA_METHOD = CAP_BASE_ABTT + (
    r"Both AUROC and cosine gap are computed over the test n$\times$n cosine matrix with no threshold and "
    r"no directory routing: AUROC treats same-directory test pairs as positives, and cosine gap is the "
    r"difference between mean same-directory and mean different-directory cosine. "
)
CAP_ROUTING_METHOD = CAP_BASE_ABTT + (
    r"For every test file we compute $s_i = \max_{j \neq i} \cos(e_i, e_j)$ against the rest of the test "
    r"set and compare to the learned threshold $\tau$ (fit on train via $F_1$-optimal cut on "
    r"same-directory vs.\ different-directory pairs). The decision is binary first (existing if "
    r"$s_i \geq \tau$, else new); only files predicted existing are then routed to their top-1 "
    r"neighbour's directory. We report \textbf{existing accuracy} (restricted to files whose true "
    r"partner is in the test set), \textbf{new accuracy} (files that should be flagged as novel), and "
    r"\textbf{overall assignment accuracy}. "
)
CAP_RANK_SINGLE_METHOD = CAP_BASE_ABTT + (
    r"This table is computed on the single-seed v2 split (no $M$-seed averaging, since the mseed "
    r"sweep was run only for SIF-conditioned variants; see "
    r"Table~\ref{tab:taskB_ranking_appendix_mseed} for the multi-seed SIF+ABTT view). "
    r"\textbf{Acc@1} is directory accuracy at rank~1, with an existing/new decomposition. "
)


def _best_layer_caption(method_label: str, metric_label: str) -> str:
    return (
        rf"Rows in bold mark the best layer per model, selected by {method_label} {metric_label}."
    )


# ----------------------------------- main ------------------------------------


def main() -> None:
    args = parse_args()
    taskA_df = pd.read_csv(args.taskA_csv)
    taskB_df = pd.read_csv(args.taskB_csv)

    audit_dir = Path(args.out_dir_audit)
    audit_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = Path(args.out_tex_dir)
    tex_dir.mkdir(parents=True, exist_ok=True)

    # -------- Task A: pairwise --------
    emit_taskA(
        taskA_df,
        models=MAIN_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["aucroc", "gap"],
        out_tex=tex_dir / "taskA_main.tex",
        out_audit=audit_dir / "taskA_main.csv",
        caption=(
            r"Per-layer Task~A pairwise duplicate-detection metrics for the three headline models "
            r"(LaTa, PhilTa, mT5-base). " + CAP_TASKA_METHOD
            + _best_layer_caption("ABTT", "AUROC")
        ),
        label="tab:taskA_main",
        banner=r"Task A: Pairwise Duplicate Detection (main)",
        best_method="abtt_optimal",
        best_metric="aucroc",
    )
    emit_taskA(
        taskA_df,
        models=APPENDIX_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["aucroc", "gap"],
        out_tex=tex_dir / "taskA_appendix.tex",
        out_audit=audit_dir / "taskA_appendix.csv",
        caption=(
            r"Per-layer Task~A pairwise metrics for the remaining three models "
            r"(LaBSE, Qwen3-0.6B, KaLM-mini), under the same two-method comparison as the main paper. "
            + CAP_TASKA_METHOD + _best_layer_caption("ABTT", "AUROC")
        ),
        label="tab:taskA_appendix",
        banner=r"Task A: Pairwise Duplicate Detection (appendix models)",
        best_method="abtt_optimal",
        best_metric="aucroc",
    )
    emit_taskA(
        taskA_df,
        models=ALL_MODELS,
        methods=["baseline", "sif_only", "sif_abtt_fixed", "sif_abtt_optimal"],
        metrics=["aucroc"],
        out_tex=tex_dir / "taskA_appendix_sif.tex",
        out_audit=audit_dir / "taskA_appendix_sif.csv",
        caption=(
            r"Per-layer Task~A AUROC across the SIF-conditioned post-processing suite for all six "
            r"models. \texttt{sif\_only} replaces mean pooling with SIF-weighted pooling; "
            r"\texttt{sif\_abtt\_fixed} adds ABTT with fixed $D{=}10$; \texttt{sif\_abtt\_optimal} "
            r"tunes $D$ per layer on the train split. Gap is omitted to keep the table narrow; the "
            r"pure-ABTT comparison (not SIF-conditioned) is in "
            r"Tables~\ref{tab:taskA_main} and~\ref{tab:taskA_appendix}. "
            + _best_layer_caption("SIF+ABTT (D-tuned)", "AUROC")
        ),
        label="tab:taskA_appendix_sif",
        banner=r"Task A: SIF-suite AUROC (appendix)",
        best_method="sif_abtt_optimal",
        best_metric="aucroc",
    )

    # -------- Task B: routing (single-seed, from Task A CSV) --------
    emit_taskB_routing(
        taskA_df,
        models=MAIN_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["existing_acc", "new_acc", "overall_assignment_acc"],
        out_tex=tex_dir / "taskB_routing_main.tex",
        out_audit=audit_dir / "taskB_routing_main.csv",
        caption=(
            r"Per-layer Task~B autonomous routing accuracy for the three headline models. "
            + CAP_ROUTING_METHOD
            + _best_layer_caption("ABTT", "overall assignment accuracy")
        ),
        label="tab:taskB_routing_main",
        banner=r"Task B: Autonomous Routing (main, file-level, $\tau$-thresholded)",
        best_method="abtt_optimal",
        best_metric="overall_assignment_acc",
    )
    emit_taskB_routing(
        taskA_df,
        models=APPENDIX_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["existing_acc", "new_acc", "overall_assignment_acc"],
        out_tex=tex_dir / "taskB_routing_appendix.tex",
        out_audit=audit_dir / "taskB_routing_appendix.csv",
        caption=(
            r"Per-layer Task~B autonomous routing accuracy for the remaining three models. "
            + CAP_ROUTING_METHOD
            + _best_layer_caption("ABTT", "overall assignment accuracy")
        ),
        label="tab:taskB_routing_appendix",
        banner=r"Task B: Autonomous Routing (appendix models)",
        best_method="abtt_optimal",
        best_metric="overall_assignment_acc",
    )
    emit_taskB_routing(
        taskA_df,
        models=ALL_MODELS,
        methods=["baseline", "sif_only", "sif_abtt_fixed", "sif_abtt_optimal"],
        metrics=["overall_assignment_acc"],
        out_tex=tex_dir / "taskB_routing_appendix_sif.tex",
        out_audit=audit_dir / "taskB_routing_appendix_sif.csv",
        caption=(
            r"Per-layer Task~B overall assignment accuracy across the SIF-conditioned suite for all "
            r"six models. Restricted to overall routing accuracy (existing/new decomposition omitted) "
            r"to keep the table compact; see Tables~\ref{tab:taskB_routing_main} and~"
            r"\ref{tab:taskB_routing_appendix} for the pure-ABTT pairwise comparison with existing/new "
            r"broken out. "
            + _best_layer_caption("SIF+ABTT (D-tuned)", "overall assignment accuracy")
        ),
        label="tab:taskB_routing_appendix_sif",
        banner=r"Task B: SIF-suite routing (appendix)",
        best_method="sif_abtt_optimal",
        best_metric="overall_assignment_acc",
    )

    # -------- Task B: ranking (single-seed, from Task A CSV) --------
    emit_taskB_ranking_single(
        taskA_df,
        models=MAIN_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["dir_acc_at_1", "existing_acc", "new_acc"],
        out_tex=tex_dir / "taskB_ranking_main.tex",
        out_audit=audit_dir / "taskB_ranking_main.csv",
        caption=(
            r"Per-layer Task~B top-$k$ ranking metrics for the three headline models. "
            + CAP_RANK_SINGLE_METHOD
            + _best_layer_caption("ABTT", "Acc@1")
        ),
        label="tab:taskB_ranking_main",
        banner=r"Task B: Top-K Ranking (main, single-seed v2)",
        best_method="abtt_optimal",
        best_metric="dir_acc_at_1",
    )
    emit_taskB_ranking_single(
        taskA_df,
        models=APPENDIX_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["dir_acc_at_1", "existing_acc", "new_acc"],
        out_tex=tex_dir / "taskB_ranking_appendix.tex",
        out_audit=audit_dir / "taskB_ranking_appendix.csv",
        caption=(
            r"Per-layer Task~B top-$k$ ranking metrics for the remaining three models. "
            + CAP_RANK_SINGLE_METHOD
            + _best_layer_caption("ABTT", "Acc@1")
        ),
        label="tab:taskB_ranking_appendix",
        banner=r"Task B: Top-K Ranking (appendix models, single-seed)",
        best_method="abtt_optimal",
        best_metric="dir_acc_at_1",
    )

    # -------- Task B: mseed SIF+ABTT (appendix, preserves pre-restructure content) --------
    emit_taskB_ranking_mseed(
        taskB_df,
        models=ALL_MODELS,
        methods=["baseline", "sif_abtt_optimal"],
        out_tex=tex_dir / "taskB_ranking_appendix_mseed.tex",
        out_audit=audit_dir / "taskB_ranking_appendix_mseed.csv",
        caption=(
            r"Per-layer Task~B top-$k$ ranking accuracy across all six models, averaged over $M=5$ "
            r"query/reference reseedings of the v2 train/test split (mean $\pm$ std). "
            r"\texttt{sif\_abtt\_optimal} applies SIF weighting plus ABTT with $D$ tuned per layer on "
            r"the train split; the mseed sweep was not run for the pure \texttt{abtt\_optimal} variant "
            r"reported in the main paper, so we include this table to give a variance-aware view of the "
            r"SIF-conditioned best variant. Overall assignment accuracy coincides with Acc@1 in the "
            r"mseed pipeline, so we report the existing/new decomposition instead. "
            + _best_layer_caption("SIF+ABTT", "Acc@1")
        ),
        label="tab:taskB_ranking_appendix_mseed",
        banner=r"Task B: Top-K Ranking (appendix, 5-seed mean $\pm$ std, SIF+ABTT)",
        best_method="sif_abtt_optimal",
    )

    print("wrote audit CSVs to", audit_dir)
    print("wrote tex tables to", tex_dir)


if __name__ == "__main__":
    main()
