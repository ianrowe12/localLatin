"""Build per-layer Task A and Task B tables for main paper + appendix.

Post-2026-04-20 layout:
  Main paper (3 models: LaTa, PhilTa, mT5-base) compares only
  `baseline` vs `abtt_optimal` (pure ABTT, not SIF+ABTT).
  Appendix carries the 3 non-T5 models (LaBSE, Qwen3-0.6B, KaLM-mini)
  under the same 2-method view, plus SIF-variant sweeps across all 6 models.

Inputs:
  - Single-seed per-layer CSV (has all 7 methods incl. `abtt_optimal`).
  - Multi-seed (mseed) CSV (has baseline + three SIF variants only; no
    pure `abtt_optimal`). Used only for the appendix mseed ranking table,
    whose bold rows are the train-selected `sif_abtt_optimal` layers taken
    from the single-seed CSV (see `taskb_mseed_selection.py`, issue #175),
    so they match `tables/taskB_topk.tex` cell for cell.

Bold rows (issue #184): every table bolds the layer the headline tables
report, chosen on the train split through `train_selected_layers`: the
argmax of `train_aucroc` for the Task A tables (the layer behind the cell in
`tables/taskA_headline.tex`) and of `train_dir_acc_at_1` for the Task B
tables (the layer behind the cell in `tables/taskB_headline.tex`), under the
table's ABTT method; both are listed in `tables/selected_layers.tex` (issue
#219). No table takes an argmax over a test column.

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

from taskb_mseed_selection import (
    MSEED_METHOD,
    SELECTION_RULE_CAPTION,
    train_selected_layers,
)


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

# Train-split columns the headline tables select on, per task (issue #184).
TASKA_SELECT_METRIC = "train_aucroc"
TASKB_SELECT_METRIC = "train_dir_acc_at_1"
# Every headline cell (Base, SIF, ABTT, SIF+ABTT; both tasks) is printed in tab:headline_ci.
HEADLINE_LABEL = {"taskA": "tab:headline_ci", "taskB": "tab:headline_ci"}

METHOD_DISPLAY = {
    "baseline": "Base",
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
    # Caption below the table, as formatting.md asks; longtable steps the table counter at
    # its start, so the label resolves from the last foot.
    lines.append(r"\noalign{\vskip 8pt}")
    lines.append(r"\caption{" + spec.caption + r"}")
    lines.append(r"\label{" + spec.label + r"} \\")
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


def _write_table_star(wide: pd.DataFrame, spec: TableSpec, out_path: Path) -> None:
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
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{" + spec.col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(banner_row)
    lines.append(r"\cmidrule(lr){3-" + str(ncols) + "}")
    lines.append(header_row)
    lines.append(r"\midrule")

    prev_model = None
    for i, row in wide.iterrows():
        if prev_model is not None and row["model"] != prev_model:
            lines.append(r"\midrule")
        cells = spec.row_cells_fn(i, row)
        if prev_model == row["model"]:
            cells[0] = ""
        lines.append(" & ".join(cells) + r" \\")
        prev_model = row["model"]

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + spec.caption + r"}")
    lines.append(r"\label{" + spec.label + r"}")
    lines.append(r"\end{table*}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ------------------------------ table emitters -------------------------------


def _mark_train_selected(wide: pd.DataFrame, selected_layers: dict[str, int]) -> set:
    """Flag the row per model that carries its train-selected layer.

    Adds a ``train_selected`` column to ``wide`` (so the audit CSV records
    the choice) and returns the index set to bold. Refuses to bold a subset:
    a selected layer missing from the pivot means the two CSVs disagree.
    """
    wide["train_selected"] = [
        int(row["layer"]) == selected_layers[row["model"]] for _, row in wide.iterrows()
    ]
    rows = set(wide.index[wide["train_selected"]])
    if len(rows) != len(selected_layers):
        missing = {
            m: l for m, l in selected_layers.items()
            if not ((wide["model"] == m) & (wide["layer"] == l)).any()
        }
        raise SystemExit(f"train-selected layers absent from the per-layer pivot: {missing}")
    return rows


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
    select_method: str,
    select_metric: str = TASKA_SELECT_METRIC,
    float_table: bool = False,
) -> pd.DataFrame:
    """Emit Task A per-layer table.

    metrics: subset of {"aucroc", "gap"}. The bold row per model is the
    train-split argmax of ``select_metric`` under ``select_method``.
    """
    wide = _pivot(df, models, methods, metrics)
    selected = train_selected_layers(df, list(models), method=select_method, metric=select_metric)
    best_rows = _mark_train_selected(wide, selected)
    wide.to_csv(out_audit, index=False)

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
                bold = is_best and method == select_method
                cells.append(_bold(_fmt3(row[f"{metric}__{method}"]), bold))
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
    writer = _write_table_star if float_table else _write_longtable
    writer(wide, spec, out_tex)
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
    select_method: str,
    select_metric: str = TASKB_SELECT_METRIC,
    float_table: bool = False,
) -> pd.DataFrame:
    """Emit Task B routing per-layer table (single-seed).

    metrics: subset of {"existing_acc", "new_acc", "overall_assignment_acc"}.
    The bold row per model is the train-split argmax of ``select_metric``
    under ``select_method``.
    """
    wide = _pivot(df, models, methods, metrics)
    selected = train_selected_layers(df, list(models), method=select_method, metric=select_metric)
    best_rows = _mark_train_selected(wide, selected)
    wide.to_csv(out_audit, index=False)

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
                bold = is_best and method == select_method
                cells.append(_bold(_fmt3(row[f"{metric}__{method}"]), bold))
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
    writer = _write_table_star if float_table else _write_longtable
    writer(wide, spec, out_tex)
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
    select_method: str,
    select_metric: str = TASKB_SELECT_METRIC,
    float_table: bool = False,
) -> pd.DataFrame:
    """Emit Task B ranking table from the single-seed Task A CSV (no ±std).

    The bold row per model is the train-split argmax of ``select_metric``
    under ``select_method``.
    """
    wide = _pivot(df, models, methods, metrics)
    selected = train_selected_layers(df, list(models), method=select_method, metric=select_metric)
    best_rows = _mark_train_selected(wide, selected)
    wide.to_csv(out_audit, index=False)

    col_spec = "ll" + "c" * (len(metrics) * len(methods))
    metric_name_display = {
        "dir_acc_at_1": "DirAcc@1",
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
                bold = is_best and method == select_method
                cells.append(_bold(_fmt3(row[f"{metric}__{method}"]), bold))
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
    writer = _write_table_star if float_table else _write_longtable
    writer(wide, spec, out_tex)
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
    selected_layers: dict[str, int],
    bold_method: str = MSEED_METHOD,
) -> pd.DataFrame:
    """Emit mseed Task B ranking table with mean ± std.

    Reports Acc@1, existing, new. ``selected_layers`` (model -> layer) marks
    the bold row per model; it is the train-selected layer shared with the
    top-K table, not an argmax over the five-seed means in this table.
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
    best_rows = _mark_train_selected(wide, selected_layers)
    wide.to_csv(out_audit, index=False)

    metric_groups = [
        ("DirAcc@1", "dir_acc_at_1"),
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
                cells.append(_bold(_fmt_pm(mean, std), is_best and m == bold_method))
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
# Caption pass (12 October 2026): the first table of each kind (taskA_main,
# taskB_routing_main, taskB_ranking_main) carries the compact key; its siblings
# point to it and say only what differs (model set, SIF variants, reseeding).


# B3 (reader pass, 10 October 2026): these tables print accuracies as fractions, the main
# text as percent; the captions say so rather than rescaling every cell.
CAP_FRACTION = r"as fractions"
CAP_BASE_ABTT = (
    r"Base: mean pooling; ABTT: fit on the training split, $D$ tuned per layer. "
)
CAP_MAIN_MODELS = r"LaTa, PhilTa and mT5-base"
CAP_APPENDIX_MODELS = r"the embedding-trained models LaBSE, Qwen3-0.6B and KaLM-mini"
# Caption-only reading caveat: the ABTT routing tables are single runs, while the
# five reseedings were run for the SIF variants only (Table tab:taskB_ranking_appendix_mseed).
CAP_SINGLE_RUN = r"computed once, without reseedings"

CAP_TASKA_MAIN = (
    rf"Per-layer test ranking AUROC and cosine gap for {CAP_MAIN_MODELS}. "
)
CAP_ROUTING_MAIN = (
    rf"Per-layer test routing assignment accuracy for {CAP_MAIN_MODELS}, overall and among "
    rf"existing and new witnesses, {CAP_FRACTION}. "
)
CAP_RANK_SINGLE_MAIN = (
    rf"Per-layer test routing DirAcc@1 for {CAP_MAIN_MODELS}, {CAP_FRACTION}, "
    rf"{CAP_SINGLE_RUN}. Existing, New: assignment accuracy on existing or new witnesses. "
)


def _selected_layer_caption(method_label: str, task: str) -> str:
    """The bold-row key: the train-only rule, tied to the headline cell.

    ``task`` is ``"taskA"`` (selection by training AUROC) or ``"taskB"``
    (selection by training DirAcc@1). ``method_label`` must be the headline
    column name (``ABTT`` or ``SIF+ABTT``) so the reader can find the cell in
    ``tab:headline_ci``; the layers themselves are listed in
    ``tab:selected_layers`` (issue #219), which Appendix I's prose cites.
    Only the layer and the ``method_label`` cells are bold, so the Base cells of
    that row never read as the Base headline cell.
    """
    metric_phrase = {
        "taskA": "AUROC",
        "taskB": "DirAcc@1",
    }[task]
    return (
        rf"Bold: the layer behind the {method_label} cell of "
        rf"Table~\ref{{{HEADLINE_LABEL[task]}}} (best training {metric_phrase} under "
        rf"{method_label}) and its {method_label} values, not always the best test layer."
    )


def _sibling_caption(what: str, first_label: str) -> str:
    """Caption of a sibling table: what it shows, then a pointer to the key."""
    return what + rf" Layout and markers as in Table~\ref{{{first_label}}}."


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
            CAP_TASKA_MAIN + CAP_BASE_ABTT + _selected_layer_caption("ABTT", "taskA")
        ),
        label="tab:taskA_main",
        banner=r"Ranking (T5 encoders)",
        select_method="abtt_optimal",
        float_table=True,
    )
    emit_taskA(
        taskA_df,
        models=APPENDIX_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["aucroc", "gap"],
        out_tex=tex_dir / "taskA_appendix.tex",
        out_audit=audit_dir / "taskA_appendix.csv",
        caption=_sibling_caption(
            rf"Per-layer test ranking AUROC and cosine gap for {CAP_APPENDIX_MODELS}.",
            "tab:taskA_main",
        ),
        label="tab:taskA_appendix",
        banner=r"Ranking (embedding-trained models)",
        select_method="abtt_optimal",
    )
    emit_taskA(
        taskA_df,
        models=ALL_MODELS,
        methods=["baseline", "sif_only", "sif_abtt_fixed", "sif_abtt_optimal"],
        metrics=["aucroc"],
        out_tex=tex_dir / "taskA_appendix_sif.tex",
        out_audit=audit_dir / "taskA_appendix_sif.csv",
        caption=(
            r"Per-layer test ranking AUROC under the SIF variants for the six panel models. "
            r"SIF: frequency-weighted pooling; SIF+ABTT$_{10}$: SIF, then ABTT with $D{=}10$; "
            r"SIF+ABTT: $D$ tuned per layer on the training split. "
            r"Base and markers as in Table~\ref{tab:taskA_main}, with bold layers selected "
            r"under SIF+ABTT."
        ),
        label="tab:taskA_appendix_sif",
        banner=r"Ranking: SIF-suite AUROC (six models)",
        select_method="sif_abtt_optimal",
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
            CAP_ROUTING_MAIN + CAP_BASE_ABTT + _selected_layer_caption("ABTT", "taskB")
        ),
        label="tab:taskB_routing_main",
        banner=r"Routing (T5 encoders, witness-level, $\tau$-thresholded)",
        select_method="abtt_optimal",
        float_table=True,
    )
    emit_taskB_routing(
        taskA_df,
        models=APPENDIX_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["existing_acc", "new_acc", "overall_assignment_acc"],
        out_tex=tex_dir / "taskB_routing_appendix.tex",
        out_audit=audit_dir / "taskB_routing_appendix.csv",
        caption=_sibling_caption(
            rf"Per-layer test routing assignment accuracy, {CAP_FRACTION}, for "
            rf"{CAP_APPENDIX_MODELS}.",
            "tab:taskB_routing_main",
        ),
        label="tab:taskB_routing_appendix",
        banner=r"Routing (embedding-trained models)",
        select_method="abtt_optimal",
    )
    emit_taskB_routing(
        taskA_df,
        models=ALL_MODELS,
        methods=["baseline", "sif_only", "sif_abtt_fixed", "sif_abtt_optimal"],
        metrics=["overall_assignment_acc"],
        out_tex=tex_dir / "taskB_routing_appendix_sif.tex",
        out_audit=audit_dir / "taskB_routing_appendix_sif.csv",
        caption=(
            rf"Per-layer test routing assignment accuracy, {CAP_FRACTION}, for the six panel "
            r"models. Variants and markers as in Table~\ref{tab:taskA_appendix_sif}, with bold "
            r"layers selected by training DirAcc@1."
        ),
        label="tab:taskB_routing_appendix_sif",
        banner=r"Routing: SIF-suite assignment accuracy (six models)",
        select_method="sif_abtt_optimal",
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
            CAP_RANK_SINGLE_MAIN + CAP_BASE_ABTT + _selected_layer_caption("ABTT", "taskB")
        ),
        label="tab:taskB_ranking_main",
        banner=r"Routing: DirAcc@1 (T5 encoders, single seed)",
        select_method="abtt_optimal",
        float_table=True,
    )
    emit_taskB_ranking_single(
        taskA_df,
        models=APPENDIX_MODELS,
        methods=["baseline", "abtt_optimal"],
        metrics=["dir_acc_at_1", "existing_acc", "new_acc"],
        out_tex=tex_dir / "taskB_ranking_appendix.tex",
        out_audit=audit_dir / "taskB_ranking_appendix.csv",
        caption=_sibling_caption(
            rf"Per-layer test routing DirAcc@1, {CAP_FRACTION}, for {CAP_APPENDIX_MODELS}, "
            rf"{CAP_SINGLE_RUN}.",
            "tab:taskB_ranking_main",
        ),
        label="tab:taskB_ranking_appendix",
        banner=r"Routing: DirAcc@1 (embedding-trained models, single seed)",
        select_method="abtt_optimal",
    )

    # -------- Task B: mseed SIF+ABTT (appendix, preserves pre-restructure content) --------
    # Bold rows come from the single-seed train metric, the same rule as the
    # headline tables and as tables/taskB_topk.tex (issue #175).
    mseed_layers = train_selected_layers(
        taskA_df, ALL_MODELS, method=MSEED_METHOD, metric=TASKB_SELECT_METRIC
    )
    emit_taskB_ranking_mseed(
        taskB_df,
        models=ALL_MODELS,
        methods=["baseline", MSEED_METHOD],
        out_tex=tex_dir / "taskB_ranking_appendix_mseed.tex",
        out_audit=audit_dir / "taskB_ranking_appendix_mseed.csv",
        caption=(
            rf"Per-layer test routing DirAcc@1 for the six panel models, overall and on existing "
            rf"or new witnesses, {CAP_FRACTION} (mean $\pm$ std over five reseedings). "
            r"Bold: the reported layer and its SIF+ABTT values. "
            + SELECTION_RULE_CAPTION
        ),
        label="tab:taskB_ranking_appendix_mseed",
        banner=r"Routing: DirAcc@1 (five reseedings, mean $\pm$ std, SIF+ABTT)",
        selected_layers=mseed_layers,
    )

    print("wrote audit CSVs to", audit_dir)
    print("wrote tex tables to", tex_dir)


if __name__ == "__main__":
    main()
