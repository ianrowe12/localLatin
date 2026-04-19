"""Build per-layer Task A and Task B tables (baseline vs sif_abtt_optimal).

Task framing (post-2026-04-19 cleanup):
  Task A is the *purely pairwise* duplicate-detection task. Inputs are pairs of
  passages and the only Task A metrics are AUROC and cosine gap on the n x n
  cosine matrix. No directory knowledge, no threshold tau, no routing.
  Task B is everything that requires directory knowledge: top-K human-assisted
  ranking *and* autonomous file-level routing with the learned threshold tau.

We restrict to two methods: `baseline` (reference) and `sif_abtt_optimal` (the
D-tuned SIF+ABTT variant the advisor called out as "just ABTT"). Other methods
(sif_only, abtt_fixed, abtt_optimal, sif_abtt_fixed, whitening) are
deliberately dropped from the main results tables.

Three tables are emitted:
  - Task A pairwise: AUROC + cosine gap per layer (single-seed eval).
  - Task B ranking: mseed Acc@1 + existing/new decomposition per layer.
  - Task B routing: single-seed existing/new/overall assignment accuracy per
    layer (autonomous file-level routing with tau).

All tables bold the best layer per model (selected on the ABTT column).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "google/mt5-base": "mT5-base",
}
MODEL_ORDER = list(MODEL_DISPLAY)

BASELINE = "baseline"
ABTT = "sif_abtt_optimal"


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
        help="Where per-layer CSVs are written (for audit / downstream use).",
    )
    p.add_argument(
        "--out_taskA_tex",
        default="overleaf_drafts/tables/taskA_per_model.tex",
    )
    p.add_argument(
        "--out_taskB_tex",
        default="overleaf_drafts/tables/taskB_per_layer.tex",
    )
    p.add_argument(
        "--out_taskB_routing_tex",
        default="overleaf_drafts/tables/taskB_routing_per_layer.tex",
    )
    return p.parse_args()


def _filter_two_methods(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[df["method"].isin([BASELINE, ABTT])].copy()
    missing = set(df["model"].unique()) - set(MODEL_DISPLAY)
    if missing:
        raise SystemExit(f"Unknown models in CSV: {missing}")
    return keep


def _ordered_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present]


def build_taskA(df: pd.DataFrame) -> pd.DataFrame:
    """Per-layer Task A: pairwise AUROC + cosine gap, baseline vs ABTT.

    Task A under the cleaned framing is purely pairwise: for the n x n cosine
    matrix on test embeddings, AUROC and cosine gap measure how well same-
    directory pairs separate from different-directory pairs. No threshold,
    no routing decision, no directory knowledge by the model.
    """
    df = _filter_two_methods(df)
    metric_cols = ["aucroc", "gap"]
    wide = df.pivot_table(
        index=["model", "layer"],
        columns="method",
        values=metric_cols,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["model_display"] = wide["model"].map(MODEL_DISPLAY)
    wide = wide.sort_values(
        ["model", "layer"],
        key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)}) if s.name == "model" else s,
    ).reset_index(drop=True)
    return wide


def build_taskB_routing(df: pd.DataFrame) -> pd.DataFrame:
    """Per-layer Task B routing: single-seed existing/new/overall assignment
    accuracy from the file-level autonomous-routing pipeline (uses learned
    threshold tau on train).
    """
    df = _filter_two_methods(df)
    metric_cols = ["existing_acc", "new_acc", "overall_assignment_acc"]
    wide = df.pivot_table(
        index=["model", "layer"],
        columns="method",
        values=metric_cols,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["model_display"] = wide["model"].map(MODEL_DISPLAY)
    wide = wide.sort_values(
        ["model", "layer"],
        key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)}) if s.name == "model" else s,
    ).reset_index(drop=True)
    return wide


def build_taskB(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-format per-layer Task B rows with baseline + ABTT columns
    (mean and std preserved for later formatting).

    Note: in the mseed aggregation, `overall_assignment_acc` is numerically
    identical to `dir_acc_at_1` (the Task B "assignment" decision is just the
    top-1 pick), so we drop it and report `existing_acc` / `new_acc` instead,
    mirroring Task A's existing-vs-new decomposition.
    """
    df = _filter_two_methods(df)
    value_cols = [
        "dir_acc_at_1_mean",
        "dir_acc_at_1_std",
        "existing_acc_mean",
        "existing_acc_std",
        "new_acc_mean",
        "new_acc_std",
    ]
    wide = df.pivot_table(
        index=["model", "layer"],
        columns="method",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["model_display"] = wide["model"].map(MODEL_DISPLAY)
    wide = wide.sort_values(
        ["model", "layer"],
        key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)}) if s.name == "model" else s,
    ).reset_index(drop=True)
    return wide


def _fmt_pm(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def write_taskA_tex(wide: pd.DataFrame, out_path: Path) -> None:
    # Best-layer per model selected by ABTT AUROC (the headline pairwise metric)
    best_by_model = wide.groupby("model")["aucroc_sif_abtt_optimal"].idxmax()
    best_rows = set(best_by_model.values)

    col_spec = "llcccc"  # model, layer, 2 metrics x 2 methods = 4
    metric_headers = [
        r"\makecell{AUROC\\(base)}",
        r"\makecell{AUROC\\(ABTT)}",
        r"\makecell{Cosine gap\\(base)}",
        r"\makecell{Cosine gap\\(ABTT)}",
    ]

    caption = (
        r"Per-layer Task~A pairwise duplicate-detection metrics for each model, comparing "
        r"\texttt{baseline} (mean pooling, no correction) against "
        r"\texttt{sif\_abtt\_optimal} (SIF weighting plus ABTT with $D$ tuned per layer on the train split). "
        r"Both metrics are computed over the test n$\times$n cosine matrix with no threshold and no directory "
        r"routing: AUROC treats same-directory test pairs as positives, and cosine gap is the difference between "
        r"mean same-directory and mean different-directory cosine. Rows in bold mark the best layer per model, "
        r"selected by ABTT AUROC."
    )

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows

        def f(x: float) -> str:
            s = "--" if pd.isna(x) else f"{x:.3f}"
            return (r"\textbf{" + s + r"}") if is_best else s

        layer_cell = (r"\textbf{" + str(int(row["layer"])) + r"}") if is_best else str(int(row["layer"]))
        return [
            row["model_display"],
            layer_cell,
            f(row["aucroc_baseline"]),
            f(row["aucroc_sif_abtt_optimal"]),
            f(row["gap_baseline"]),
            f(row["gap_sif_abtt_optimal"]),
        ]

    _write_longtable(
        wide=wide,
        row_cells=row_cells,
        col_spec=col_spec,
        metric_headers=metric_headers,
        banner=r"Task A: Pairwise Duplicate Detection",
        caption=caption,
        label="tab:taskA_per_layer",
        best_rows=best_rows,
        out_path=out_path,
    )


def write_taskB_routing_tex(wide: pd.DataFrame, out_path: Path) -> None:
    """Per-layer Task B autonomous routing accuracy table (single-seed)."""
    best_by_model = wide.groupby("model")["overall_assignment_acc_sif_abtt_optimal"].idxmax()
    best_rows = set(best_by_model.values)

    col_spec = "llcccccc"
    metric_headers = [
        r"\makecell{Existing\\(base)}",
        r"\makecell{Existing\\(ABTT)}",
        r"\makecell{New\\(base)}",
        r"\makecell{New\\(ABTT)}",
        r"\makecell{Overall\\(base)}",
        r"\makecell{Overall\\(ABTT)}",
    ]

    caption = (
        r"Per-layer Task~B autonomous routing accuracy for each model. For every test file we compute "
        r"$s_i = \max_{j \neq i} \cos(e_i, e_j)$ against the rest of the test set and compare to the "
        r"learned threshold $\tau$ (fit on train via $F_1$-optimal cut on same-directory vs.\ "
        r"different-directory pairs). The decision is binary first (existing if $s_i \geq \tau$, else new); "
        r"only files predicted existing are then routed to their top-1 neighbour's directory. We report "
        r"\textbf{existing accuracy} (restricted to files whose true partner is in the test set), "
        r"\textbf{new accuracy} (files that should be flagged as novel), and \textbf{overall assignment "
        r"accuracy}. Rows in bold mark the best layer per model, selected by ABTT overall assignment accuracy."
    )

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows

        def f(x: float) -> str:
            s = "--" if pd.isna(x) else f"{x:.3f}"
            return (r"\textbf{" + s + r"}") if is_best else s

        layer_cell = (r"\textbf{" + str(int(row["layer"])) + r"}") if is_best else str(int(row["layer"]))
        return [
            row["model_display"],
            layer_cell,
            f(row["existing_acc_baseline"]),
            f(row["existing_acc_sif_abtt_optimal"]),
            f(row["new_acc_baseline"]),
            f(row["new_acc_sif_abtt_optimal"]),
            f(row["overall_assignment_acc_baseline"]),
            f(row["overall_assignment_acc_sif_abtt_optimal"]),
        ]

    _write_longtable(
        wide=wide,
        row_cells=row_cells,
        col_spec=col_spec,
        metric_headers=metric_headers,
        banner=r"Task B: Autonomous Routing (file-level, $\tau$-thresholded)",
        caption=caption,
        label="tab:taskB_routing_per_layer",
        best_rows=best_rows,
        out_path=out_path,
    )


def _write_longtable(
    wide: pd.DataFrame,
    row_cells,
    col_spec: str,
    metric_headers: list[str],
    banner: str,
    caption: str,
    label: str,
    best_rows: set,
    out_path: Path,
) -> None:
    """Emit a longtable with a Task banner, repeated header on each page,
    midrule between model groups, and the `Model` column blanked on
    continuation rows within a group so the grouping reads cleanly.
    """
    ncols = len(col_spec)
    metric_span = ncols - 2
    banner_row = (
        r"\multicolumn{2}{c}{}"
        + f" & \\multicolumn{{{metric_span}}}{{c}}{{\\textbf{{{banner}}}}}"
        + r" \\"
    )
    header_row = r"\textbf{Model} & \textbf{Layer}"
    for h in metric_headers:
        header_row += " & " + h
    header_row += r" \\"

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{longtable}{" + col_spec + r"}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"} \\")
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
        cells = row_cells(i, row)
        # Blank the Model cell on continuation rows so visual grouping stays clean.
        if prev_model == row["model"]:
            cells[0] = ""
        lines.append(" & ".join(cells) + r" \\")
        prev_model = row["model"]

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_taskB_tex(wide: pd.DataFrame, out_path: Path) -> None:
    # Best-layer per model selected by ABTT dir_acc_at_1_mean
    best_by_model = wide.groupby("model")["dir_acc_at_1_mean_sif_abtt_optimal"].idxmax()
    best_rows = set(best_by_model.values)

    col_spec = "llcccccc"  # model, layer, 3 metrics x 2 methods = 6
    metric_headers = [
        r"\makecell{Acc@1\\(base)}",
        r"\makecell{Acc@1\\(ABTT)}",
        r"\makecell{Existing\\(base)}",
        r"\makecell{Existing\\(ABTT)}",
        r"\makecell{New\\(base)}",
        r"\makecell{New\\(ABTT)}",
    ]

    caption = (
        r"Per-layer Task~B accuracy for each model, averaged over $M=5$ query/reference "
        r"reseedings of the v2 train/test split (mean $\pm$ std). \texttt{baseline} is mean pooling "
        r"without correction; \texttt{sif\_abtt\_optimal} applies SIF weighting plus ABTT with $D$ "
        r"tuned per layer on the train split. Overall assignment accuracy coincides with Acc@1 in "
        r"the Task~B mseed pipeline (assignment = top-1 pick), so we instead report the existing "
        r"vs.\ new decomposition. Rows in bold mark the best layer per model, selected by ABTT Acc@1. "
        r"See Table~\ref{tab:taskB_routing_per_layer} for the complementary autonomous-routing view."
    )

    def row_cells(i: int, row: pd.Series) -> list[str]:
        is_best = i in best_rows

        def f(mean: float, std: float) -> str:
            s = _fmt_pm(mean, std)
            return (r"\textbf{" + s + r"}") if is_best else s

        layer_cell = (r"\textbf{" + str(int(row["layer"])) + r"}") if is_best else str(int(row["layer"]))
        return [
            row["model_display"],
            layer_cell,
            f(row["dir_acc_at_1_mean_baseline"], row["dir_acc_at_1_std_baseline"]),
            f(row["dir_acc_at_1_mean_sif_abtt_optimal"], row["dir_acc_at_1_std_sif_abtt_optimal"]),
            f(row["existing_acc_mean_baseline"], row["existing_acc_std_baseline"]),
            f(row["existing_acc_mean_sif_abtt_optimal"], row["existing_acc_std_sif_abtt_optimal"]),
            f(row["new_acc_mean_baseline"], row["new_acc_std_baseline"]),
            f(row["new_acc_mean_sif_abtt_optimal"], row["new_acc_std_sif_abtt_optimal"]),
        ]

    _write_longtable(
        wide=wide,
        row_cells=row_cells,
        col_spec=col_spec,
        metric_headers=metric_headers,
        banner=r"Task B: Top-K Ranking (5-seed mean $\pm$ std)",
        caption=caption,
        label="tab:taskB_per_layer",
        best_rows=best_rows,
        out_path=out_path,
    )


def main() -> None:
    args = parse_args()
    taskA_df = pd.read_csv(args.taskA_csv)
    taskB_df = pd.read_csv(args.taskB_csv)

    wideA = build_taskA(taskA_df)
    wideB = build_taskB(taskB_df)
    wideB_routing = build_taskB_routing(taskA_df)  # routing reuses single-seed CSV

    audit_dir = Path(args.out_dir_audit)
    audit_dir.mkdir(parents=True, exist_ok=True)
    wideA.to_csv(audit_dir / "taskA_per_layer.csv", index=False)
    wideB.to_csv(audit_dir / "taskB_per_layer.csv", index=False)
    wideB_routing.to_csv(audit_dir / "taskB_routing_per_layer.csv", index=False)

    write_taskA_tex(wideA, Path(args.out_taskA_tex))
    write_taskB_tex(wideB, Path(args.out_taskB_tex))
    write_taskB_routing_tex(wideB_routing, Path(args.out_taskB_routing_tex))

    print(f"wrote {audit_dir / 'taskA_per_layer.csv'} ({len(wideA)} rows)")
    print(f"wrote {audit_dir / 'taskB_per_layer.csv'} ({len(wideB)} rows)")
    print(f"wrote {audit_dir / 'taskB_routing_per_layer.csv'} ({len(wideB_routing)} rows)")
    print(f"wrote {args.out_taskA_tex}")
    print(f"wrote {args.out_taskB_tex}")
    print(f"wrote {args.out_taskB_routing_tex}")

    # Sanity: per-model row counts
    print("\nTask A rows per model:")
    print(wideA.groupby("model_display").size().to_string())
    print("\nTask B (ranking) rows per model:")
    print(wideB.groupby("model_display").size().to_string())
    print("\nTask B (routing) rows per model:")
    print(wideB_routing.groupby("model_display").size().to_string())


if __name__ == "__main__":
    main()
