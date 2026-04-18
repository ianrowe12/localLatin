"""Build a mean-vs-lasttok pooling comparison table for the appendix.

Reads two result CSVs (one for mean pooling, one for last-token pooling,
both filtered to ABTT methods), picks the best layer per (model, pooling)
pair by `--select_on` (default `overall_assignment_acc`), and writes a
compact LaTeX appendix table mirroring the Task A per-model table style.

The script tolerates missing CSVs: if either/both inputs are absent, it
emits a placeholder .tex file with TODO comments so downstream work can
proceed before the evaluation pipeline finishes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-8B",
}
MODEL_ORDER = list(MODEL_DISPLAY.values())  # canonical display-name ordering

POOL_DISPLAY = {"mean": "Mean", "lasttok": "LastTok"}
POOL_ORDER = ["Mean", "LastTok"]

ALLOWED_METHODS = {"abtt_fixed", "abtt_optimal"}
ALLOWED_POOLINGS = {"mean", "lasttok"}

CAPTION = (
    "Mean vs.\\ last-token pooling on the balanced split, restricted to ABTT "
    "post-processing. Best layer per (model, pooling) selected by Task A "
    "assignment accuracy. Task B metrics (Acc@1, cosine gap) reported at that "
    "same layer."
)
LABEL = "tab:lasttok_comparison"
COLUMN_FORMAT = "llccccc"
HEADER = "Model & Pool & Layer & Method & Assign Acc & Acc@1 & Gap \\\\"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mean_csv",
        default="runs/active/resubmit/results/lasttok/phase_resubmit_results_mean_abttonly.csv",
    )
    p.add_argument(
        "--lasttok_csv",
        default="runs/active/resubmit/results/lasttok/phase_resubmit_results_lasttok.csv",
    )
    p.add_argument(
        "--out_tex",
        default="overleaf_drafts/tables/appendix_lasttok_comparison.tex",
    )
    p.add_argument(
        "--select_on",
        default="overall_assignment_acc",
        help="Column used to pick the best layer per (model, pooling) pair.",
    )
    return p.parse_args()


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"WARN: {path} not found — skipping.", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - tolerate malformed inputs
        print(f"WARN: failed to read {path}: {exc}", file=sys.stderr)
        return None


def _format_method(row: pd.Series) -> str:
    method = row["method"]
    try:
        d_val = int(row["D"]) if pd.notna(row["D"]) else None
    except (TypeError, ValueError):
        d_val = None
    if method == "abtt_fixed":
        if d_val is not None and d_val != 10:
            return f"ABTT (D={d_val})"
        return "ABTT (D=10)"
    if method == "abtt_optimal":
        if d_val is not None:
            return f"ABTT (D*={d_val})"
        return "ABTT (tuned D)"
    return str(method)


def build_table(df: pd.DataFrame, select_on: str) -> pd.DataFrame:
    if select_on not in df.columns:
        raise SystemExit(
            f"--select_on={select_on!r} not in CSV columns: {list(df.columns)}"
        )
    keep = df[
        df["method"].isin(ALLOWED_METHODS)
        & df["pooling"].isin(ALLOWED_POOLINGS)
        & df[select_on].notna()
    ].copy()
    if keep.empty:
        return keep
    idx = keep.groupby(["model", "pooling"])[select_on].idxmax()
    best = keep.loc[idx].copy()

    best["Model"] = best["model"].map(MODEL_DISPLAY).fillna(best["model"])
    best["Pool"] = best["pooling"].map(POOL_DISPLAY).fillna(best["pooling"])
    best["Layer"] = best["layer"].astype(int)
    best["Method"] = best.apply(_format_method, axis=1)
    best["Assign Acc"] = best["overall_assignment_acc"].map(lambda v: f"{v:.3f}")
    best["Acc@1"] = best["dir_acc_at_1"].map(lambda v: f"{v:.3f}")
    best["Gap"] = best["gap"].map(lambda v: f"{v:.3f}")

    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    pool_rank = {p: i for i, p in enumerate(POOL_ORDER)}
    best["_model_rank"] = best["Model"].map(lambda m: model_rank.get(m, len(MODEL_ORDER)))
    best["_pool_rank"] = best["Pool"].map(lambda p: pool_rank.get(p, len(POOL_ORDER)))
    best = best.sort_values(["_model_rank", "Model", "_pool_rank"]).reset_index(drop=True)

    display_cols = ["Model", "Pool", "Layer", "Method", "Assign Acc", "Acc@1", "Gap"]
    return best[display_cols]


def _missing_pairs(table: pd.DataFrame) -> list[tuple[str, str]]:
    present = set(zip(table["Model"].tolist(), table["Pool"].tolist())) if not table.empty else set()
    missing = []
    for model in MODEL_ORDER:
        for pool in POOL_ORDER:
            if (model, pool) not in present:
                missing.append((model, pool))
    return missing


def _render_body(table: pd.DataFrame) -> str:
    if table.empty:
        return "% (no rows — both inputs empty after filtering)"
    lines = []
    for _, row in table.iterrows():
        cells = [
            row["Model"],
            row["Pool"],
            str(row["Layer"]),
            row["Method"],
            row["Assign Acc"],
            row["Acc@1"],
            row["Gap"],
        ]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def _render_tex(table: pd.DataFrame, todo_comments: list[str]) -> str:
    header_comments = "\n".join(todo_comments)
    if header_comments:
        header_comments += "\n"
    body = _render_body(table)
    return (
        f"{header_comments}\\begin{{table*}}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        f"\\caption{{{CAPTION}}}\n"
        f"\\label{{{LABEL}}}\n"
        f"\\begin{{tabular}}{{{COLUMN_FORMAT}}}\n"
        "\\toprule\n"
        f"{HEADER}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def _slug_for_sbatch(model_display: str) -> str:
    return model_display.lower().replace(".", "").replace("-", "_")


def main() -> None:
    args = parse_args()
    mean_path = Path(args.mean_csv)
    lasttok_path = Path(args.lasttok_csv)
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    mean_df = _load_csv(mean_path)
    lasttok_df = _load_csv(lasttok_path)

    todo_comments: list[str] = []

    if mean_df is None and lasttok_df is None:
        todo_comments.append(
            "% TODO: both CSVs missing — run slurm/resubmit/lasttok_evaluate.sbatch first"
        )
        tex = _render_tex(pd.DataFrame(columns=["Model", "Pool", "Layer", "Method", "Assign Acc", "Acc@1", "Gap"]), todo_comments)
        out_tex.write_text(tex)
        print(f"wrote placeholder {out_tex} (both CSVs missing)")
        return

    frames = [df for df in (mean_df, lasttok_df) if df is not None]
    joined = pd.concat(frames, ignore_index=True)
    table = build_table(joined, args.select_on)

    for model, pool in _missing_pairs(table):
        slug = _slug_for_sbatch(model)
        pool_slug = "mean" if pool == "Mean" else "lasttok"
        todo_comments.append(
            f"% TODO: {model} {pool_slug} missing — rerun sbatch lasttok_extract_{slug}.sbatch"
        )

    tex = _render_tex(table, todo_comments)
    out_tex.write_text(tex)
    print(f"selection metric: {args.select_on}")
    print(f"wrote {out_tex}")
    if not table.empty:
        print(table.to_string(index=False))
    if todo_comments:
        print("TODO comments emitted:")
        for c in todo_comments:
            print(f"  {c}")


if __name__ == "__main__":
    main()
