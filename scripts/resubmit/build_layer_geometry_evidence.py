"""Join layer geometry diagnostics to retrieval outcomes and layer rules."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
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

DIAGNOSTIC_COLUMNS = [
    "anisotropy_mean_cosine",
    "cosine_std",
    "cosine_iqr",
    "pc1_variance_ratio",
    "pc10_cumulative_variance_ratio",
    "effective_rank_entropy",
    "delta_anisotropy_mean_cosine",
    "delta_pc1_variance_ratio",
    "delta_effective_rank_entropy",
    "delta_cosine_std",
    "delta_cosine_iqr",
]

RETRIEVAL_TARGETS = [
    "aucroc__baseline",
    "aucroc__abtt_optimal",
    "aucroc_gain__abtt_optimal_minus_baseline",
    "gap__baseline",
    "gap__abtt_optimal",
    "gap_gain__abtt_optimal_minus_baseline",
    "dir_acc_at_1__baseline",
    "dir_acc_at_1__abtt_optimal",
    "dir_acc_at_1_gain__abtt_optimal_minus_baseline",
    "overall_assignment_acc__baseline",
    "overall_assignment_acc__abtt_optimal",
    "overall_assignment_acc_gain__abtt_optimal_minus_baseline",
    "train_dir_acc_at_1__abtt_optimal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics_csv",
        default="runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv",
    )
    parser.add_argument(
        "--retrieval_csv",
        default="runs/active/resubmit/results/phase_resubmit_results.csv",
    )
    parser.add_argument(
        "--taskb_csv",
        default="runs/active/resubmit/taskb_mseed/aggregated_results.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="runs/active/resubmit/layer_diagnostics",
    )
    parser.add_argument(
        "--tolerance_pp",
        type=float,
        default=0.5,
        help="Tie band in percentage points for layer-rule candidates.",
    )
    return parser.parse_args()


def pivot_retrieval(retrieval: pd.DataFrame) -> pd.DataFrame:
    keep = retrieval[
        (retrieval["repr"] == "hidden")
        & (retrieval["pooling"] == "mean")
        & (retrieval["method"].isin(["baseline", "abtt_optimal"]))
    ].copy()
    values = [
        "D",
        "tau",
        "aucroc",
        "gap",
        "dir_acc_at_1",
        "dir_acc_at_3",
        "overall_assignment_acc",
        "existing_acc",
        "new_acc",
        "train_aucroc",
        "train_dir_acc_at_1",
    ]
    wide = keep.pivot_table(
        index=["model", "repr", "pooling", "layer"],
        columns="method",
        values=values,
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    wide = wide.reset_index()

    for metric in [
        "aucroc",
        "gap",
        "dir_acc_at_1",
        "dir_acc_at_3",
        "overall_assignment_acc",
        "existing_acc",
        "new_acc",
    ]:
        base = f"{metric}__baseline"
        abtt = f"{metric}__abtt_optimal"
        if base in wide.columns and abtt in wide.columns:
            wide[f"{metric}_gain__abtt_optimal_minus_baseline"] = wide[abtt] - wide[base]
    return wide


def pivot_taskb_mseed(taskb: pd.DataFrame) -> pd.DataFrame:
    if taskb.empty:
        return pd.DataFrame()
    keep = taskb[
        (taskb["repr"] == "hidden")
        & (taskb["pooling"] == "mean")
        & (taskb["method"] == "baseline")
    ].copy()
    if keep.empty:
        return pd.DataFrame()
    cols = [
        "model",
        "repr",
        "pooling",
        "layer",
        "dir_acc_at_1_mean",
        "dir_acc_at_1_std",
        "overall_assignment_acc_mean",
        "overall_assignment_acc_std",
    ]
    keep = keep[[c for c in cols if c in keep.columns]]
    rename = {
        "dir_acc_at_1_mean": "mseed_baseline_dir_acc_at_1_mean",
        "dir_acc_at_1_std": "mseed_baseline_dir_acc_at_1_std",
        "overall_assignment_acc_mean": "mseed_baseline_overall_assignment_acc_mean",
        "overall_assignment_acc_std": "mseed_baseline_overall_assignment_acc_std",
    }
    return keep.rename(columns=rename)


def build_join(
    diagnostics: pd.DataFrame,
    retrieval: pd.DataFrame,
    taskb: pd.DataFrame,
) -> pd.DataFrame:
    geo = diagnostics[
        (diagnostics["split"] == "test")
        & (diagnostics["view"] == "raw")
        & (diagnostics["repr"] == "hidden")
        & (diagnostics["pooling"] == "mean")
    ].copy()
    retrieval_wide = pivot_retrieval(retrieval)
    joined = geo.merge(
        retrieval_wide,
        on=["model", "repr", "pooling", "layer"],
        how="left",
        validate="one_to_one",
    )
    taskb_wide = pivot_taskb_mseed(taskb)
    if not taskb_wide.empty:
        joined = joined.merge(
            taskb_wide,
            on=["model", "repr", "pooling", "layer"],
            how="left",
            validate="one_to_one",
        )
    joined["model_display"] = joined["model"].map(MODEL_DISPLAY)
    joined["model_group"] = np.where(joined["model"].isin(MAIN_MODELS), "main", "appendix")
    return joined.sort_values(["model_group", "model", "layer"]).reset_index(drop=True)


def finite_pair_count(a: pd.Series, b: pd.Series) -> int:
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    return int(mask.sum())


def corr_value(a: pd.Series, b: pd.Series, method: str) -> float:
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    return float(a[mask].corr(b[mask], method=method))


def correlation_rows(joined: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    groups: List[tuple[str, pd.DataFrame]] = [
        ("main_all", joined[joined["model"].isin(MAIN_MODELS)]),
        ("appendix_all", joined[joined["model"].isin(APPENDIX_MODELS)]),
        ("all_models", joined),
    ]
    groups.extend((str(model), sub) for model, sub in joined.groupby("model"))

    for group_name, sub in groups:
        if sub.empty:
            continue
        for diagnostic in DIAGNOSTIC_COLUMNS:
            if diagnostic not in sub.columns:
                continue
            for target in RETRIEVAL_TARGETS:
                if target not in sub.columns:
                    continue
                rows.append(
                    {
                        "group": group_name,
                        "diagnostic": diagnostic,
                        "retrieval_target": target,
                        "n": finite_pair_count(sub[diagnostic], sub[target]),
                        "pearson": corr_value(sub[diagnostic], sub[target], "pearson"),
                        "spearman": corr_value(sub[diagnostic], sub[target], "spearman"),
                    }
                )
    return rows


def earliest_within_band(sub: pd.DataFrame, metric: str, tolerance_pp: float) -> tuple[int, float, float]:
    values = sub[["layer", metric]].dropna().copy()
    if values.empty:
        return -1, float("nan"), float("nan")
    best_value = float(values[metric].max())
    threshold = best_value - tolerance_pp / 100.0
    chosen = values[values[metric] >= threshold].sort_values("layer").iloc[0]
    return int(chosen["layer"]), float(chosen[metric]), best_value


def layer_rule_rows(joined: pd.DataFrame, tolerance_pp: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, sub in joined.groupby("model"):
        sub = sub.sort_values("layer")
        train_metric = "train_dir_acc_at_1__abtt_optimal"
        fallback_metric = "dir_acc_at_1__abtt_optimal"
        metric = train_metric if train_metric in sub.columns and sub[train_metric].notna().any() else fallback_metric
        layer, chosen_value, best_value = earliest_within_band(sub, metric, tolerance_pp)

        raw_anis_col = "anisotropy_mean_cosine"
        collapse_layer = -1
        if raw_anis_col in sub.columns and sub[raw_anis_col].notna().any():
            collapse_layer = int(sub.loc[sub[raw_anis_col].idxmax(), "layer"])

        eff_rank_layer = -1
        if "effective_rank_entropy" in sub.columns and sub["effective_rank_entropy"].notna().any():
            eff_rank_layer = int(sub.loc[sub["effective_rank_entropy"].idxmin(), "layer"])

        pc1_layer = -1
        if "pc1_variance_ratio" in sub.columns and sub["pc1_variance_ratio"].notna().any():
            pc1_layer = int(sub.loc[sub["pc1_variance_ratio"].idxmax(), "layer"])

        retrieval_test_layer, retrieval_test_value, retrieval_test_best = earliest_within_band(
            sub, fallback_metric, tolerance_pp
        )
        rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY.get(model_name, model_name),
                "recommended_operational_layer": layer,
                "selection_metric": metric,
                "selected_metric_value": chosen_value,
                "best_metric_value": best_value,
                "tie_band_pp": tolerance_pp,
                "fallback_test_dir_acc_layer": retrieval_test_layer,
                "fallback_test_dir_acc_value": retrieval_test_value,
                "fallback_test_dir_acc_best": retrieval_test_best,
                "recovered_collapse_layer_by_max_anisotropy": collapse_layer,
                "diagnostic_layer_by_min_effective_rank": eff_rank_layer,
                "diagnostic_layer_by_max_pc1_dominance": pc1_layer,
                "recommendation": "attribute_retrieval_selected_layer",
                "note": (
                    "Geometry layers diagnose collapse/recovery; operational attribution "
                    "uses the earliest train-selected retrieval layer in the tied band."
                ),
            }
        )
    return rows


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    diagnostics_csv = Path(args.diagnostics_csv)
    retrieval_csv = Path(args.retrieval_csv)
    taskb_csv = Path(args.taskb_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(diagnostics_csv)
    retrieval = pd.read_csv(retrieval_csv)
    taskb = pd.read_csv(taskb_csv) if taskb_csv.exists() else pd.DataFrame()

    joined = build_join(diagnostics, retrieval, taskb)
    main_join = joined[joined["model"].isin(MAIN_MODELS)].copy()
    appendix_join = joined[joined["model"].isin(APPENDIX_MODELS)].copy()
    correlations = pd.DataFrame(correlation_rows(joined))
    layer_rules = pd.DataFrame(layer_rule_rows(joined, args.tolerance_pp))

    outputs = {
        "geometry_retrieval_join_all": out_dir / "geometry_retrieval_join_all.csv",
        "geometry_retrieval_join_main": out_dir / "geometry_retrieval_join_main.csv",
        "geometry_retrieval_join_appendix": out_dir / "geometry_retrieval_join_appendix.csv",
        "geometry_correlation_summary": out_dir / "geometry_correlation_summary.csv",
        "layer_rule_candidates": out_dir / "layer_rule_candidates.csv",
    }
    write_csv(joined, outputs["geometry_retrieval_join_all"])
    write_csv(main_join, outputs["geometry_retrieval_join_main"])
    write_csv(appendix_join, outputs["geometry_retrieval_join_appendix"])
    write_csv(correlations, outputs["geometry_correlation_summary"])
    write_csv(layer_rules, outputs["layer_rule_candidates"])

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/resubmit/build_layer_geometry_evidence.py",
        "diagnostics_csv": str(diagnostics_csv),
        "retrieval_csv": str(retrieval_csv),
        "taskb_csv": str(taskb_csv),
        "tolerance_pp": args.tolerance_pp,
        "n_joined_rows": int(len(joined)),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "claim_guardrail": (
            "Correlation rows validate associations only. They are not layer selectors; "
            "layer rules use train-only retrieval when available."
        ),
    }
    (out_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote joined evidence tables to {out_dir}")


if __name__ == "__main__":
    main()
