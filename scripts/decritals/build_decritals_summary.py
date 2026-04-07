"""Build per-model summary table for the decritals cross-dataset run.

Reads:
  runs/active/decritals/results/decritals_taskA_results.csv
  runs/active/decritals/results/decritals_taskB_mseed/aggregated_results.csv

Writes:
  runs/active/decritals/results/decritals_per_model_summary.csv

For each of the 6 models, picks the best layer under the canon paper's main
configuration (repr=hidden, pooling=sif, method=sif_abtt_optimal) and reports
its Task A + Task B headline metrics.

Also prints to stdout any model whose decritals headline accuracy differs from
canon by more than 0.10 absolute (cross-dataset anomaly check).

Fails loudly if either decritals CSV is missing or if any model has all-NaN
rows for the chosen configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DECR_TASKA = REPO_ROOT / "runs/active/decritals/results/decritals_taskA_results.csv"
DECR_TASKB = REPO_ROOT / "runs/active/decritals/results/decritals_taskB_mseed/aggregated_results.csv"
OUT_CSV = REPO_ROOT / "runs/active/decritals/results/decritals_per_model_summary.csv"

CANON_TASKA = REPO_ROOT / "runs/active/resubmit/results/phase_resubmit_results.csv"
CANON_TASKB = REPO_ROOT / "runs/active/resubmit/taskb_mseed/aggregated_results.csv"

# Canon paper's main configuration: SIF + ABTT with optimal D, hidden states.
MAIN_REPR = "hidden"
MAIN_POOLING = "sif"
MAIN_METHOD = "sif_abtt_optimal"

MODELS = [
    "bowphs/LaTa",
    "bowphs/PhilTa",
    "google/mt5-base",
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
]

ANOMALY_THRESHOLD = 0.10


def filter_main_config(df: pd.DataFrame) -> pd.DataFrame:
    """Subset rows to the canon paper's main configuration."""
    return df[
        (df["repr"] == MAIN_REPR)
        & (df["pooling"] == MAIN_POOLING)
        & (df["method"] == MAIN_METHOD)
    ].copy()


def best_taskA_row(taska_df: pd.DataFrame, model: str) -> pd.Series | None:
    """Return the best Task A row for a model, picked by overall_assignment_acc."""
    sub = taska_df[taska_df["model"] == model]
    if sub.empty:
        return None
    if sub["overall_assignment_acc"].isna().all():
        return None
    return sub.loc[sub["overall_assignment_acc"].idxmax()]


def best_taskB_row(taskb_df: pd.DataFrame, model: str) -> pd.Series | None:
    """Return the best Task B row for a model, picked by overall_assignment_acc_mean."""
    sub = taskb_df[taskb_df["model"] == model]
    if sub.empty:
        return None
    if sub["overall_assignment_acc_mean"].isna().all():
        return None
    return sub.loc[sub["overall_assignment_acc_mean"].idxmax()]


def main():
    if not DECR_TASKA.exists():
        print(f"ERROR: {DECR_TASKA} does not exist. Did the eval job finish?", file=sys.stderr)
        sys.exit(1)
    if not DECR_TASKB.exists():
        print(f"ERROR: {DECR_TASKB} does not exist. Did the taskb_mseed job finish?", file=sys.stderr)
        sys.exit(1)

    decr_taska = filter_main_config(pd.read_csv(DECR_TASKA))
    decr_taskb = filter_main_config(pd.read_csv(DECR_TASKB))

    if decr_taska.empty:
        print(
            f"ERROR: no rows in {DECR_TASKA} match config "
            f"(repr={MAIN_REPR}, pooling={MAIN_POOLING}, method={MAIN_METHOD})",
            file=sys.stderr,
        )
        sys.exit(1)
    if decr_taskb.empty:
        print(
            f"ERROR: no rows in {DECR_TASKB} match config "
            f"(repr={MAIN_REPR}, pooling={MAIN_POOLING}, method={MAIN_METHOD})",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    missing = []
    for model in MODELS:
        a_row = best_taskA_row(decr_taska, model)
        b_row = best_taskB_row(decr_taskb, model)
        if a_row is None:
            missing.append(f"{model} (Task A)")
            continue
        if b_row is None:
            missing.append(f"{model} (Task B)")
            continue
        rows.append({
            "model": model,
            "taskA_best_layer": int(a_row["layer"]),
            "taskA_acc_at_1": float(a_row["dir_acc_at_1"]),
            "taskA_overall_assignment_acc": float(a_row["overall_assignment_acc"]),
            "taskA_aucroc": float(a_row["aucroc"]),
            "taskB_best_layer": int(b_row["layer"]),
            "taskB_acc_at_1_mean": float(b_row["dir_acc_at_1_mean"]),
            "taskB_acc_at_1_std": float(b_row["dir_acc_at_1_std"]),
        })

    if missing:
        print("ERROR: missing rows for the following models:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print()
    print("=== Decritals per-model summary ===")
    print(summary.to_string(index=False))
    print()

    # --- Cross-dataset anomaly check ---
    if not CANON_TASKA.exists() or not CANON_TASKB.exists():
        print(
            f"NOTE: skipping cross-dataset anomaly check "
            f"(canon CSVs missing at {CANON_TASKA} or {CANON_TASKB})"
        )
        return

    canon_taska = filter_main_config(pd.read_csv(CANON_TASKA))
    canon_taskb = filter_main_config(pd.read_csv(CANON_TASKB))

    print("=== Cross-dataset anomaly check (decritals vs canon) ===")
    print(f"Threshold for flagging: |delta| > {ANOMALY_THRESHOLD}")
    flagged = []
    for model in MODELS:
        canon_a = best_taskA_row(canon_taska, model)
        canon_b = best_taskB_row(canon_taskb, model)
        if canon_a is None or canon_b is None:
            print(f"  {model}: no canon baseline row, skipping")
            continue
        decr_a = best_taskA_row(decr_taska, model)
        decr_b = best_taskB_row(decr_taskb, model)
        delta_taska = float(decr_a["dir_acc_at_1"]) - float(canon_a["dir_acc_at_1"])
        delta_taskb = float(decr_b["dir_acc_at_1_mean"]) - float(canon_b["dir_acc_at_1_mean"])
        marker_a = "  FLAG" if abs(delta_taska) > ANOMALY_THRESHOLD else ""
        marker_b = "  FLAG" if abs(delta_taskb) > ANOMALY_THRESHOLD else ""
        print(
            f"  {model:60s}  "
            f"taskA_delta={delta_taska:+.3f}{marker_a}  "
            f"taskB_delta={delta_taskb:+.3f}{marker_b}"
        )
        if abs(delta_taska) > ANOMALY_THRESHOLD or abs(delta_taskb) > ANOMALY_THRESHOLD:
            flagged.append(model)

    if flagged:
        print()
        print(f"Models with cross-dataset anomalies: {flagged}")
    else:
        print()
        print("No cross-dataset anomalies above threshold.")


if __name__ == "__main__":
    main()
