#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
RESULTS_CSV="${RESULTS_CSV:-$RUNS_ROOT/phase11_filtered/phase11_results.csv}"
SPLIT_CSV="${SPLIT_CSV:-$RUNS_ROOT/phase9/phase9_split.csv}"
OUT_ROOT="${OUT_ROOT:-$RUNS_ROOT/phase12f_examples}"
PC_ROOT="${PC_ROOT:-$RUNS_ROOT/phase12_release/pcs}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="conda run -n localLatin python"
  else
    PYTHON_BIN="/u/irowerojas/.conda/envs/localLatin/bin/python"
  fi
fi

if [[ ! -f "$RESULTS_CSV" ]]; then
  RESULTS_CSV="$RUNS_ROOT/phase11/phase11_results.csv"
fi

if [[ ! -d "$PC_ROOT" ]]; then
  if [[ -d "$RUNS_ROOT/phase12_filtered/pcs" ]]; then
    PC_ROOT="$RUNS_ROOT/phase12_filtered/pcs"
  else
    PC_ROOT="$RUNS_ROOT/phase12/pcs"
  fi
fi

EXAMPLES_CSV="$OUT_ROOT/phase12f_examples.csv"
FIG_DIR="$OUT_ROOT/figures"

mkdir -p "$OUT_ROOT" "$FIG_DIR"

$PYTHON_BIN scripts/run_phase12f_select_pair_examples.py \
  --results_csv "$RESULTS_CSV" \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_csv "$EXAMPLES_CSV"

$PYTHON_BIN scripts/run_phase12e_pair_explanations.py \
  --examples_csv "$EXAMPLES_CSV" \
  --pc_root "$PC_ROOT" \
  --out_dir "$OUT_ROOT" \
  --token_filter tokenizer_empty \
  --trust_remote_code \
  --skip_existing

$PYTHON_BIN scripts/run_phase12f_visualize.py \
  --examples_csv "$EXAMPLES_CSV" \
  --artifacts_dir "$OUT_ROOT/artifacts" \
  --fig_dir "$FIG_DIR"
