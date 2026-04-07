#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
PHASE11_DIR="${PHASE11_DIR:-$RUNS_ROOT/phase11_release}"
PHASE12_DIR="${PHASE12_DIR:-$RUNS_ROOT/phase12_release}"
PHASE12E_DIR="${PHASE12E_DIR:-$RUNS_ROOT/phase12e_release}"
SPLIT_CSV="${SPLIT_CSV:-$RUNS_ROOT/phase9/phase9_split.csv}"
PYTHON_BIN="${PYTHON_BIN:-conda run -n localLatin python}"

mkdir -p "$PHASE12_DIR/pcs" "$PHASE12_DIR/taskb" "$PHASE12E_DIR"

$PYTHON_BIN scripts/run_phase12_prepare_pcs.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --hidden_subdir "hidden_mean_tokempty" \
  --out_dir "$PHASE12_DIR/pcs" \
  --models "bowphs/LaTa,bowphs/PhilTa,sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,google/mt5-base" \
  --fail_if_missing

$PYTHON_BIN scripts/run_phase11_taskb_human_simulation.py \
  --results_csv "$PHASE11_DIR/phase11_results.csv" \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_dir "$PHASE12_DIR/taskb" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty"

$PYTHON_BIN scripts/run_phase12e_select_examples.py \
  --taskb_csv "$PHASE12_DIR/taskb/taskb_top5_predictions.csv" \
  --out_csv "$PHASE12E_DIR/phase12e_examples.csv"

$PYTHON_BIN scripts/run_phase12e_pair_explanations.py \
  --examples_csv "$PHASE12E_DIR/phase12e_examples.csv" \
  --pc_root "$PHASE12_DIR/pcs" \
  --out_dir "$PHASE12E_DIR" \
  --token_filter tokenizer_empty

$PYTHON_BIN scripts/run_phase12e_visualize.py \
  --examples_csv "$PHASE12E_DIR/phase12e_examples.csv" \
  --artifacts_dir "$PHASE12E_DIR/artifacts" \
  --fig_dir "$PHASE12E_DIR/figures"
