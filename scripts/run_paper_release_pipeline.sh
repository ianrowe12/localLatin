#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
PHASE11_DIR="${PHASE11_DIR:-$RUNS_ROOT/phase11_release}"
PHASE12_DIR="${PHASE12_DIR:-$RUNS_ROOT/phase12_release}"
PHASE12E_DIR="${PHASE12E_DIR:-$RUNS_ROOT/phase12e_release}"
PAPER_DIR="${PAPER_DIR:-$RUNS_ROOT/paper_release_assets}"
PYTHON_BIN="${PYTHON_BIN:-conda run -n localLatin python}"

bash "$REPO_ROOT/scripts/run_phase11_tokempty_pipeline.sh"
bash "$REPO_ROOT/scripts/run_phase12e_release_pipeline.sh"

$PYTHON_BIN scripts/package_paper_release_assets.py \
  --phase11_csv "$PHASE11_DIR/phase11_results.csv" \
  --phase11_fig_dir "$PHASE11_DIR/figures" \
  --dist_dir "$PHASE11_DIR/distributions" \
  --taskb_dir "$PHASE12_DIR/taskb" \
  --phase12e_fig_dir "$PHASE12E_DIR/figures" \
  --out_dir "$PAPER_DIR"
