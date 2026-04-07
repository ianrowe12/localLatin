#!/bin/bash
set -euo pipefail

DEFAULT_WORK="${WORK:-$HOME}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_WORK/localLatin}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
BASE_RUN_DIR="${BASE_RUN_DIR:-$RUNS_ROOT/ff1_lata_postact/run_20260125_154356}"
PHASE5_ROOT="${PHASE5_ROOT:-$RUNS_ROOT/phase5_hidden12_mean}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
TOP_N="${TOP_N:-5}"

LAYER="${LAYER:-12}"
VAR_DROPS="${VAR_DROPS:-25,50}"
CORR_THRESH="${CORR_THRESH:-0.90,0.85}"
CORR_TARGET_DIMS="${CORR_TARGET_DIMS:-576}"
PC_REMOVE="${PC_REMOVE:-1,3,5,10}"
CHAIN_VAR_DROPS="${CHAIN_VAR_DROPS:-25}"
CHAIN_PC_REMOVE="${CHAIN_PC_REMOVE:-3,5}"
PC_REMOVE_CENTER="${PC_REMOVE_CENTER:-1}"
INCLUDE_CENTER="${INCLUDE_CENTER:-1}"
INCLUDE_STANDARDIZE="${INCLUDE_STANDARDIZE:-1}"
CHAIN_CENTER_PC_REMOVE="${CHAIN_CENTER_PC_REMOVE:-1}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT not found: $REPO_ROOT"
  exit 1
fi

cd "$REPO_ROOT"

echo "Phase 5 run id: $RUN_ID"
echo "Base run: $BASE_RUN_DIR"
echo "Output root: $PHASE5_ROOT"

PHASE5_CMD=(
  python src/phase5_hidden12_cli.py
  --base_run_dir "$BASE_RUN_DIR"
  --out_root_dir "$PHASE5_ROOT"
  --run_id "$RUN_ID"
  --layer "$LAYER"
  --variance_drop_percents "$VAR_DROPS"
  --corr_thresholds "$CORR_THRESH"
  --corr_target_dims "$CORR_TARGET_DIMS"
  --pc_remove_components "$PC_REMOVE"
  --chain_variance_drop_percents "$CHAIN_VAR_DROPS"
  --chain_pc_remove_components "$CHAIN_PC_REMOVE"
)

if [ "$PC_REMOVE_CENTER" = "1" ]; then
  PHASE5_CMD+=(--pc_remove_center)
fi
if [ "$INCLUDE_CENTER" = "1" ]; then
  PHASE5_CMD+=(--include_center)
fi
if [ "$INCLUDE_STANDARDIZE" = "1" ]; then
  PHASE5_CMD+=(--include_standardize)
fi
if [ "$CHAIN_CENTER_PC_REMOVE" = "1" ]; then
  PHASE5_CMD+=(--chain_center_pc_remove)
fi

"${PHASE5_CMD[@]}"

PHASE5_RUN_DIR="$PHASE5_ROOT/$RUN_ID"

python src/phase5_full_eval_cli.py \
  --screening_csv "$PHASE5_RUN_DIR/phase5_screening_scoreboard.csv" \
  --top_n "$TOP_N" \
  --layer "$LAYER" \
  --output_csv "$PHASE5_RUN_DIR/phase5_full_eval_scoreboard.csv"

echo "Phase 5 done."
echo "Outputs: $PHASE5_RUN_DIR"
