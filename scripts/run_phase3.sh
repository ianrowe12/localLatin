#!/bin/bash
set -euo pipefail

DEFAULT_WORK="${WORK:-$HOME}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_WORK/localLatin}"
RUNS_ROOT="${RUNS_ROOT:-$DEFAULT_WORK/runs}"
BASE_RUN_DIR="${BASE_RUN_DIR:-$RUNS_ROOT/ff1_lata_postact/run_20260125_154356}"
PHASE3_ROOT="${PHASE3_ROOT:-$RUNS_ROOT/phase3_hidden12_mean}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
TOP_N="${TOP_N:-5}"

LAYER="${LAYER:-12}"
VAR_DROPS="${VAR_DROPS:-10,25,50,75,90}"
CORR_THRESH="${CORR_THRESH:-0.95,0.975,0.99}"
PCA_KS="${PCA_KS:-32,64,96,128,256,384,512,640}"
RAND_DROPS="${RAND_DROPS:-10,25,50,75,90}"
RAND_TRIALS="${RAND_TRIALS:-10}"
RAND_SEED="${RAND_SEED:-13}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT not found: $REPO_ROOT"
  exit 1
fi

cd "$REPO_ROOT"

echo "Phase 3 run id: $RUN_ID"
echo "Base run: $BASE_RUN_DIR"
echo "Output root: $PHASE3_ROOT"

python src/phase3_hidden12_cli.py \
  --base_run_dir "$BASE_RUN_DIR" \
  --out_root_dir "$PHASE3_ROOT" \
  --run_id "$RUN_ID" \
  --layer "$LAYER" \
  --variance_drop_percents "$VAR_DROPS" \
  --corr_thresholds "$CORR_THRESH" \
  --pca_components "$PCA_KS" \
  --random_drop_percents "$RAND_DROPS" \
  --random_trials "$RAND_TRIALS" \
  --random_seed "$RAND_SEED"

PHASE3_RUN_DIR="$PHASE3_ROOT/$RUN_ID"

python src/phase3_full_eval_cli.py \
  --screening_csv "$PHASE3_RUN_DIR/phase3_screening_scoreboard.csv" \
  --top_n "$TOP_N" \
  --layer "$LAYER" \
  --output_csv "$PHASE3_RUN_DIR/phase3_full_eval_scoreboard.csv"

echo "Phase 3 done."
echo "Outputs: $PHASE3_RUN_DIR"
