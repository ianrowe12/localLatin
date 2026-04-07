#!/bin/bash
set -euo pipefail

DEFAULT_WORK="${WORK:-$HOME}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_WORK/localLatin}"
CANON_ROOT="${CANON_ROOT:-$REPO_ROOT/canon}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs/ff1_lata_postact}"
MODEL_NAME="${MODEL_NAME:-bowphs/LaTa}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LAYERS_FF1="${LAYERS_FF1:-1-12}"
LAYERS_HIDDEN="${LAYERS_HIDDEN:-0-12}"
BUCKET_LAYERS_FF1="${BUCKET_LAYERS_FF1:-1-12}"
BUCKET_LAYERS_HIDDEN="${BUCKET_LAYERS_HIDDEN:-0-12}"
NUM_BUCKETS="${NUM_BUCKETS:-4}"

RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_ID}"

export HF_HOME="${HF_HOME:-$DEFAULT_WORK/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$DEFAULT_WORK/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DEFAULT_WORK/.cache/huggingface}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT not found: $REPO_ROOT"
  echo "Set REPO_ROOT explicitly or ensure WORK/HOME is correct."
  exit 1
fi

cd "$REPO_ROOT"

echo "Run dir: $RUN_DIR"

if command -v module >/dev/null 2>&1; then
  if [ -n "${MODULE_LOAD:-}" ]; then
    module reset
    module load ${MODULE_LOAD}
  fi
fi

if [ -n "${VENV_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if [ "${PIP_INSTALL_REQUIREMENTS:-0}" = "1" ]; then
  pip install -r "$REPO_ROOT/requirements.txt"
fi

python src/index_canon_cli.py \
  --canon_root "$CANON_ROOT" \
  --runs_root "$RUNS_ROOT" \
  --model_name "$MODEL_NAME" \
  --max_length "$MAX_LENGTH" \
  --batch_size 128

python src/extract_ff1_cli.py \
  --meta_csv "$RUNS_ROOT/meta.csv" \
  --runs_root "$RUNS_ROOT" \
  --run_dir "$RUN_DIR" \
  --model_name "$MODEL_NAME" \
  --layers "$LAYERS_FF1" \
  --pooling mean \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE"

python src/extract_ff1_cli.py \
  --meta_csv "$RUNS_ROOT/meta.csv" \
  --runs_root "$RUNS_ROOT" \
  --run_dir "$RUN_DIR" \
  --model_name "$MODEL_NAME" \
  --layers "$LAYERS_FF1" \
  --pooling lasttok \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE"

python src/extract_hidden_cli.py \
  --meta_csv "$RUNS_ROOT/meta.csv" \
  --runs_root "$RUNS_ROOT" \
  --run_dir "$RUN_DIR" \
  --model_name "$MODEL_NAME" \
  --layers "$LAYERS_HIDDEN" \
  --pooling mean \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE"

python src/extract_hidden_cli.py \
  --meta_csv "$RUNS_ROOT/meta.csv" \
  --runs_root "$RUNS_ROOT" \
  --run_dir "$RUN_DIR" \
  --model_name "$MODEL_NAME" \
  --layers "$LAYERS_HIDDEN" \
  --pooling lasttok \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE"

python src/eval_retrieval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr ff1 \
  --pooling mean \
  --layers "$LAYERS_FF1"

python src/eval_retrieval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr ff1 \
  --pooling lasttok \
  --layers "$LAYERS_FF1"

python src/eval_retrieval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr hidden \
  --pooling mean \
  --layers "$LAYERS_HIDDEN"

python src/eval_retrieval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr hidden \
  --pooling lasttok \
  --layers "$LAYERS_HIDDEN"

python src/bucket_eval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr ff1 \
  --pooling mean \
  --layers "$BUCKET_LAYERS_FF1" \
  --num_buckets "$NUM_BUCKETS"

python src/bucket_eval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr hidden \
  --pooling mean \
  --layers "$BUCKET_LAYERS_HIDDEN" \
  --num_buckets "$NUM_BUCKETS"

echo "Bucket summaries:"
echo "  - $RUN_DIR/bucket_summary_ff1_mean.csv"
echo "  - $RUN_DIR/bucket_summary_hidden_mean.csv"

echo "All done."
