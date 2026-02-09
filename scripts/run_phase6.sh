#!/bin/bash
set -euo pipefail

DEFAULT_WORK="${WORK:-$HOME}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_WORK/localLatin}"
CANON_ROOT="${CANON_ROOT:-$DEFAULT_WORK/canon}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"

META_ROOT="${META_ROOT:-$RUNS_ROOT/ff1_lata_postact}"
META_CSV="${META_CSV:-$META_ROOT/meta.csv}"

PHASE6_ROOT="${PHASE6_ROOT:-$RUNS_ROOT/phase6_generalization}"
PHASE5_RUN_DIR="${PHASE5_RUN_DIR:-}"

LATA_MODEL="${LATA_MODEL:-bowphs/LaTa}"
PHILTA_MODEL="${PHILTA_MODEL:-bowphs/PhilTa}"

LAYERS_HIDDEN="${LAYERS_HIDDEN:-0,1,12}"
LAYERS_FF1="${LAYERS_FF1:-1,12}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-12}"

BASE_HIDDEN_LATA="${BASE_HIDDEN_LATA:-$RUNS_ROOT/phase6_bases/lata_hidden_mean}"
BASE_FF1_LATA="${BASE_FF1_LATA:-$RUNS_ROOT/phase6_bases/lata_ff1_mean}"
BASE_HIDDEN_PHILTA="${BASE_HIDDEN_PHILTA:-$RUNS_ROOT/phase6_bases/philta_hidden_mean}"
BASE_FF1_PHILTA="${BASE_FF1_PHILTA:-$RUNS_ROOT/phase6_bases/philta_ff1_mean}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT not found: $REPO_ROOT"
  exit 1
fi

cd "$REPO_ROOT"

if [ -z "$PHASE5_RUN_DIR" ] || [ ! -d "$PHASE5_RUN_DIR" ]; then
  echo "ERROR: PHASE5_RUN_DIR must point to a valid Phase 5 run dir."
  exit 1
fi

if [ ! -f "$META_CSV" ]; then
  echo "meta.csv missing, building at: $META_CSV"
  python src/index_canon_cli.py \
    --canon_root "$CANON_ROOT" \
    --runs_root "$META_ROOT" \
    --model_name "$LATA_MODEL" \
    --max_length "$MAX_LENGTH" \
    --batch_size 128
fi

ensure_hidden_embeddings() {
  local base_dir="$1"
  local model_name="$2"
  local layers="$3"
  local missing=0
  for layer in ${layers//,/ }; do
    if [ ! -f "$base_dir/hidden_layer${layer}_embeddings_norm.npy" ]; then
      missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then
    echo "Extracting hidden embeddings: $model_name -> $base_dir"
    python src/extract_hidden_cli.py \
      --meta_csv "$META_CSV" \
      --runs_root "$RUNS_ROOT" \
      --run_dir "$base_dir" \
      --model_name "$model_name" \
      --layers "$layers" \
      --pooling mean \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE"
  fi
  return 0
}

ensure_ff1_embeddings() {
  local base_dir="$1"
  local model_name="$2"
  local layers="$3"
  local missing=0
  for layer in ${layers//,/ }; do
    if [ ! -f "$base_dir/ff1_layer${layer}_embeddings_norm.npy" ]; then
      missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then
    echo "Extracting FF1 embeddings: $model_name -> $base_dir"
    python src/extract_ff1_cli.py \
      --meta_csv "$META_CSV" \
      --runs_root "$RUNS_ROOT" \
      --run_dir "$base_dir" \
      --model_name "$model_name" \
      --layers "$layers" \
      --pooling mean \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE"
  fi
  return 0
}

ensure_hidden_embeddings "$BASE_HIDDEN_LATA" "$LATA_MODEL" "$LAYERS_HIDDEN"
ensure_ff1_embeddings "$BASE_FF1_LATA" "$LATA_MODEL" "$LAYERS_FF1"
ensure_hidden_embeddings "$BASE_HIDDEN_PHILTA" "$PHILTA_MODEL" "$LAYERS_HIDDEN"
if ! ensure_ff1_embeddings "$BASE_FF1_PHILTA" "$PHILTA_MODEL" "$LAYERS_FF1"; then
  echo "Warning: PhilTa FF1 extraction failed; Phase 6 will skip missing embeddings."
fi

python src/phase6_generalize_cli.py \
  --phase5_run_dir "$PHASE5_RUN_DIR" \
  --out_root_dir "$PHASE6_ROOT" \
  --target "$LATA_MODEL|hidden|mean|$LAYERS_HIDDEN|$BASE_HIDDEN_LATA" \
  --target "$LATA_MODEL|ff1|mean|$LAYERS_FF1|$BASE_FF1_LATA" \
  --target "$PHILTA_MODEL|hidden|mean|$LAYERS_HIDDEN|$BASE_HIDDEN_PHILTA" \
  --target "$PHILTA_MODEL|ff1|mean|$LAYERS_FF1|$BASE_FF1_PHILTA"

echo "Phase 6 done. Outputs under: $PHASE6_ROOT"
