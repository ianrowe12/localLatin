#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
META_CSV="${META_CSV:-$RUNS_ROOT/phase9_bases/bowphs_LaTa/hidden_mean/meta.csv}"
SPLIT_CSV="${SPLIT_CSV:-$RUNS_ROOT/phase9/phase9_split.csv}"
PYTHON_BIN="${PYTHON_BIN:-conda run -n localLatin python}"

extract_t5() {
  local model_name="$1"
  local slug="$2"
  local layers="$3"

  if [ ! -f "$RUNS_ROOT/phase9_bases/$slug/hidden_mean_noempty/hidden_layer$(echo "$layers" | awk -F- '{print $2}')_embeddings.npy" ]; then
    $PYTHON_BIN src/extract_hidden_cli.py \
      --meta_csv "$META_CSV" \
      --runs_root "$RUNS_ROOT/phase9_bases" \
      --run_dir "$RUNS_ROOT/phase9_bases/$slug/hidden_mean_noempty" \
      --model_name "$model_name" \
      --layers "$layers" \
      --pooling mean \
      --token_filter no_empty \
      --max_length 512 \
      --batch_size 8
  fi

  if [ ! -f "$RUNS_ROOT/phase9_bases/$slug/hidden_sif_noempty/hidden_layer$(echo "$layers" | awk -F- '{print $2}')_embeddings_sif.npy" ]; then
    $PYTHON_BIN src/extract_hidden_cli.py \
      --meta_csv "$META_CSV" \
      --runs_root "$RUNS_ROOT/phase9_bases" \
      --run_dir "$RUNS_ROOT/phase9_bases/$slug/hidden_sif_noempty" \
      --model_name "$model_name" \
      --layers "$layers" \
      --pooling sif \
      --token_filter no_empty \
      --split_csv "$SPLIT_CSV" \
      --max_length 512 \
      --batch_size 8
  fi
}

extract_encoder() {
  local model_name="$1"
  local slug="$2"
  local layers="$3"
  local trust_flag="${4:-}"

  if [ ! -f "$RUNS_ROOT/phase9_bases/$slug/hidden_mean_noempty/hidden_layer$(echo "$layers" | awk -F- '{print $2}')_embeddings.npy" ]; then
    $PYTHON_BIN src/extract_encoder_cli.py \
      --meta_csv "$META_CSV" \
      --runs_root "$RUNS_ROOT/phase9_bases" \
      --run_dir "$RUNS_ROOT/phase9_bases/$slug/hidden_mean_noempty" \
      --model_name "$model_name" \
      --repr hidden \
      --layers "$layers" \
      --pooling mean \
      --token_filter no_empty \
      --max_length 512 \
      --batch_size 8 \
      $trust_flag
  fi

  if [ ! -f "$RUNS_ROOT/phase9_bases/$slug/hidden_sif_noempty/hidden_layer$(echo "$layers" | awk -F- '{print $2}')_embeddings_sif.npy" ]; then
    $PYTHON_BIN src/extract_encoder_cli.py \
      --meta_csv "$META_CSV" \
      --runs_root "$RUNS_ROOT/phase9_bases" \
      --run_dir "$RUNS_ROOT/phase9_bases/$slug/hidden_sif_noempty" \
      --model_name "$model_name" \
      --repr hidden \
      --layers "$layers" \
      --pooling sif \
      --token_filter no_empty \
      --split_csv "$SPLIT_CSV" \
      --max_length 512 \
      --batch_size 8 \
      $trust_flag
  fi
}

mkdir -p "$RUNS_ROOT/phase11_filtered/distributions" "$RUNS_ROOT/phase11_filtered/figures"

extract_t5 "bowphs/LaTa" "bowphs_LaTa" "1-12"
extract_t5 "bowphs/PhilTa" "bowphs_PhilTa" "1-12"
extract_t5 "google/mt5-base" "google_mt5-base" "1-12"
extract_encoder "sentence-transformers/LaBSE" "sentence-transformers_LaBSE" "1-12"
extract_encoder "Qwen/Qwen3-Embedding-0.6B" "Qwen_Qwen3-Embedding-0.6B" "1-28" "--trust_remote_code"
extract_encoder "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" "1-24" "--trust_remote_code"

$PYTHON_BIN scripts/run_phase11_evaluate.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_csv "$RUNS_ROOT/phase11_filtered/phase11_results.csv" \
  --models "bowphs/LaTa,bowphs/PhilTa,google/mt5-base" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --reprs "hidden" \
  --hidden_mean_subdir "hidden_mean_noempty" \
  --hidden_sif_subdir "hidden_sif_noempty" \
  --D_values "1,2,3,5,7,10" \
  --sif_a 0.001 \
  --dist_dir "$RUNS_ROOT/phase11_filtered/distributions"

$PYTHON_BIN scripts/run_phase11_extra_distributions.py \
  --results_csv "$RUNS_ROOT/phase11_filtered/phase11_results.csv" \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --models "bowphs/LaTa,bowphs/PhilTa,sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,google/mt5-base" \
  --hidden_mean_subdir "hidden_mean_noempty" \
  --hidden_sif_subdir "hidden_sif_noempty" \
  --dist_dir "$RUNS_ROOT/phase11_filtered/distributions"

$PYTHON_BIN scripts/visualize_phase11.py \
  --results_csv "$RUNS_ROOT/phase11_filtered/phase11_results.csv" \
  --dist_dir "$RUNS_ROOT/phase11_filtered/distributions" \
  --out_dir "$RUNS_ROOT/phase11_filtered/figures"
