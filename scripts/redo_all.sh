#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"
OUT_DIR="$REPO_ROOT/runs/redo_results"
cd "$REPO_ROOT"

echo "=== Task 1: Data Preparation ==="
python scripts/redo_data_prep.py \
    --canon_root "$REPO_ROOT/canon" \
    --output_dir "$OUT_DIR" \
    --random_seed 42

echo ""
echo "=== Task 2: Experiment 1 (Canon) ==="
python scripts/redo_experiment1.py \
    --meta_csv "$OUT_DIR/meta_split_v2.csv" \
    --canon_root "$REPO_ROOT/canon" \
    --output "$OUT_DIR/experiment1_canon.csv" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001 \
    --batch_size 12 \
    --max_length 512

echo ""
echo "=== Task 3: Experiment 2 (MUSTS) ==="
python scripts/redo_experiment2.py \
    --models "Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,sentence-transformers/LaBSE" \
    --languages "english,french,sinhala,tamil" \
    --output "$OUT_DIR/experiment2_musts.csv" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001 \
    --batch_size 32 \
    --max_length 128 \
    --random_seed 42

echo ""
echo "=== Done. Results in $OUT_DIR ==="
