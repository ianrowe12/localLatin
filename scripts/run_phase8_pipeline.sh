#!/bin/bash
# Phase 8: Scientific Validation Pipeline
# Master orchestration script. Run stages sequentially.
#
# Usage:
#   bash scripts/run_phase8_pipeline.sh [stage]
#
# Stages: split, canon, musts_labse, musts_qwen, musts_gemma, viz, all

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"
CANON_ROOT="${CANON_ROOT:-$REPO_ROOT/canon}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
OUT_DIR="${OUT_DIR:-$RUNS_ROOT/phase8_results}"
SPLIT_CSV="$OUT_DIR/meta_split.csv"

STAGE="${1:-all}"

cd "$REPO_ROOT"

# --- Part 0: Create and lock split ---
run_split() {
    echo "=== Part 0: Creating canon split ==="
    python scripts/create_canon_split.py \
        --canon_root "$CANON_ROOT" \
        --output_csv "$SPLIT_CSV" \
        --test_fraction 0.2 \
        --random_seed 42
    echo "Split locked at: $SPLIT_CSV"
}

# --- Part 1: Canon re-validation (LaTa, PhilTa, LaBSE) ---
run_canon() {
    echo "=== Part 1: Canon sweep ==="
    python scripts/run_phase8_canon_sweep.py \
        --split_csv "$SPLIT_CSV" \
        --runs_root "$RUNS_ROOT" \
        --out_dir "$OUT_DIR" \
        --models "bowphs/LaTa,bowphs/PhilTa" \
        --reprs "hidden,ff1" \
        --layers_hidden "0-12" \
        --layers_ff1 "1-12" \
        --D 10 \
        --sif_a 0.001 \
        --encoder_models "sentence-transformers/LaBSE" \
        --encoder_layers "0-12"
    echo "Canon sweep done."
}

# --- Part 2: MUSTS layer-wise sweeps ---
run_musts_labse() {
    echo "=== Part 2a: MUSTS LaBSE ==="
    python scripts/run_phase8_musts_sweep.py \
        --models "sentence-transformers/LaBSE" \
        --languages "english,french,sinhala,tamil" \
        --layers "0-12" \
        --D_values "1,2,3,5,7,10" \
        --sif_a 0.001 \
        --out_dir "$OUT_DIR" \
        --batch_size 32 \
        --max_length 128
    echo "MUSTS LaBSE done."
}

run_musts_qwen() {
    echo "=== Part 2b: MUSTS Qwen2-7B ==="
    python scripts/run_phase8_musts_sweep.py \
        --models "Qwen/Qwen2-7B" \
        --languages "english,french,sinhala,tamil" \
        --layers "0-28" \
        --D_values "1,2,3,5,7,10" \
        --sif_a 0.001 \
        --out_dir "$OUT_DIR" \
        --batch_size 8 \
        --max_length 128 \
        --half_precision \
        --trust_remote_code
    echo "MUSTS Qwen2-7B done."
}

run_musts_gemma() {
    echo "=== Part 2c: MUSTS Gemma-7B ==="
    python scripts/run_phase8_musts_sweep.py \
        --models "google/gemma-7b" \
        --languages "english,french,sinhala,tamil" \
        --layers "0-28" \
        --D_values "1,2,3,5,7,10" \
        --sif_a 0.001 \
        --out_dir "$OUT_DIR" \
        --batch_size 8 \
        --max_length 128 \
        --half_precision
    echo "MUSTS Gemma-7B done."
}

# --- Part 3: Visualize ---
run_viz() {
    echo "=== Part 3: Visualization ==="
    python scripts/visualize_phase8.py \
        --musts_csv "$OUT_DIR/phase8_musts_sweep.csv" \
        --canon_csv "$OUT_DIR/phase8_canon_sweep.csv" \
        --out_dir "$OUT_DIR/figures" \
        --dpi 300
    echo "Visualization done."
}

# --- Dispatch ---
case "$STAGE" in
    split)       run_split ;;
    canon)       run_canon ;;
    musts_labse) run_musts_labse ;;
    musts_qwen)  run_musts_qwen ;;
    musts_gemma) run_musts_gemma ;;
    viz)         run_viz ;;
    all)
        run_split
        run_canon
        run_musts_labse
        run_musts_qwen
        run_musts_gemma
        run_viz
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 {split|canon|musts_labse|musts_qwen|musts_gemma|viz|all}"
        exit 1
        ;;
esac

echo "Phase 8 pipeline stage '$STAGE' complete."
