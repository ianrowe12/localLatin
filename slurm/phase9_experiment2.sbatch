#!/bin/bash
#SBATCH --job-name=phase9_exp2
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export EXP2_DIR="$REPO_ROOT/runs/phase9/experiment2"

cd "$REPO_ROOT" || exit 1

# ─────────────────────────────────────────────────────────
# Step 1: Data prep — 50/50 split for 5 languages
# ─────────────────────────────────────────────────────────
echo "=== Step 1: Data Prep ==="
conda run -n localLatin python scripts/run_phase9_exp2_data_prep.py \
    --out_dir "$EXP2_DIR" \
    --seed 42

# ─────────────────────────────────────────────────────────
# Step 2: Evaluate LaBSE (~12 layers, encoder, ~1h)
# ─────────────────────────────────────────────────────────
echo "=== Step 2: LaBSE ==="
conda run -n localLatin python scripts/run_phase9_exp2_evaluate.py \
    --data_dir "$EXP2_DIR" \
    --out_dir "$EXP2_DIR" \
    --models "sentence-transformers/LaBSE" \
    --languages "english,french,sinhala,tamil,serbian"

# ─────────────────────────────────────────────────────────
# Step 3: Evaluate Qwen2.5-Embedding-0.5B (~24 layers, ~2h)
# ─────────────────────────────────────────────────────────
echo "=== Step 3: Qwen2.5-Embedding-0.5B ==="
conda run -n localLatin python scripts/run_phase9_exp2_evaluate.py \
    --data_dir "$EXP2_DIR" \
    --out_dir "$EXP2_DIR" \
    --models "Qwen/Qwen2.5-Embedding-0.5B" \
    --languages "english,french,sinhala,tamil,serbian" \
    --trust_remote_code

# ─────────────────────────────────────────────────────────
# Step 4: Evaluate KaLM-mini (~18 layers, ~2h)
# ─────────────────────────────────────────────────────────
echo "=== Step 4: KaLM-mini ==="
conda run -n localLatin python scripts/run_phase9_exp2_evaluate.py \
    --data_dir "$EXP2_DIR" \
    --out_dir "$EXP2_DIR" \
    --models "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --languages "english,french,sinhala,tamil,serbian" \
    --trust_remote_code

# ─────────────────────────────────────────────────────────
# Step 5: ROUGE/BLEU diagnostics
# ─────────────────────────────────────────────────────────
echo "=== Step 5: Diagnostics ==="
conda run -n localLatin python scripts/run_phase9_exp2_diagnostics.py \
    --data_dir "$EXP2_DIR" \
    --out_dir "$EXP2_DIR"

# ─────────────────────────────────────────────────────────
# Step 6: Visualization
# ─────────────────────────────────────────────────────────
echo "=== Step 6: Visualization ==="
conda run -n localLatin python scripts/run_phase9_exp2_visualize.py \
    --results_csv "$EXP2_DIR/results.csv" \
    --diagnostics_csv "$EXP2_DIR/diagnostics.csv" \
    --cache_dir "$EXP2_DIR/cache" \
    --out_dir "$EXP2_DIR/figures"

echo "=== Phase 9 Experiment 2 Complete ==="
