#!/bin/bash
#SBATCH --job-name=phase9_exp2_qwen
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

# ── Step 1: Evaluate Qwen3-Embedding-0.6B (appends to existing results.csv) ──
echo "=== Step 1: Qwen3-Embedding-0.6B Evaluation ==="
conda run -n localLatin python scripts/run_phase9_exp2_evaluate.py \
    --models "Qwen/Qwen3-Embedding-0.6B" \
    --languages "english,french,sinhala,tamil,serbian" \
    --data_dir "$EXP2_DIR" \
    --out_dir "$EXP2_DIR" \
    --cache_dir "$EXP2_DIR/cache" \
    --layers "0-36" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001 \
    --batch_size 32 \
    --max_length 128 \
    --trust_remote_code

# ── Step 2: Contingency for Qwen (appends to existing contingency_results.csv) ──
echo "=== Step 2: Qwen3 Contingency ==="
conda run -n localLatin python -m pip install pot 2>/dev/null
conda run -n localLatin python scripts/run_phase9_exp2_contingency.py \
    --data_dir "$EXP2_DIR" \
    --results_csv "$EXP2_DIR/results.csv" \
    --out_dir "$EXP2_DIR" \
    --batch_size 32 \
    --max_length 128 \
    --sif_a 0.001 \
    --D_values "1,2,3,5,7,10" \
    --trust_remote_code

# ── Step 3: Regenerate visualizations with all 3 models ──
echo "=== Step 3: Regenerate Visualizations ==="
conda run -n localLatin python scripts/visualize_experiment2.py \
    --results_csv "$EXP2_DIR/results.csv" \
    --diagnostics_csv "$EXP2_DIR/diagnostics.csv" \
    --contingency_csv "$EXP2_DIR/contingency_results.csv" \
    --cache_dir "$EXP2_DIR/cache" \
    --out_dir "$EXP2_DIR/figures" \
    --dpi 300

echo "=== Phase 9 Experiment 2: Qwen3 Addition Complete ==="
