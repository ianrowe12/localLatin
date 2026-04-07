#!/bin/bash
#SBATCH --job-name=phase11
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export RUNS_ROOT="$REPO_ROOT/runs"
export SPLIT_CSV="$RUNS_ROOT/phase9/phase9_split.csv"

cd "$REPO_ROOT" || exit 1

# =============================================================
# Part A: Evaluate (all 5 models, layers >= 1)
# =============================================================

echo "=== Part A: Phase 11 Evaluation ==="
conda run -n localLatin python scripts/run_phase11_evaluate.py \
    --split_csv "$SPLIT_CSV" \
    --runs_root "$RUNS_ROOT" \
    --out_csv "$RUNS_ROOT/phase11/phase11_results.csv" \
    --models "bowphs/LaTa,bowphs/PhilTa" \
    --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001 \
    --dist_dir "$RUNS_ROOT/phase11/distributions"

# =============================================================
# Part A.5: Extra Distributions
# =============================================================

echo "=== Part A.5: Extra Distributions ==="
conda run -n localLatin python scripts/run_phase11_extra_distributions.py \
    --results_csv "$RUNS_ROOT/phase11/phase11_results.csv" \
    --split_csv "$SPLIT_CSV" \
    --runs_root "$RUNS_ROOT" \
    --dist_dir "$RUNS_ROOT/phase11/distributions"

# =============================================================
# Part B: Visualize
# =============================================================

echo "=== Part B: Phase 11 Visualization ==="
conda run -n localLatin python scripts/visualize_phase11.py \
    --results_csv "$RUNS_ROOT/phase11/phase11_results.csv" \
    --dist_dir "$RUNS_ROOT/phase11/distributions" \
    --out_dir "$RUNS_ROOT/phase11/figures"

echo "=== Phase 11 Complete ==="
