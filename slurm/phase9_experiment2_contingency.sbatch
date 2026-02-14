#!/bin/bash
#SBATCH --job-name=phase9_exp2_cont
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export EXP2_DIR="$REPO_ROOT/runs/phase9/experiment2"

cd "$REPO_ROOT" || exit 1

# Install POT for Optimal Transport
conda run -n localLatin pip install pot

# ─────────────────────────────────────────────────────────
# Phase 9.5: Last-token pooling + Optimal Transport
# ─────────────────────────────────────────────────────────
echo "=== Phase 9.5: Contingency ==="
conda run -n localLatin python scripts/run_phase9_exp2_contingency.py \
    --data_dir "$EXP2_DIR" \
    --results_csv "$EXP2_DIR/results.csv" \
    --out_dir "$EXP2_DIR" \
    --trust_remote_code

echo "=== Phase 9.5 Contingency Complete ==="
