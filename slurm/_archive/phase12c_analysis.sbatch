#!/bin/bash
#SBATCH --job-name=ph12c_analysis
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH -o slurm/logs/phase12c_analysis_%j.out
#SBATCH -e slurm/logs/phase12c_analysis_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12c: Analysis ==="
conda run -n localLatin python scripts/run_phase12c_analysis.py \
    --attr_dir "$REPO_ROOT/runs/phase12c/attributions" \
    --fig_dir "$REPO_ROOT/runs/phase12c/figures"

echo "=== Done ==="
