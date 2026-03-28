#!/bin/bash
#SBATCH --job-name=ph12_prep
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12: Prepare PCs ==="
conda run -n localLatin python scripts/run_phase12_prepare_pcs.py \
    --split_csv "$REPO_ROOT/runs/phase9/phase9_split.csv" \
    --runs_root "$REPO_ROOT/runs" \
    --out_dir "$REPO_ROOT/runs/phase12/pcs"

echo "=== Phase 12: Select Samples ==="
conda run -n localLatin python scripts/run_phase12_select_samples.py \
    --split_csv "$REPO_ROOT/runs/phase9/phase9_split.csv" \
    --out_csv "$REPO_ROOT/runs/phase12/phase12_samples.csv" \
    --n_similar 30 \
    --n_dissimilar 30

echo "=== Done ==="
