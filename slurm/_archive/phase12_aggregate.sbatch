#!/bin/bash
#SBATCH --job-name=ph12_agg
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12: Aggregate & Visualize ==="
conda run -n localLatin python scripts/run_phase12_aggregate.py \
    --attr_dir "$REPO_ROOT/runs/phase12/attributions" \
    --results_csv "$REPO_ROOT/runs/phase11/phase11_results.csv" \
    --out_dir "$REPO_ROOT/runs/phase12/figures"

echo "=== Done ==="
