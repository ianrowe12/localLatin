#!/bin/bash
#SBATCH --job-name=resub_figs
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/resubmit_figures_%j.out
#SBATCH --error=slurm/logs/resubmit_figures_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

echo "=== Generating resubmit figures ==="
conda run -n localLatin python scripts/resubmit/visualize_resubmit.py \
  --results_csv "$REPO_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv" \
  --dist_dir "$REPO_ROOT/runs/active/resubmit/distributions" \
  --out_dir "$REPO_ROOT/overleaf_drafts/figures"

echo "=== Done ==="
