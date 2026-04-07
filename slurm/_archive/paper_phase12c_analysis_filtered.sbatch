#!/bin/bash
#SBATCH --job-name=paper_p12c_ana
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/paper_phase12c_analysis_%j.out
#SBATCH --error=slurm/logs/paper_phase12c_analysis_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/phase12c_filtered/figures"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/run_phase12c_analysis.py \
  --attr_dir "$REPO_ROOT/runs/phase12c_filtered/attributions" \
  --fig_dir "$REPO_ROOT/runs/phase12c_filtered/figures" \
  --fail_if_empty
