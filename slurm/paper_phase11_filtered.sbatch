#!/bin/bash
#SBATCH --job-name=paper_p11f
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/paper_phase11_filtered_%j.out
#SBATCH --error=slurm/logs/paper_phase11_filtered_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export RUNS_ROOT="$REPO_ROOT/runs"
mkdir -p "$REPO_ROOT/slurm/logs"
cd "$REPO_ROOT" || exit 1

bash scripts/run_phase11_filtered_pipeline.sh
