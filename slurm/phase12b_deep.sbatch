#!/bin/bash
#SBATCH --job-name=phase12b_deep
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/phase12b_deep_%j.out
#SBATCH --error=slurm/logs/phase12b_deep_%j.err

# Phase 12b Deep Analysis: 5 methods (SNR, ΔIG, Cosine Gap, Eff. Dim., Rank Shift)

set -euo pipefail
module load miniforge3-python

echo "=== Phase 12b Deep Analysis (5 methods) ==="
echo "Start: $(date)"
conda run -n localLatin python -u scripts/run_phase12b_deep_analysis.py \
  --emb_base runs/phase9_bases \
  --pc_dir  runs/phase12/pcs \
  --attr_dir runs/phase12/attributions \
  --split_csv runs/phase9/phase9_split.csv \
  --phase11_csv runs/phase11/phase11_results.csv \
  --out_dir runs/phase12b/deep
echo "End: $(date)"
