#!/bin/bash
#SBATCH --job-name=phase12b_analysis
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/phase12b_analysis_%j.out
#SBATCH --error=slurm/logs/phase12b_analysis_%j.err

# Phase 12b: Embedding-level metrics + token attribution delta
# CPU-only work, but gpuA100x4 required by cluster policy.

set -euo pipefail
module load miniforge3-python

echo "=== Phase 12b: Embedding Analysis ==="
echo "Start: $(date)"
conda run -n localLatin python -u scripts/run_phase12b_embedding_analysis.py \
  --emb_base runs/phase9_bases \
  --pc_dir  runs/phase12/pcs \
  --split_csv runs/phase9/phase9_split.csv \
  --phase11_csv runs/phase11/phase11_results.csv \
  --out_dir runs/phase12b
echo "Embedding analysis done: $(date)"

echo ""
echo "=== Phase 12b: Token Attribution Delta ==="
conda run -n localLatin python -u scripts/run_phase12b_token_delta.py \
  --attr_dir runs/phase12/attributions \
  --phase11_csv runs/phase11/phase11_results.csv \
  --out_dir runs/phase12b
echo "Token delta analysis done: $(date)"

echo ""
echo "=== All Phase 12b analyses complete ==="
echo "End: $(date)"
