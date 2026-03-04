#!/bin/bash
#SBATCH --job-name=p12d_filtering
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=beto-delta-gpu
#SBATCH --output=slurm/logs/phase12d_filtering_%j.out
#SBATCH --error=slurm/logs/phase12d_filtering_%j.err

module load anaconda3_gpu
conda activate localLatin

cd /u/irowerojas/localLatin
mkdir -p runs/phase12d slurm/logs

echo "========================================="
echo "Phase 12d: Filtering Sanity Check"
echo "========================================="

# LaTa
echo ">>> LaTa"
python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name bowphs/LaTa --model_type t5 \
  --layers 1,4,8,12 --D 10

# PhilTa
echo ">>> PhilTa"
python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name bowphs/PhilTa --model_type t5 \
  --layers 1,6,9,12 --D 10

# LaBSE
echo ">>> LaBSE"
python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name sentence-transformers/LaBSE --model_type bert \
  --layers 1,6,9,11 --D 10

# Qwen
echo ">>> Qwen3-0.6B"
python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name Qwen/Qwen3-Embedding-0.6B --model_type decoder \
  --layers 1,4,14,28 --D 10 --trust_remote_code

# KaLM
echo ">>> KaLM-mini"
python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 \
  --model_type decoder \
  --layers 1,5,13,23 --D 10 --trust_remote_code

echo "========================================="
echo "All models complete!"
echo "========================================="

# Generate combined summary
python -c "
import pandas as pd
from pathlib import Path
import glob

dfs = []
for f in sorted(glob.glob('runs/phase12d/filtering_results_*.csv')):
    dfs.append(pd.read_csv(f))
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('runs/phase12d/filtering_results_all.csv', index=False)
    print(combined.to_string(index=False))
"
