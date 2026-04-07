#!/bin/bash
#SBATCH --job-name=paper_p12prep
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_phase12_prepare_%j.out
#SBATCH --error=slurm/logs/paper_phase12_prepare_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/phase12_filtered/pcs"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/run_phase12_prepare_pcs.py \
  --split_csv "$REPO_ROOT/runs/phase9/phase9_split.csv" \
  --runs_root "$REPO_ROOT/runs" \
  --hidden_subdir "hidden_mean_noempty" \
  --out_dir "$REPO_ROOT/runs/phase12_filtered/pcs" \
  --models "bowphs/LaTa,bowphs/PhilTa,sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,google/mt5-base" \
  --fail_if_missing
