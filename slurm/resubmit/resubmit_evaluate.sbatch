#!/bin/bash
#SBATCH --job-name=resub_eval
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/resubmit_evaluate_%j.out
#SBATCH --error=slurm/logs/resubmit_evaluate_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/active/resubmit_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results"
DIST_DIR="$REPO_ROOT/runs/active/resubmit/distributions"

mkdir -p "$RESULTS_DIR" "$DIST_DIR"

echo "=== Phase 11 evaluation ==="
conda run -n localLatin python scripts/run_phase11_evaluate.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --models "bowphs/LaTa,bowphs/PhilTa,google/mt5-base" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --reprs "hidden" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty" \
  --D_values "1,2,3,5,7,10" \
  --sif_a 0.001 \
  --dist_dir "$DIST_DIR"

echo "=== Extra distributions ==="
conda run -n localLatin python scripts/run_phase11_extra_distributions.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --models "bowphs/LaTa,bowphs/PhilTa,sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,google/mt5-base" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty" \
  --dist_dir "$DIST_DIR"

echo "=== Done ==="
