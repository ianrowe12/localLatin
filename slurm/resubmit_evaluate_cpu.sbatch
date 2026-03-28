#!/bin/bash
#SBATCH --job-name=resub_eval
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/resubmit_evaluate_%j.out
#SBATCH --error=slurm/logs/resubmit_evaluate_%j.err

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/phase_resubmit/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/phase_resubmit_bases"
RESULTS_DIR="$REPO_ROOT/runs/phase_resubmit/results"
DIST_DIR="$REPO_ROOT/runs/phase_resubmit/distributions"

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

echo "=== Verifying results CSV ==="
if [ ! -f "$RESULTS_DIR/phase_resubmit_results.csv" ]; then
  echo "ERROR: phase_resubmit_results.csv was not created!" >&2
  exit 1
fi
echo "Results CSV lines: $(wc -l < "$RESULTS_DIR/phase_resubmit_results.csv")"

echo "=== Extra distributions ==="
conda run -n localLatin python scripts/run_phase11_extra_distributions.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --models "bowphs/LaTa,bowphs/PhilTa,sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,google/mt5-base" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty" \
  --dist_dir "$DIST_DIR"

echo "=== Generating figures ==="
conda run -n localLatin python scripts/visualize_phase_resubmit.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --dist_dir "$DIST_DIR" \
  --out_dir "$REPO_ROOT/overleaf_drafts/figures"

echo "=== All done ==="
