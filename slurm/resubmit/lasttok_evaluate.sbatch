#!/bin/bash
#SBATCH --job-name=lasttok_evaluate
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/lasttok_evaluate_%j.out
#SBATCH --error=slurm/logs/lasttok_evaluate_%j.err

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="/projects/beto/irowerojas/worktrees/run-1-agent-3"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

META_CSV="$REPO_ROOT/runs/active/resubmit/data/meta.csv"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/active/resubmit_bases"
OUT_DIR="$REPO_ROOT/runs/active/resubmit/results/lasttok"
mkdir -p "$OUT_DIR"

echo "=== Eval: lasttok pooling, ABTT-only ==="
conda run -n localLatin python scripts/resubmit/run_resubmit_evaluate.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$BASES" \
  --out_csv "$OUT_DIR/phase_resubmit_results_lasttok.csv" \
  --models "bowphs/LaTa,bowphs/PhilTa" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,Qwen/Qwen3-Embedding-8B" \
  --reprs hidden \
  --hidden_lasttok_subdir "hidden_lasttok_tokempty" \
  --poolings lasttok \
  --methods abtt_fixed,abtt_optimal

echo "=== Eval: mean pooling, ABTT-only (comparable baseline) ==="
conda run -n localLatin python scripts/resubmit/run_resubmit_evaluate.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$BASES" \
  --out_csv "$OUT_DIR/phase_resubmit_results_mean_abttonly.csv" \
  --models "bowphs/LaTa,bowphs/PhilTa" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,Qwen/Qwen3-Embedding-8B" \
  --reprs hidden \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --poolings mean \
  --methods abtt_fixed,abtt_optimal

echo "=== Done ==="
