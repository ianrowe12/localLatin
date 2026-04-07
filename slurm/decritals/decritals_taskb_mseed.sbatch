#!/bin/bash
#SBATCH --job-name=decr_taskb_mseed
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/decritals_taskb_mseed_%j.out
#SBATCH --error=slurm/logs/decritals_taskb_mseed_%j.err

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/active/decritals/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/active/decritals_bases"
OUT_DIR="$REPO_ROOT/runs/active/decritals/results/decritals_taskB_mseed"

mkdir -p "$OUT_DIR"

echo "=== Decritals Task B M-seed evaluation (M=10) ==="
conda run -n localLatin python scripts/resubmit/run_taskb_mseed.py \
  --canon_root data/decritals_labelled \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_dir "$OUT_DIR" \
  --M 10 --base_seed 42 \
  --models "bowphs/LaTa,bowphs/PhilTa,google/mt5-base" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --D_values "1,2,3,5,7,10" \
  --sif_a 0.001 \
  --top_k 5

echo "=== Verifying Task B results ==="
if [ ! -f "$OUT_DIR/aggregated_results.csv" ]; then
  echo "ERROR: aggregated_results.csv was not created!" >&2
  exit 1
fi
echo "Aggregated results lines: $(wc -l < "$OUT_DIR/aggregated_results.csv")"
echo "All seeds results lines: $(wc -l < "$OUT_DIR/all_seeds_results.csv")"
echo "Seed directories: $(ls -d "$OUT_DIR"/seed_* 2>/dev/null | wc -l)"

echo "=== Done ==="
