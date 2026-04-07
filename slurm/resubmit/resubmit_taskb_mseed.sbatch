#!/bin/bash
#SBATCH --job-name=resub_taskb_mseed
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/resubmit_taskb_mseed_%j.out
#SBATCH --error=slurm/logs/resubmit_taskb_mseed_%j.err

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/active/resubmit_bases"
OUT_DIR="$REPO_ROOT/runs/active/resubmit/taskb_mseed"
FIG_DIR="$REPO_ROOT/overleaf_drafts/figures"

mkdir -p "$OUT_DIR" "$FIG_DIR"

echo "=== Task B M-seed evaluation (M=5) ==="
conda run -n localLatin python scripts/resubmit/run_taskb_mseed.py \
  --canon_root canon_labelled \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_dir "$OUT_DIR" \
  --M 5 --base_seed 42 \
  --models "bowphs/LaTa,bowphs/PhilTa,google/mt5-base" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --D_values "1,2,3,5,7,10" \
  --sif_a 0.001 \
  --top_k 5

echo "=== Verifying results ==="
if [ ! -f "$OUT_DIR/aggregated_results.csv" ]; then
  echo "ERROR: aggregated_results.csv was not created!" >&2
  exit 1
fi
echo "Aggregated results lines: $(wc -l < "$OUT_DIR/aggregated_results.csv")"
echo "All seeds results lines: $(wc -l < "$OUT_DIR/all_seeds_results.csv")"
echo "Seed directories: $(ls -d "$OUT_DIR"/seed_* | wc -l)"

echo "=== Generating Task B M-seed figures ==="
conda run -n localLatin python scripts/resubmit/visualize_taskb_mseed.py \
  --agg_csv "$OUT_DIR/aggregated_results.csv" \
  --out_dir "$FIG_DIR"

echo "=== Verifying figures ==="
for f in taskb_mseed_table.png taskb_mseed_table.tex taskb_mseed_topk_bar.png taskb_mseed_breakdown.png; do
  if [ ! -f "$FIG_DIR/$f" ]; then
    echo "WARNING: $f was not created" >&2
  else
    echo "OK: $f"
  fi
done

echo "=== All done ==="
