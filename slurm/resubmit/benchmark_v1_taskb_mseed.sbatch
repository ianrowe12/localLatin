#!/bin/bash
#SBATCH --job-name=bv1_mseed
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/benchmark_v1_mseed_%j.out
#SBATCH --error=slurm/logs/benchmark_v1_mseed_%j.err

# Task B five-seed evaluation for the benchmark v1 re-run (issue #113).
# Reads cached embeddings only, so CPU.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/active/resubmit_bases"
OUT_DIR="$REPO_ROOT/runs/active/resubmit/taskb_mseed"
FIG_DIR="$REPO_ROOT/overleaf_drafts/figures"

mkdir -p "$OUT_DIR" "$FIG_DIR"

echo "=== Task B M-seed evaluation (M=5) ==="
$PY scripts/resubmit/run_taskb_mseed.py \
  --canon_root data/canon_labelled \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_dir "$OUT_DIR" \
  --M 5 --base_seed 42 \
  --models "bowphs/LaTa,bowphs/PhilTa,google/mt5-base" \
  --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty" \
  --D_values "1,2,3,5,7,10" \
  --sif_a 0.001 \
  --top_k 5

test -f "$OUT_DIR/aggregated_results.csv"
echo "Aggregated results lines: $(wc -l < "$OUT_DIR/aggregated_results.csv")"

echo "=== Task B M-seed figures ==="
$PY scripts/resubmit/visualize_taskb_mseed.py \
  --agg_csv "$OUT_DIR/aggregated_results.csv" \
  --out_dir "$FIG_DIR" \
  --paper_table_tex "$REPO_ROOT/overleaf_drafts/tables/taskB_topk.tex"

echo "=== All done ==="
