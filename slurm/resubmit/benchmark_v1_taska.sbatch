#!/bin/bash
#SBATCH --job-name=bv1_taska
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/benchmark_v1_taska_%j.out
#SBATCH --error=slurm/logs/benchmark_v1_taska_%j.err

# Task A/B evaluation for the benchmark v1 re-run (issue #113).
# Embeddings are unchanged, so this is CPU work: it only reads the cached
# matrices and rescores them against the corrected split.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/active/resubmit_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results"
DIST_DIR="$REPO_ROOT/runs/active/resubmit/distributions"
FIG_DIR="$REPO_ROOT/overleaf_drafts/figures"
TABLE_DIR="$REPO_ROOT/overleaf_drafts/tables"

mkdir -p "$RESULTS_DIR" "$DIST_DIR" "$FIG_DIR" "$TABLE_DIR"

MODELS="bowphs/LaTa,bowphs/PhilTa,google/mt5-base"
ENCODER_MODELS="sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"

echo "=== Alignment gate ==="
$PY scripts/resubmit/verify_embedding_alignment.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT/phase9_bases" | tail -5

echo "=== Task A/B evaluation ==="
$PY scripts/resubmit/run_resubmit_evaluate.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --models "$MODELS" \
  --encoder_models "$ENCODER_MODELS" \
  --reprs "hidden" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty" \
  --D_values "1,2,3,5,7,10" \
  --sif_a 0.001 \
  --dist_dir "$DIST_DIR"

test -f "$RESULTS_DIR/phase_resubmit_results.csv"
echo "Results CSV lines: $(wc -l < "$RESULTS_DIR/phase_resubmit_results.csv")"

echo "=== Extra distributions ==="
$PY scripts/resubmit/run_resubmit_distributions.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --models "$MODELS,$ENCODER_MODELS" \
  --hidden_mean_subdir "hidden_mean_tokempty" \
  --hidden_sif_subdir "hidden_sif_tokempty" \
  --dist_dir "$DIST_DIR"

echo "=== Figures ==="
$PY scripts/resubmit/visualize_resubmit.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --dist_dir "$DIST_DIR" \
  --out_dir "$FIG_DIR"

echo "=== Headline tables ==="
$PY scripts/resubmit/build_headline_tables.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --out_dir "$TABLE_DIR"

echo "=== Task A per-model summary ==="
$PY scripts/resubmit/build_taskA_per_model_table.py \
  --results_csv "$RESULTS_DIR/phase_resubmit_results.csv" \
  --out_csv "$RESULTS_DIR/taskA_per_model_summary.csv" \
  --out_tex "$TABLE_DIR/taskA_per_model.tex"

echo "=== All done ==="
