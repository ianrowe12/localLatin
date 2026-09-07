#!/bin/bash
#SBATCH --job-name=bv1_clusterviz
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/benchmark_v1_cluster_viz_%j.out
#SBATCH --error=slurm/logs/benchmark_v1_cluster_viz_%j.err

# The four cluster-geometry figures the paper prints, plus the silhouette CSVs
# the prose quotes (issue #113). These stems had no committed producer: the
# committed cluster-viz job only ever wrote the fig_{proj}_per_model variants.
#
# --force_recompute is not optional here. The projection caches under
# cluster_viz/ hold coordinates in the pre-correction row order; reusing one
# against the corrected split would colour seventeen points from the wrong
# directories.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/active/resubmit_bases/phase9_bases"
BEST_LAYERS="$REPO_ROOT/runs/active/resubmit/results/taskA_per_model_summary.csv"
INTERMEDIATE="$REPO_ROOT/runs/active/resubmit/cluster_viz"
FIG_DIR="$REPO_ROOT/overleaf_drafts/figures"

echo "=== Alignment gate ==="
$PY scripts/resubmit/verify_embedding_alignment.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$BASES" | tail -5

run_group() {
  local group="$1"; shift
  local suffix="_${group}"
  echo "=== cluster geometry: $group ($*) ==="
  $PY scripts/resubmit/visualize_clusters_2d.py \
    --split_csv "$SPLIT_CSV" \
    --bases_root "$BASES" \
    --best_layer_csv "$BEST_LAYERS" \
    --intermediate_dir "$INTERMEDIATE" \
    --figures_dir "$FIG_DIR" \
    --methods baseline abtt_fixed \
    --projections tsne umap \
    --output_stem "fig_{proj}${suffix}" \
    --silhouette_csv_suffix "$group" \
    --force_recompute \
    --models "$@"
}

run_group "main" LaTa PhilTa mT5-base
run_group "appendix" LaBSE Qwen3-0.6B KaLM-mini

echo "=== All done ==="
