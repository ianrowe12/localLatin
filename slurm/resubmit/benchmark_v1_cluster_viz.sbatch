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
# the prose quotes (issue #113), plus the six-model fig_{proj}_per_model grids.
#
# Layers are train-selected (issue #184): the script derives them from the
# single-seed results CSV with the headline rule (abtt_optimal, argmax of
# train_dir_acc_at_1), so the panels sit at the ABTT subscript of Table 3.
# The previous input, taskA_per_model_summary.csv, was the test-set argmax
# across methods and put LaTa at layer 1 instead of 8.
#
# --force_recompute is not optional here. The projection caches under
# cluster_viz/ hold coordinates in the pre-correction row order; reusing one
# against the corrected split would colour seventeen points from the wrong
# directories.

set -euo pipefail

# Under SLURM BASH_SOURCE is the spooled copy of this script, so the dirname
# fallback lands in /var/spool; the submit directory is the checkout.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/active/resubmit_bases/phase9_bases"
RESULTS_CSV="$REPO_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv"
INTERMEDIATE="$REPO_ROOT/runs/active/resubmit/cluster_viz"
FIG_DIR="$REPO_ROOT/overleaf_drafts/figures"

echo "=== Alignment gate ==="
$PY scripts/resubmit/verify_embedding_alignment.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$BASES" | tail -5

# run_group <stem suffix> <silhouette suffix> <methods...> -- <models...>
run_group() {
  local stem_suffix="$1" sil_suffix="$2"; shift 2
  local methods=()
  while [ "$1" != "--" ]; do methods+=("$1"); shift; done
  shift
  echo "=== cluster geometry: fig_{proj}${stem_suffix} (${methods[*]}; $*) ==="
  $PY scripts/resubmit/visualize_clusters_2d.py \
    --split_csv "$SPLIT_CSV" \
    --bases_root "$BASES" \
    --results_csv "$RESULTS_CSV" \
    --layer_select_method abtt_optimal \
    --layer_select_metric train_dir_acc_at_1 \
    --intermediate_dir "$INTERMEDIATE" \
    --figures_dir "$FIG_DIR" \
    --methods "${methods[@]}" \
    --projections tsne umap \
    --top_k 6 \
    --output_stem "fig_{proj}${stem_suffix}" \
    --silhouette_csv_suffix "$sil_suffix" \
    --force_recompute \
    --models "$@"
}

run_group "_main" "main" baseline abtt_fixed -- LaTa PhilTa mT5-base
run_group "_appendix" "appendix" baseline abtt_fixed -- LaBSE Qwen3-0.6B KaLM-mini
run_group "_per_model" "" baseline sif_only sif_abtt_optimal whitening -- \
  LaTa PhilTa mT5-base LaBSE Qwen3-0.6B KaLM-mini

echo "=== All done ==="
