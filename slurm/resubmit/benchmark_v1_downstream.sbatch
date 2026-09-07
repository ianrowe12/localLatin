#!/bin/bash
#SBATCH --job-name=bv1_downstream
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/benchmark_v1_downstream_%j.out
#SBATCH --error=slurm/logs/benchmark_v1_downstream_%j.err

# Everything that needs both the Task A results and the Task B seed sweep:
# layer geometry diagnostics, the per-layer appendix tables, the unlabelled
# predictions and the query-query matrices the webapp serves (issue #113).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
RUNS_ROOT="$REPO_ROOT/runs/active/resubmit_bases"
RESULTS_CSV="$REPO_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv"
MSEED_CSV="$REPO_ROOT/runs/active/resubmit/taskb_mseed/aggregated_results.csv"
DIAG_DIR="$REPO_ROOT/runs/active/resubmit/layer_diagnostics"
UNLAB_DIR="$REPO_ROOT/runs/active/resubmit/unlabelled"
TABLE_DIR="$REPO_ROOT/overleaf_drafts/tables"

export PYTHONDONTWRITEBYTECODE=1

echo "=== Alignment gate ==="
$PY scripts/resubmit/verify_embedding_alignment.py \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT/phase9_bases" | tail -5

echo "=== Layer geometry diagnostics ==="
$PY scripts/resubmit/run_layer_geometry_diagnostics.py \
  --models all \
  --split_csv "$SPLIT_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_dir "$DIAG_DIR"

echo "=== Layer geometry evidence join ==="
$PY scripts/resubmit/build_layer_geometry_evidence.py \
  --diagnostics_csv "$DIAG_DIR/geometry_per_layer.csv" \
  --retrieval_csv "$RESULTS_CSV" \
  --taskb_csv "$MSEED_CSV" \
  --out_dir "$DIAG_DIR"

echo "=== Per-layer appendix tables ==="
$PY scripts/resubmit/build_per_layer_tables.py \
  --taskA_csv "$RESULTS_CSV" \
  --taskB_csv "$MSEED_CSV" \
  --out_dir_audit "$REPO_ROOT/runs/active/resubmit/results/perlayer_tables" \
  --out_tex_dir "$TABLE_DIR"

echo "=== Unlabelled predictions (4 variants) ==="
for VARIANT in raw sif abtt sif_abtt; do
  echo "--- variant $VARIANT"
  $PY scripts/resubmit/run_resubmit_unlabelled_retrieval.py \
    --split_csv "$SPLIT_CSV" \
    --unlabelled_meta "$REPO_ROOT/runs/active/resubmit/data/unlabelled_meta.csv" \
    --results_csv "$RESULTS_CSV" \
    --labelled_bases "$RUNS_ROOT/phase9_bases" \
    --unlabelled_bases "$UNLAB_DIR/bases" \
    --out_dir "$UNLAB_DIR" \
    --data_root "$REPO_ROOT/data" \
    --variant "$VARIANT"
done

echo "=== Query-query matrices ==="
$PY scripts/resubmit/build_qq_matrices.py \
  --split_csv "$SPLIT_CSV" \
  --unlabelled_meta "$REPO_ROOT/runs/active/resubmit/data/unlabelled_meta.csv" \
  --results_csv "$RESULTS_CSV" \
  --labelled_bases "$RUNS_ROOT/phase9_bases" \
  --unlabelled_bases "$UNLAB_DIR/bases" \
  --data_root "$REPO_ROOT/data" \
  --out_dir "$UNLAB_DIR"

echo "=== All done ==="
