#!/bin/bash
#SBATCH --job-name=attr_200pair_methods
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/attr_200pair_methods_%j.out
#SBATCH --error=slurm/logs/attr_200pair_methods_%j.err

# Stage 1 (IG NPZs) + Stage 2 (MaRC) already ran in job 17773865.
# This sbatch fills the missing step (persist 6 other attribution methods into
# NPZs) and reruns the metrics computation so the resulting summary.csv/table
# actually contain IG/BertScore/OT/DLA/attention rows.

set -euo pipefail

module reset
module load miniforge3-python

cd /projects/beto/irowerojas/localLatin
mkdir -p slurm/logs

PY=/u/irowerojas/.conda/envs/localLatin/bin/python
export PYTHONPATH="/projects/beto/irowerojas/localLatin/src:${PYTHONPATH:-}"

EXAMPLES_CSV=runs/active/ig_examples_200pair/random200_examples.csv
ARTIFACTS_DIR=runs/active/ig_examples_200pair/artifacts
METRICS_OUT_DIR=runs/active/ig_examples_200pair/attribution_metrics
TEX_OUT=overleaf_drafts/tables/attribution_metrics_200pair.tex

echo "=== Stage 4/5: persist 6 attribution methods into NPZs ($(date)) ==="
$PY scripts/resubmit/persist_attribution_methods.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== Stage 5/5: rerun attribution-quality metrics ($(date)) ==="
$PY scripts/ig/run_attribution_metrics.py \
    --examples_csv   "$EXAMPLES_CSV" \
    --artifacts_root "$ARTIFACTS_DIR" \
    --out_root       "$METRICS_OUT_DIR" \
    --tex_out        "$TEX_OUT"

echo ""
echo "=== Done ($(date)) ==="
echo "Summary CSV: $METRICS_OUT_DIR/summary.csv"
echo "Paper table: $TEX_OUT"
