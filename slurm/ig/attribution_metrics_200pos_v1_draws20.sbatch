#!/bin/bash
#SBATCH --job-name=attr_metrics_v1_d20
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/attr_metrics_v1_d20_%j.out
#SBATCH --error=slurm/logs/attr_metrics_v1_d20_%j.err

# Issue #206 (G10). The benchmark v1 attribution metrics again, with the
# chance-corrected deletion and insertion references averaged over 20 random
# orderings instead of 5, as A8 of docs/research/attribution_metrics_decision.md
# advised and docs/research/delauc_sensitivity.md measured (its `draws20` arm).
# Same artifacts, same token filter, same seed stream (RANDOM_ORDER_SEED): the
# first five orderings are the ones the 5-draw run used. Writes to a NEW
# directory; attribution_metrics/ (5 draws) is kept untouched for the record.
# CPU only: --backend hidden never loads a model. The 5-draw job (22072507)
# took 38 min including the A8 operator spot check, which is not repeated here
# because it is a check on the operator, not a table input.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/ig:${PYTHONPATH:-}"

RUN="${RUN:-runs/active/ig_examples_200pos_v1}"
METRICS="$RUN/attribution_metrics_draws20"
# Must equal the artifact generator's --token_filter for this run; the
# metrics stage checks it against the value stored in every NPZ.
TOKEN_FILTER="${TOKEN_FILTER:-tokenizer_empty}"
DRAWS="${DRAWS:-20}"
mkdir -p "$METRICS"

echo "=== Hidden-backend metrics over 600 pairs, $DRAWS random-order draws ($(date)) ==="
$PY scripts/ig/run_attribution_metrics.py --backend hidden \
    --token_filter "$TOKEN_FILTER" \
    --random_order_draws "$DRAWS" \
    --examples_csv "$RUN/positive200_examples.csv" \
    --artifacts_root "$RUN/artifacts" \
    --out_root "$METRICS" \
    --out_subdir v2_hidden \
    --summary_out "$METRICS/summary_v2.csv" \
    --sweep_summary_out "$METRICS/summary_v2_sweep_long.csv" \
    --tex_out "" --require_artifacts

echo ""
echo "=== Done ($(date)) ==="
