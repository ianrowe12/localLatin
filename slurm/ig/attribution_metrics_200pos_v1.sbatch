#!/bin/bash
#SBATCH --job-name=attr_metrics_v1
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/attr_metrics_v1_%j.out
#SBATCH --error=slurm/logs/attr_metrics_v1_%j.err

# Issue #141. Attribution-quality metrics for the benchmark v1 re-sample.
# CPU only: --backend hidden recomputes every masked cosine from the hidden
# states already cached in each NPZ, so it needs neither a GPU nor a model.
# The A8 operator spot check at the end is the one place a model is loaded,
# and it is 20 pairs of one model on CPU.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/ig:${PYTHONPATH:-}"

RUN="${RUN:-runs/active/ig_examples_200pos_v1}"
METRICS="$RUN/attribution_metrics"
SPOT="$METRICS/operator_spot_check"
# Must equal the artifact generator's --token_filter for this run. The
# generator filters the pooling but stores unfiltered hidden states, so a
# consumer that means over every attended row rebuilds a different vector than
# the PCs in the NPZ were fit on.
TOKEN_FILTER="${TOKEN_FILTER:-tokenizer_empty}"
mkdir -p "$METRICS" "$SPOT"

echo "=== Hidden-backend metrics over 600 pairs ($(date)) ==="
$PY scripts/ig/run_attribution_metrics.py --backend hidden \
    --token_filter "$TOKEN_FILTER" \
    --examples_csv "$RUN/positive200_examples.csv" \
    --artifacts_root "$RUN/artifacts" \
    --out_root "$METRICS" \
    --out_subdir v2_hidden \
    --summary_out "$METRICS/summary_v2.csv" \
    --sweep_summary_out "$METRICS/summary_v2_sweep_long.csv" \
    --tex_out "" --require_artifacts

# A8 of docs/research/attribution_metrics_decision.md: does the sign of the
# baseline-versus-ABTT difference on the AUC columns survive a change of
# erasure operator? Bounded to one model, IG only, the first 20 pairs, run
# under both operators on the identical subset. These numbers are a check on
# the main table and must never be pooled into it.
for BACKEND in model hidden; do
    echo ""
    echo "=== A8 spot check, --backend $BACKEND ($(date)) ==="
    $PY scripts/ig/run_attribution_metrics.py --backend "$BACKEND" --device cpu \
        --token_filter "$TOKEN_FILTER" \
        --models bowphs/LaTa --methods ig --max_pairs_per_model 20 \
        --metrics insertion_auc,deletion_auc --skip_pseudo_baselines \
        --examples_csv "$RUN/positive200_examples.csv" \
        --artifacts_root "$RUN/artifacts" \
        --out_root "$SPOT" --out_subdir "${BACKEND}20" \
        --summary_out "$SPOT/${BACKEND}20_summary.csv" \
        --sweep_summary_out "$SPOT/${BACKEND}20_sweep.csv" \
        --tex_out "" --require_artifacts
done

echo ""
echo "=== Done ($(date)) ==="
