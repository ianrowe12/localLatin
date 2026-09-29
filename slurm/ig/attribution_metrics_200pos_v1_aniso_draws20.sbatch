#!/bin/bash
#SBATCH --job-name=attr_metrics_aniso_d20
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/attr_metrics_aniso_d20_%j.out
#SBATCH --error=slurm/logs/attr_metrics_aniso_d20_%j.err

# Issue #227. The metrics stage of slurm/ig/attribution_metrics_200pos_v1_draws20.sbatch,
# command for command, over the anisotropic-layer artifacts written by
# slurm/ig/run_attribution_200pos_v1_aniso.sbatch: hidden backend, token filter
# tokenizer_empty, 20 random-order draws, same seed stream. The hidden backend
# reads the layer, the per-token hidden states and the PCs from each NPZ, so
# nothing here names a layer. CPU only. The v1 draws20 pass (22098904) took
# 6:33, so 30 minutes is a generous cap.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
CODE_ROOT="${CODE_ROOT:-$REPO_ROOT}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts/ig:${PYTHONPATH:-}"

RUN="${RUN:-runs/active/ig_examples_200pos_v1_aniso}"
METRICS="$RUN/attribution_metrics_draws20"
TOKEN_FILTER="${TOKEN_FILTER:-tokenizer_empty}"
DRAWS="${DRAWS:-20}"

case "$RUN" in
    *ig_examples_200pos_v1|*ig_examples_200pos_v1/)
        echo "FAIL: RUN is the v1 run of record; this job never writes there" >&2
        exit 1 ;;
esac
mkdir -p "$METRICS"

echo "code=$CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown))"
echo "=== Hidden-backend metrics over 600 pairs, $DRAWS random-order draws ($(date)) ==="
$PY "$CODE_ROOT/scripts/ig/run_attribution_metrics.py" --backend hidden \
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
