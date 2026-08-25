#!/bin/bash
#SBATCH --job-name=ig_bulk
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --output=slurm/logs/ig_bulk_%j.out
#SBATCH --error=slurm/logs/ig_bulk_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL
#
# Issue #84 -- one model's chunk of the full-corpus IG attribution run.
#
# --time is NOT set here. Every chunk has a different pair count and a different
# per-pair cost, and CLAUDE.md charges the reserved wallclock, not the elapsed
# one. The submitter sizes --time per chunk from the measured pilot rate and
# passes it on the sbatch command line, where it overrides these directives:
#
#   sbatch --time=HH:MM:00 --job-name=ig_bulk_lata \
#          --export=ALL,BULK_MODEL=bowphs/LaTa \
#          slurm/ig/bulk_attribution_chunk.sbatch
#
# Chain the chunks with --dependency=afterany so a chunk that stops early (or
# fails) does not strand the rest: the budget guard below, not the dependency
# graph, is what ends the chain.
#
# The guard runs BEFORE anything expensive. Below the floor it prints
# "BUDGET FLOOR REACHED" and exits 0, so the remaining chained chunks each
# no-op in a few seconds instead of erroring.

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:?set CODE_ROOT to the checkout holding scripts/ig/run_bulk_attribution.py}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
BULK_MODEL="${BULK_MODEL:?set BULK_MODEL to a HuggingFace model id}"
BUDGET_ACCOUNT="${BUDGET_ACCOUNT:-beto-delta-gpu}"
BUDGET_FLOOR="${BUDGET_FLOOR:-5.0}"
BULK_LIMIT="${BULK_LIMIT:-0}"
BULK_REGISTER="${BULK_REGISTER:-1}"

cd "$CODE_ROOT"
mkdir -p slurm/logs

# shellcheck source=scripts/common/gpu_budget_guard.sh
source scripts/common/gpu_budget_guard.sh

echo "=== ${BULK_MODEL} chunk, job ${SLURM_JOB_ID:-local} ==="
budget_log "$BUDGET_ACCOUNT" "at chunk start"
if budget_floor_reached "$BUDGET_ACCOUNT" "$BUDGET_FLOOR"; then
    echo "BUDGET FLOOR REACHED (< ${BUDGET_FLOOR} h on ${BUDGET_ACCOUNT}) -- skipping ${BULK_MODEL}"
    exit 0
fi

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"

IG_ROOT="$DATA_ROOT/runs/active/ig_examples"
EXAMPLES_CSV="$IG_ROOT/phase12f_examples.csv"

# Stop with time to spare so the registry is always written and the job never
# TIMEOUTs (a TIMEOUT still bills the full reservation and trips the watchdog).
# The generator measures this budget from PROCESS start, so setup is spent out
# of it and these 8 minutes are headroom for the registry pass alone. Do not
# re-read that as "setup + registry": setup can be ten minutes by itself on a
# large vocabulary, which would push a capped chunk past its wallclock.
if [[ -z "${BULK_MAX_SECONDS:-}" ]]; then
    limit_min="$(squeue -h -j "${SLURM_JOB_ID}" -o '%l' 2>/dev/null | awk -F: '
        NF==3 { split($1, d, "-"); h = (length(d)>1 ? d[2] : d[1]);
                days = (length(d)>1 ? d[1] : 0);
                print (days*1440) + (h*60) + $2; next }
        NF==2 { print $1 + 0; next }')"
    if [[ -n "${limit_min}" && "${limit_min}" -gt 10 ]]; then
        BULK_MAX_SECONDS=$(( (limit_min - 8) * 60 ))
    else
        BULK_MAX_SECONDS=0
    fi
fi
echo "max_seconds for the pair loop: ${BULK_MAX_SECONDS}"

$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== [1/2] Generate artifacts ($(date)) ==="
$PY -u scripts/ig/run_bulk_attribution.py \
    --model "$BULK_MODEL" \
    --data_root "$DATA_ROOT/data" \
    --unlabelled_root "$DATA_ROOT/runs/active/resubmit/unlabelled" \
    --labelled_bases "$DATA_ROOT/runs/active/resubmit_bases/phase9_bases" \
    --split_csv "$DATA_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv" \
    --artifacts_dir "$IG_ROOT/artifacts" \
    --pc_root "$IG_ROOT/bulk_pcs" \
    --registry_dir "$IG_ROOT/bulk_registry" \
    --token_probs_cache "$IG_ROOT/bulk_sif_token_probs" \
    --limit "$BULK_LIMIT" \
    --max_seconds "$BULK_MAX_SECONDS" \
    --report_truncation \
    --trust_remote_code

SLUG="${BULK_MODEL//\//_}"
if [[ "$BULK_REGISTER" == "1" ]]; then
    echo ""
    echo "=== [2/2] Register into phase12f_examples.csv ($(date)) ==="
    $PY -u scripts/ig/register_bulk_examples.py \
        --registry "$IG_ROOT/bulk_registry/${SLUG}.csv" \
        --examples_csv "$EXAMPLES_CSV"
else
    echo ""
    echo "=== [2/2] Registration skipped (BULK_REGISTER=0) ==="
fi

echo ""
budget_log "$BUDGET_ACCOUNT" "at chunk end"
echo "=== Done ($(date)) ==="
