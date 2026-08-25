#!/bin/bash
#SBATCH --job-name=ig_bulk_pilot
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/ig_bulk_pilot_%j.out
#SBATCH --error=slurm/logs/ig_bulk_pilot_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL
#
# Issue #84 -- 50-pair pilot on LaTa, to measure s/pair and KB/NPZ before the
# chain is sized and submitted. Nothing is registered into
# phase12f_examples.csv: the pilot's 50 artifacts are real and are picked up by
# the LaTa chunk's own registry pass, so registering here would only add a
# rollback to undo.
#
#   sbatch --export=ALL,CODE_ROOT=/path/to/checkout \
#          slurm/ig/bulk_attribution_pilot.sbatch
#
# 20 minutes is a deliberate over-reservation for a 50-pair job (~0.33 GPU-h
# charged). It has to cover a cold model download, the O(vocab) token-keep
# lookup, the train-corpus SIF estimate and two D sweeps, none of which the
# per-pair rate should include.

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:?set CODE_ROOT to the checkout holding scripts/ig/run_bulk_attribution.py}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
BULK_MODEL="${BULK_MODEL:-bowphs/LaTa}"
PILOT_PAIRS="${PILOT_PAIRS:-50}"

cd "$CODE_ROOT"
mkdir -p slurm/logs

# shellcheck source=scripts/common/gpu_budget_guard.sh
source scripts/common/gpu_budget_guard.sh
budget_log "beto-delta-gpu" "at pilot start"
if budget_floor_reached "beto-delta-gpu" "${BUDGET_FLOOR:-5.0}"; then
    echo "BUDGET FLOOR REACHED -- skipping pilot"
    exit 0
fi

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"
IG_ROOT="$DATA_ROOT/runs/active/ig_examples"

$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== ${PILOT_PAIRS}-pair pilot: ${BULK_MODEL} ($(date)) ==="
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
    --limit "$PILOT_PAIRS" \
    --report_truncation \
    --trust_remote_code

echo ""
budget_log "beto-delta-gpu" "at pilot end"
echo "=== Done ($(date)) ==="
