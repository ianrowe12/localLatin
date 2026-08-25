#!/bin/bash
#SBATCH --job-name=ig_bulk_pilot_fam
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/ig_bulk_pilot_fam_%j.out
#SBATCH --error=slurm/logs/ig_bulk_pilot_fam_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL
#
# Issue #84 -- rate probe for the four model families the LaTa pilot does not
# cover. Sizing every chunk off LaTa's 2.3 s/pair would be wrong by a large
# factor: Qwen3-0.6B and KaLM-mini are several times LaTa's size, and integrated
# gradients runs a full forward and an embedding-layer backward per step no
# matter which layer the target reads, so the cost tracks the model, not the
# layer. With the projected total already above the balance, guessing that
# factor decides which chunks get submitted at all.
#
# 20 pairs each. The artifacts are real and resume-safe, so the chunks pick them
# up rather than redoing them. Nothing is registered.

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:?set CODE_ROOT}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
PILOT_PAIRS="${PILOT_PAIRS:-20}"

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

for MODEL in \
    "google/mt5-base" \
    "sentence-transformers/LaBSE" \
    "Qwen/Qwen3-Embedding-0.6B" \
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
do
    echo ""
    echo "############ ${MODEL} ($(date)) ############"
    $PY -u scripts/ig/run_bulk_attribution.py \
        --model "$MODEL" \
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
        --trust_remote_code || echo "[WARN] ${MODEL} probe failed; continuing"
done

echo ""
budget_log "beto-delta-gpu" "at pilot end"
echo "=== Done ($(date)) ==="
