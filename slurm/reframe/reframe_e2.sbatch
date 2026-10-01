#!/bin/bash
#SBATCH --job-name=reframe_e2
#SBATCH --partition=gpuA40x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/reframe_e2_%j.out
#SBATCH --error=slurm/logs/reframe_e2_%j.err

# Issue #252 (E2): token audit of the dominant directions, six panel models.
# One GPU: two forward passes per model with all hidden states (the extraction CLIs'
# tokenization, model class and batch size), then the metric cells on the CPUs.
# Reads the cached mean-pooled (and, where present, SIF-pooled) vectors; writes small
# CSVs only: runs/active/reframe/e2/*.csv, facts_e2.md and
# overleaf_drafts/tables/e2_token_audit.tex under CODE_ROOT.
#
# Overridable through the environment (sbatch --export=ALL with the variables set):
#   CODE_ROOT        checkout holding the scripts and receiving the outputs
#                    (default: the submit directory)
#   BASES_ROOT       directory holding phase9_bases/<slug>/hidden_mean_tokempty/
#   PY               python launcher, may be several words, e.g.
#                    "conda run --no-capture-output -p <env prefix> python"
#   MODELS           '|'-separated display names or ids (default: all six);
#                    '|' because --export splits on commas. A run replaces only its
#                    models' rows in the CSVs, so the work can be split across jobs.
#   LIMIT            N = smoke run on the first N train and N test passages, written to
#                    runs/active/reframe/e2/smoke_limit<N>/ (table included), gates skipped
#   CONTROL_METRICS  auroc (default) or full: metric block of the two random controls
#   HF_CACHE         HuggingFace cache holding the six models (compute nodes are offline;
#                    default /work/nvme/bimc/swong2/hf_cache, where they were downloaded)
#   TOL_AUROC        AUROC tolerance of gates 1a and 2a (default 1e-6)
#
# Time and memory (A40, 8 CPUs; the first full run, jobs 22597913 and 22598253, took
# 10.5 and 11.2 min before the mass-matched control was added; it adds 50 AUROC cells
# per model-layer, about 2 min per job). Peak memory is the pass-2 buffer, 110 pooled
# vectors per passage and layer: about 30 GB for Qwen3-0.6B, 24 GB for KaLM-mini,
# 12 GB for a 12-layer model.
#
# From a checkout without Ian's paths (pick the account with sbatch --test-only),
# a smoke run and then the two halves:
#   export CODE_ROOT=$PWD BASES_ROOT=<checkout>/runs/active/resubmit_bases \
#          PY=<env prefix>/bin/python
#   LIMIT=64 MODELS='LaTa|Qwen3-0.6B' sbatch --export=ALL --time=00:15:00 \
#       --account=<account> --partition=gpuA40x4-interactive slurm/reframe/reframe_e2.sbatch
#   MODELS='LaTa|PhilTa|mT5-base|LaBSE' sbatch --export=ALL --time=00:30:00 \
#       --account=<account> --partition=gpuA40x4-interactive slurm/reframe/reframe_e2.sbatch
#   MODELS='Qwen3-0.6B|KaLM-mini' sbatch --export=ALL --time=00:30:00 \
#       --account=<account> --partition=gpuA40x4-interactive slurm/reframe/reframe_e2.sbatch
# Each job gates every model whose rows are in the CSVs and requires rows for its own
# MODELS. After the last job, on a login node (no GPU, no caches, no torch):
#   $PY scripts/paper/reframe/e2_token_audit.py check     # all six models, exit 3 on failure
#   $PY scripts/paper/reframe/e2_token_audit.py render
# The caches were extracted on an A40; the same GPU type gives the closest vectors.

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BASES_ROOT="${BASES_ROOT:-/projects/beto/irowerojas/localLatin/runs/active/resubmit_bases}"
PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
MODELS="${MODELS:-}"
LIMIT="${LIMIT:-0}"
CONTROL_METRICS="${CONTROL_METRICS:-auroc}"
TOL_AUROC="${TOL_AUROC:-}"
NPROC="${SLURM_CPUS_PER_TASK:-8}"
cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

# Compute nodes have no network: the models must already be in the HuggingFace cache.
# The cache is set explicitly, because a shell profile may point HF_HOME elsewhere and
# transformers 4 would follow TRANSFORMERS_CACHE.
export HF_HOME="${HF_CACHE:-/work/nvme/bimc/swong2/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# One BLAS thread per metric worker; the parallelism is across metric cells.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "code: $CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse --short HEAD)) bases: $BASES_ROOT"
echo "python: $PY  HF_HOME: $HF_HOME  models: ${MODELS:-all}  limit: $LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

EXTRA=()
if [ -n "$MODELS" ]; then EXTRA+=(--models "${MODELS//|/,}"); fi
TOL=()
if [ -n "$TOL_AUROC" ]; then TOL+=(--tol_auroc "$TOL_AUROC"); fi
RENDER=()
if [ "$LIMIT" != "0" ]; then
  SMOKE="runs/active/reframe/e2/smoke_limit$LIMIT"
  EXTRA+=(--out_dir "$SMOKE")
  RENDER+=(--tab_dir "$SMOKE")  # never overwrite the paper table with smoke numbers
fi

echo "=== audit + gates ==="
RC=0
$PY scripts/paper/reframe/e2_token_audit.py audit --check \
  --bases_root "$BASES_ROOT" --workers "$NPROC" --limit "$LIMIT" \
  --control_metrics "$CONTROL_METRICS" \
  ${EXTRA[@]+"${EXTRA[@]}"} ${TOL[@]+"${TOL[@]}"} || RC=$?

# 3 = the CSVs were written but a gate failed: still render, so the facts file shows
# which gate and by how much, then exit non-zero.
if [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; then
  echo "=== audit failed (exit $RC), nothing rendered ==="
  exit "$RC"
fi

# The table and the facts file cover every model already in the CSVs; --allow_missing
# keeps a model that another job has yet to run from being reported as a failed gate here.
echo "=== render ==="
$PY scripts/paper/reframe/e2_token_audit.py render --allow_missing \
  ${EXTRA[@]+"${EXTRA[@]}"} ${RENDER[@]+"${RENDER[@]}"} ${TOL[@]+"${TOL[@]}"}

echo "=== done (gate exit $RC) ==="
exit "$RC"
