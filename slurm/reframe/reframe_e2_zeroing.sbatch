#!/bin/bash
#SBATCH --job-name=reframe_e2_zeroing
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/reframe_e2_zeroing_%j.out
#SBATCH --error=slurm/logs/reframe_e2_zeroing_%j.err

# Issue #252 (E2, zeroing follow-up): what survives coordinate zeroing, at every layer
# of the six panel models.
# CPU only: reads the cached mean-pooled vectors, never extracts or trains.
# Writes runs/active/reframe/e2/e2_zeroing_followup.csv, e2_zeroing_gate_check.csv and
# facts_e2_zeroing.md under CODE_ROOT. The gates read
# runs/active/reframe/e1/e1_coordinate_ablation.csv and
# runs/active/reframe/h1/h1_d_ablation.csv under CODE_ROOT.
#
# Overridable through the environment (sbatch --export=ALL with the variables set):
#   CODE_ROOT      checkout holding the scripts and receiving the outputs
#                  (default: the submit directory)
#   BASES_ROOT     directory holding phase9_bases/<slug>/hidden_mean_tokempty/
#   PY             python launcher, may be several words, e.g.
#                  "conda run --no-capture-output -p <env prefix> python"
#   MODELS         '|'-separated display names or ids (default: all six);
#                  '|' because --export splits on commas
#   ALLOW_MISSING  1 = skip a model whose cache is absent instead of failing
#   TOL            tolerance of reproduction gates 2, 3 and 4 (default 1e-6)
#
# From a checkout without Ian's paths (pick the account with sbatch --test-only):
#   CODE_ROOT=$PWD BASES_ROOT=<checkout>/runs/active/resubmit_bases \
#   PY=<env prefix>/bin/python \
#   sbatch --export=ALL --account=<account> --partition=cpu-interactive \
#       slurm/reframe/reframe_e2_zeroing.sbatch

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BASES_ROOT="${BASES_ROOT:-/projects/beto/irowerojas/localLatin/runs/active/resubmit_bases}"
PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
MODELS="${MODELS:-}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
TOL="${TOL:-}"
NPROC="${SLURM_CPUS_PER_TASK:-16}"
cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

# One BLAS thread per forked worker; the parallelism is across model-layers.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "code: $CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse --short HEAD)) bases: $BASES_ROOT"
echo "python: $PY"

EXTRA=()
if [ -n "$MODELS" ]; then EXTRA+=(--models "${MODELS//|/,}"); fi
if [ "$ALLOW_MISSING" = "1" ]; then EXTRA+=(--allow_missing); fi
if [ -n "$TOL" ]; then EXTRA+=(--tol "$TOL"); fi

echo "=== compute + reproduction gates ==="
RC=0
$PY scripts/paper/reframe/e2_zeroing_followup.py compute --check \
  --bases_root "$BASES_ROOT" --workers "$NPROC" \
  ${EXTRA[@]+"${EXTRA[@]}"} || RC=$?

# 3 = the CSV was written but a reproduction gate failed: still render, so the facts
# file shows which gate and by how much, then exit non-zero.
if [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; then
  echo "=== compute failed (exit $RC), nothing rendered ==="
  exit "$RC"
fi

echo "=== render ==="
$PY scripts/paper/reframe/e2_zeroing_followup.py render \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "=== done (gate exit $RC) ==="
exit "$RC"
