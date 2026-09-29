#!/bin/bash
#SBATCH --job-name=reframe_ci_pq
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/reframe_ci_pq_%j.out
#SBATCH --error=slurm/logs/reframe_ci_pq_%j.err

# Issue #233 (CI + PQ): directory-level bootstrap intervals on every headline
# cell and the per-query routing checks. CPU only: reads cached embeddings and
# never trains or extracts anything.
#
# CODE_ROOT is the checkout holding the scripts; DATA_ROOT holds runs/ and
# data/ (they differ when the code runs from a git worktree). Outputs go under
# DATA_ROOT/runs/active/reframe/ and are never overwritten.
#
# Pilot: sbatch --time=00:25:00 --export=ALL,B=200,ONLY_MODELS='LaBSE|Qwen3-0.6B',OUT_TAG=ci_pq_pilot2 ...
# (ONLY_MODELS is '|'-separated because --export splits on commas.)

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
B="${B:-10000}"
SEED="${SEED:-233}"
ONLY_MODELS="${ONLY_MODELS:-}"
OUT_TAG="${OUT_TAG:-ci_pq}"
NPROC="${SLURM_CPUS_PER_TASK:-16}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
# One BLAS thread per forked worker; the parallelism is across processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "code: $CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse --short HEAD)) data: $DATA_ROOT"

echo "=== compute (B=$B seed=$SEED) ==="
$PY scripts/paper/reframe/ci_pq.py compute \
  --data_root "$DATA_ROOT" \
  --out_dir "runs/active/reframe/$OUT_TAG" \
  --B "$B" --seed "$SEED" \
  --workers "$NPROC" --procs "$NPROC" \
  ${ONLY_MODELS:+--only_models "$ONLY_MODELS"}

if [ "$SKIP_SWEEP" = "1" ]; then echo "=== sweep skipped ==="; exit 0; fi

echo "=== sweep (every layer, three threshold grids) ==="
$PY scripts/paper/reframe/ci_pq.py sweep \
  --data_root "$DATA_ROOT" \
  --out_dir "runs/active/reframe/${OUT_TAG}_sweep" \
  --procs "$NPROC" \
  ${ONLY_MODELS:+--only_models "$ONLY_MODELS"}

echo "=== done ==="
