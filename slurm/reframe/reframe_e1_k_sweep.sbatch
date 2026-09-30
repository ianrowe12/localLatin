#!/bin/bash
#SBATCH --job-name=reframe_e1_k_sweep
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48g
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/reframe_e1_k_sweep_%j.out
#SBATCH --error=slurm/logs/reframe_e1_k_sweep_%j.err

# Issue #246, post hoc: E1 zeroing sweep beyond k = 10, with a random-coordinate control.
# CPU only: reads the cached mean-pooled vectors. Writes runs/active/reframe/e1/
# e1_k_sweep.csv, e1_k_sweep_gate.csv, facts_e1_k_sweep.md and e1_k_sweep.pdf.
#
# Environment (sbatch --export=ALL,...):
#   CODE_ROOT   checkout holding the scripts and receiving the outputs (default: submit dir)
#   BASES_ROOT  directory holding phase9_bases/<slug>/hidden_mean_tokempty/
#   PY          python launcher
# Account: pass --account=<picked> on the command line (sbatch --test-only picks it).

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BASES_ROOT="${BASES_ROOT:-$CODE_ROOT/runs/active/resubmit_bases}"
PY="${PY:-/projects/bimc/swong2/envs/localLatin/bin/python}"
NPROC="${SLURM_CPUS_PER_TASK:-16}"
cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "code: $CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse --short HEAD)) bases: $BASES_ROOT"

RC=0
$PY scripts/paper/reframe/e1_k_sweep.py compute --check \
  --bases_root "$BASES_ROOT" --workers "$NPROC" || RC=$?
if [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; then
  echo "=== compute failed (exit $RC) ==="
  exit "$RC"
fi
$PY scripts/paper/reframe/e1_k_sweep.py render
echo "=== done (gate exit $RC) ==="
exit "$RC"
