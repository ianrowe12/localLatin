#!/bin/bash
#SBATCH --job-name=external_fixes
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/external_fixes_%j.out
#SBATCH --error=slurm/logs/external_fixes_%j.err

# Submit from the checkout with slurm/logs present. --partition=cpu-interactive starts faster.
# BASES_ROOT and EXT_ROOT, if set, must contain phase9_bases/.
set -euo pipefail
cd "${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}" scripts/paper/reframe/external_fixes.py \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --bases_root "${BASES_ROOT:-runs/active/resubmit_bases}" \
    --ext_root "${EXT_ROOT:-runs/active/decritals_bases}"
