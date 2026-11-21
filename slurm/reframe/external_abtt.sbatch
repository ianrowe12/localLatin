#!/bin/bash
#SBATCH --job-name=ext_abtt
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=/u/irowerojas/localLatin/slurm/logs/ext_abtt_%j.out
#SBATCH --error=/u/irowerojas/localLatin/slurm/logs/ext_abtt_%j.err
# Siddique 2026-11-21: ABTT components fit on the decretals instead of the canon train split.
# CPU only, ~7 min. The cpu queue can wait days; --partition=cpu-interactive starts at once.
set -euo pipefail
cd /u/irowerojas/localLatin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
/u/irowerojas/.conda/envs/localLatin/bin/python "${CODE_ROOT:-/u/irowerojas/localLatin}"/scripts/paper/reframe/external_abtt.py --workers 16
