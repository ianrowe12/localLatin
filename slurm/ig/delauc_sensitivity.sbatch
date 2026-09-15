#!/bin/bash
#SBATCH --job-name=delauc_sens
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/delauc_sens_%j.out
#SBATCH --error=slurm/logs/delauc_sens_%j.err

# Issue #195 (G9). Sensitivity of the chance-corrected deletion AUC gap to its
# own knobs, over the stored hidden states of the benchmark v1 attribution
# sample. CPU only: the hidden backend never loads a model, and the arithmetic
# is a cumulative sum plus a cosine per deletion point.
#
# BLAS threads are pinned to one on purpose. The matrices here are (n+1, 768)
# with n in the low hundreds, so a threaded GEMM spends more time synchronising
# than computing: the same sweep takes 57 s wall with default threading and
# 32 s with one thread, and the difference is all system time.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/ig:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# The tokenizers for the token-filter arms are already in the local HF cache;
# a compute node has no outbound network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN="${RUN:-runs/active/ig_examples_200pos_v1}"
OUT="$RUN/attribution_metrics/sensitivity"
mkdir -p "$OUT"

echo "=== DelAUC-gap sensitivity sweep ($(date)) ==="
$PY scripts/ig/run_delauc_sensitivity.py \
    --run_dir "$RUN" \
    --out_dir "$OUT" \
    --verify_cache "$RUN/attribution_metrics/v2_hidden"

echo ""
echo "=== Appendix table ($(date)) ==="
$PY scripts/ig/build_delauc_sensitivity_table.py \
    --configs_csv "$OUT/configs.csv" \
    --cells_csv "$OUT/cells.csv" \
    --out overleaf_drafts/tables/attribution_delauc_sensitivity.tex

echo ""
echo "=== Done ($(date)) ==="
