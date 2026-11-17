#!/bin/bash
#SBATCH --job-name=d2_score
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/dataset2_score_%j.out
#SBATCH --error=slurm/logs/dataset2_score_%j.err
#
# Issues #289 and #298: dataset-2 scoring and paper parity (CPU caches only).
#
#   1. manifest       sha256 + row-order check of the Qwen3-8B caches written by
#                     dataset2_extract_qwen8b.sbatch (skipped if absent)
#   2. select_qwen8b  the paper's train DirAcc@1 rule for Qwen3-8B on benchmark
#                     v1 train (36 layers x 3 poolings, parallel)
#   3. score          rankings + tables under runs/active/dataset2/
#   4. compare        character n-grams + train-best paired differences
#   5. layers         agreed acc@1/MRR at every cached layer, train-only D
#   6. render         CSV-driven LaTeX tables + depth PDF/PNG
#
# --time: about an hour of work, with the same 3x margin as before. The
# selected-layer steps were estimated at ~25 minutes. `layers` is ~900 cells at
# 0.1-1.1 s each (timed at real sizes on synthetic data), and 2.5-6.6 s for a
# train D retune, which only runs where a per-layer D record is missing: at
# most ~30 minutes. `compare` and `render` take minutes. No GPU allocation,
# extraction or connection to another host is needed.
#
# Submit from DATA_ROOT (the main checkout, whose runs/ holds the caches):
#   cd /u/irowerojas/localLatin && sbatch $CODE_ROOT/slurm/resubmit/dataset2_score.sbatch
# SNAPSHOT and OUT_DIR may be overridden; SKIP_QWEN8B=1 skips 8B preparation.
# It does not remove existing 8B caches from the all-layer sweep.

set -euo pipefail

module reset
module load miniforge3-python

DATA_ROOT="${DATA_ROOT:-/u/irowerojas/localLatin}"
CODE_ROOT="${CODE_ROOT:-$DATA_ROOT}"
SNAPSHOT="${SNAPSHOT:-runs/active/dataset2/snapshot_20261115_run43762601690}"
OUT_DIR="${OUT_DIR:-runs/active/dataset2}"
N_BOOT="${N_BOOT:-2000}"
SEED="${SEED:-289}"
cd "$DATA_ROOT" || exit 1

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="conda run --no-capture-output -n localLatin python"
SCORE="$CODE_ROOT/scripts/resubmit/score_dataset2.py"
SLUG="Qwen_Qwen3-Embedding-8B"
UNLAB="runs/active/resubmit/unlabelled/bases/$SLUG"
LAB="runs/active/resubmit_bases/phase9_bases/$SLUG"
META="runs/active/dataset2/extraction_meta"

if [ "${SKIP_QWEN8B:-0}" != 1 ]; then
  echo "=== Manifest of the Qwen3-8B dataset 2 caches ==="
  $PY "$SCORE" manifest \
    --run_dirs "$UNLAB/hidden_mean_tokempty" "$UNLAB/hidden_lasttok_tokempty" \
               "$UNLAB/hidden_sif_tokempty" "$LAB/hidden_sif_tokempty" \
    --expected_meta "$META/unlabelled_meta.csv" "$META/labelled_meta.csv" \
                    runs/active/resubmit/data/unlabelled_meta.csv \
    --out_json runs/active/dataset2/qwen8b_extraction_manifest.json

  echo "=== Qwen3-8B train-only layer selection ==="
  $PY "$SCORE" select_qwen8b --workers "${SLURM_CPUS_PER_TASK:-8}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== Score ==="
$PY "$SCORE" score --snapshot "$SNAPSHOT" --out_dir "$OUT_DIR" \
  --n_boot "$N_BOOT" --seed "$SEED"

echo "=== Reference and paired differences ==="
$PY "$SCORE" compare --snapshot "$SNAPSHOT" --out_dir "$OUT_DIR" \
  --n_boot "$N_BOOT" --seed "$SEED"

echo "=== All cached layers ==="
$PY "$SCORE" layers --snapshot "$SNAPSHOT" --out_dir "$OUT_DIR"

echo "=== Paper outputs ==="
$PY "$SCORE" render --out_dir "$OUT_DIR"

echo "=== Done ==="
