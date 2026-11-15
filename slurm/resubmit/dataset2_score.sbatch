#!/bin/bash
#SBATCH --job-name=d2_score
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/dataset2_score_%j.out
#SBATCH --error=slurm/logs/dataset2_score_%j.err
#
# Issue #289: score the paper's models on dataset 2 (CPU only; reads caches).
#
#   1. manifest       sha256 + row-order check of the Qwen3-8B caches written by
#                     dataset2_extract_qwen8b.sbatch (skipped if absent)
#   2. select_qwen8b  the paper's train DirAcc@1 rule for Qwen3-8B on benchmark
#                     v1 train (36 layers x 3 poolings, parallel)
#   3. score          rankings + tables under runs/active/dataset2/
#
# --time: selection is ~108 layer jobs of ~10 train DirAcc@1 evaluations each
# over 16 workers (~10 min); scoring is ~45 (model, method) cells, each a few
# seconds of matrix work plus one D sweep (~15 min). 90 min is about 3x that.
#
# Submit from DATA_ROOT (the main checkout, whose runs/ holds the caches):
#   cd /u/irowerojas/localLatin && sbatch $CODE_ROOT/slurm/resubmit/dataset2_score.sbatch
# SNAPSHOT and OUT_DIR may be overridden; SKIP_QWEN8B=1 skips the 8B steps.

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:-/projects/beto/irowerojas/worktrees/issue-289}"
DATA_ROOT="${DATA_ROOT:-/u/irowerojas/localLatin}"
SNAPSHOT="${SNAPSHOT:-runs/active/dataset2/snapshot_20261115_run43762601690}"
cd "$DATA_ROOT" || exit 1

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

echo "=== Score ==="
$PY "$SCORE" score --snapshot "$SNAPSHOT" --out_dir "${OUT_DIR:-runs/active/dataset2}"

echo "=== Done ==="
