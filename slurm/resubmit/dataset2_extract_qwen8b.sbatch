#!/bin/bash
#SBATCH --job-name=d2_qwen8b
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/dataset2_extract_qwen8b_%j.out
#SBATCH --error=slurm/logs/dataset2_extract_qwen8b_%j.err
#
# Issue #289: Qwen3-Embedding-8B embeddings for the 2,238 unlabelled queries
# (dataset 2), the one GPU extraction Ian approved on 2026-11-15.
#
# Writes, never overwriting (each target directory must not exist yet):
#   unlabelled: runs/active/resubmit/unlabelled/bases/Qwen_Qwen3-Embedding-8B/
#       hidden_{mean,lasttok,sif}_tokempty/   layers 0-36
#   labelled:   runs/active/resubmit_bases/phase9_bases/Qwen_Qwen3-Embedding-8B/
#       hidden_sif_tokempty/                  layers 0-36
# mean and lasttok are the two poolings the labelled 8B cache already holds
# (lasttok_extract_qwen8b*.sbatch). SIF did not exist for 8B on either side;
# the labelled SIF run is what makes the SIF methods scoreable at all, and both
# SIF runs take P(w) from the benchmark v1 train split only.
#
# --time: the April labelled runs (17708594/17708595, 1,705 files, batch 2)
# took 29 min per pooling, because the per-layer extractor reran the model 37
# times. That is ~0.8 min of forward pass per 1,705 files per layer; scaled to
# 2,238 files it is ~1 min per pooling with --one_pass, plus ~2 min model load
# per invocation. 5 invocations (parity check + 4 caches) come to ~15-20 min;
# 50 min leaves room for a slow load from /projects.
#
# Needs: REPO_ROOT is a checkout of this branch whose runs/ and data/ resolve to
# the main checkout's (symlinks), so outputs land in the canonical cache.

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/worktrees/issue-289}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

MODEL="Qwen/Qwen3-Embedding-8B"
SLUG="Qwen_Qwen3-Embedding-8B"
SPLIT_CSV="runs/active/resubmit/data/phase_resubmit_split.csv"
D2="runs/active/dataset2"
META_DIR="$D2/extraction_meta"
LAB_BASES="runs/active/resubmit_bases/phase9_bases"
UNLAB_BASES="runs/active/resubmit/unlabelled/bases"
PY="conda run --no-capture-output -n localLatin python"

for d in \
  "$UNLAB_BASES/$SLUG/hidden_mean_tokempty" \
  "$UNLAB_BASES/$SLUG/hidden_lasttok_tokempty" \
  "$UNLAB_BASES/$SLUG/hidden_sif_tokempty" \
  "$LAB_BASES/$SLUG/hidden_sif_tokempty"; do
  if [ -e "$d" ]; then
    echo "REFUSING: $d already exists; this job never overwrites a cache." >&2
    exit 1
  fi
done

echo "=== Extraction metas (paths relative to the repo root) ==="
$PY scripts/resubmit/dataset2_extraction_meta.py --out_dir "$META_DIR"

COMMON=(--model_name "$MODEL" --repr hidden --layers "0-36" --token_filter tokenizer_empty
        --max_length 512 --batch_size 2 --half_precision --trust_remote_code --one_pass)

echo "=== Parity: --one_pass vs the cached labelled lasttok and mean runs (16 files) ==="
PARITY="$D2/one_pass_parity_${SLURM_JOB_ID:-manual}"
for pool in lasttok mean; do
  CACHE="$LAB_BASES/$SLUG/hidden_${pool}_tokempty"
  $PY scripts/resubmit/check_one_pass_parity.py make_meta \
    --cache_dir "$CACHE" --split_csv "$SPLIT_CSV" --n_files 16 \
    --out_meta "$PARITY/meta_${pool}.csv"
  $PY src/extract_encoder_cli.py "${COMMON[@]}" --layers "0,1,18,35,36" \
    --pooling "$pool" --meta_csv "$PARITY/meta_${pool}.csv" \
    --runs_root "$PARITY" --run_dir "$PARITY/hidden_${pool}"
  suffix=""; [ "$pool" = lasttok ] && suffix="_lasttok"
  $PY scripts/resubmit/check_one_pass_parity.py compare \
    --cache_dir "$CACHE" --fresh_dir "$PARITY/hidden_${pool}" --suffix "$suffix"
done

echo "=== Unlabelled: mean, lasttok, sif ==="
for pool in mean lasttok sif; do
  EXTRA=()
  [ "$pool" = sif ] && EXTRA=(--split_csv "$SPLIT_CSV")
  $PY src/extract_encoder_cli.py "${COMMON[@]}" --pooling "$pool" ${EXTRA[@]+"${EXTRA[@]}"} \
    --meta_csv "$META_DIR/unlabelled_meta.csv" \
    --runs_root "$UNLAB_BASES" --run_dir "$UNLAB_BASES/$SLUG/hidden_${pool}_tokempty"
done

echo "=== Labelled: sif ==="
$PY src/extract_encoder_cli.py "${COMMON[@]}" --pooling sif --split_csv "$SPLIT_CSV" \
  --meta_csv "$META_DIR/labelled_meta.csv" \
  --runs_root "$LAB_BASES" --run_dir "$LAB_BASES/$SLUG/hidden_sif_tokempty"

echo "=== Done ==="
