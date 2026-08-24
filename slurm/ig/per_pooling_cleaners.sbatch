#!/bin/bash
#SBATCH --job-name=ig_pooling_pcs
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/ig_pooling_pcs_%j.out
#SBATCH --error=slurm/logs/ig_pooling_pcs_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# Issue #87 -- rebuild every artifact's sif_abtt panel in the SIF-pooled ABTT
# subspace.
#
# No GPU. The artifacts already store full raw token hidden states, attention
# and both IG vectors, so swapping the cleaner behind the sif_abtt variant is
# pure post-processing over data on disk: fit the deployed SIF-pooled cleaner
# from the labelled train split, re-clean the tokens with it, rebuild the six
# pair matrices, reweight by the SIF token weights. Measured on a login node:
# ~2 min for the D sweeps, 38 s to rewrite all 128 artifacts, ~2 min to verify.
#
# What this does NOT change: baseline / abtt / sif matrices, hidden states, IG
# vectors, decoded tokens and the retrieval_mark sidecar are all left
# byte-identical. Only pair_matrix_*_sif_abtt, its topk_* arrays, and the new
# pcs_sif / mean_vec_sif / D_sif / sif_abtt_cleaner_pooling keys move.

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$CODE_ROOT"
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"

IG_ROOT="$DATA_ROOT/runs/active/ig_examples"
EXAMPLES_CSV="$IG_ROOT/phase12f_examples.csv"
ARTIFACTS_DIR="$IG_ROOT/artifacts"
PC_ROOT="$DATA_ROOT/runs/phase12_release/pcs"
LAB_BASES="$DATA_ROOT/runs/active/resubmit_bases/phase9_bases"
SPLIT_CSV="$DATA_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
PROBS_CACHE="$IG_ROOT/sif_token_probs"

echo "=== Environment ==="
echo "code_root=$CODE_ROOT"
echo "data_root=$DATA_ROOT"
date

echo ""
echo "=== [1/3] Fit the SIF-pooled cleaner per (model, layer) ($(date)) ==="
# --seed_mean_from_artifacts fills in the mean keys for the two (model, layer)
# pairs that never had a PC file (LaBSE L12, Qwen L23); it copies them out of
# the artifacts and never overwrites an existing mean fit.
$PY -u scripts/ig/fit_pooling_cleaners.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --labelled_bases "$LAB_BASES" \
    --split_csv "$SPLIT_CSV" \
    --pc_root "$PC_ROOT" \
    --seed_mean_from_artifacts

echo ""
echo "=== [2/3] Rebuild sif / sif_abtt over all artifacts ($(date)) ==="
# No --refresh_probs: the examples CSV is unchanged, so the cached train-only
# token probabilities are the same ones the shipped weights were built from.
$PY -u scripts/resubmit/persist_sif_attribution.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --split_csv "$SPLIT_CSV" \
    --repo_root "$DATA_ROOT" \
    --token_probs_cache "$PROBS_CACHE" \
    --pc_root "$PC_ROOT"

echo ""
echo "=== [3/3] Verify subspaces and reproduce the matrices ($(date)) ==="
$PY -u scripts/ig/verify_pooling_cleaners.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --labelled_bases "$LAB_BASES" \
    --split_csv "$SPLIT_CSV"

$PY -u scripts/ig/verify_demo_artifacts.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== Done ($(date)) ==="
