#!/bin/bash
# Issue #123: supervised fine-tuning REFERENCE CEILING for LaTa -- GPU half.
#
# This job does only the work that needs an A100: contrastive fine-tuning of
# LaTa's encoder on the train split's positive pairs, and extraction of
# mean-pooled embeddings for all 1,705 labelled files at every encoder layer.
# It also re-extracts with the PRE-TRAINED weights and diffs against the
# paper's cached embeddings, which proves the harness is byte-for-byte
# comparable before any fine-tuned number is believed.
#
# Scoring the embeddings (Task A, Task B, the 5-seed protocol) only reads .npy
# files, so per the repo's GPU-budget rule it runs in the companion CPU job:
#
#     sbatch slurm/resubmit/finetune_lata_ceiling_eval.sbatch
#
# Budget: SLURM charges the reserved wall time, not the elapsed time, so
# --time is kept close to the measured runtime. On one A100 the parity pass,
# 7 training epochs and extraction of all 12 layers took 1m24s end to end;
# the 45-minute reservation is margin for a slower node or more epochs.
#
#SBATCH --job-name=ft_lata_ceiling
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/finetune_lata_ceiling_%j.out
#SBATCH --error=slurm/logs/finetune_lata_ceiling_%j.err

set -euo pipefail

module reset
module load miniforge3-python

# CODE_ROOT holds scripts/, src/ and data/; REPO_ROOT holds the runs/ tree the
# paper's artifacts already live in. They are the same path in a normal
# checkout, and differ when the job is submitted from a git worktree.
CODE_ROOT="${CODE_ROOT:-/u/irowerojas/localLatin}"
REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"

export PYTHONUNBUFFERED=1

cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
BASELINE_BASES="$REPO_ROOT/runs/active/resubmit_bases"
BASELINE_RESULTS="$REPO_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv"
OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune"
BASES_ROOT="$REPO_ROOT/runs/active/resubmit_finetune_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results/finetune"

mkdir -p "$OUT_DIR" "$BASES_ROOT" "$RESULTS_DIR"

echo "=== Host/GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== Fine-tune + extract (parity-checked) ==="
# --no-capture-output: plain `conda run` buffers everything until the process
# exits, which hides progress and makes the SLURM watchdog's log-staleness
# check fire on a perfectly healthy job.
conda run --no-capture-output -n localLatin python scripts/resubmit/finetune_lata_ceiling.py \
  --split_csv "$SPLIT_CSV" \
  --data_root "$CODE_ROOT" \
  --model_name "bowphs/LaTa" \
  --ft_model_label "bowphs/LaTa-ft" \
  --out_dir "$OUT_DIR" \
  --bases_root "$BASES_ROOT" \
  --baseline_bases_root "$BASELINE_BASES" \
  --baseline_results_csv "$BASELINE_RESULTS" \
  --results_dir "$RESULTS_DIR" \
  --stages train,extract \
  --parity_check \
  --seed 42 \
  --dev_dir_frac 0.15 \
  --epochs 8 \
  --lr 2e-5 \
  --batch_pairs 16 \
  --temperature 0.05 \
  --patience 3 \
  --max_length 512 \
  --layers "1-12"

echo "=== Verifying outputs ==="
FT_DIR="$BASES_ROOT/phase9_bases/bowphs_LaTa-ft/hidden_mean_tokempty"
for f in "$OUT_DIR/encoder_best.pt" "$OUT_DIR/dev_curve.csv" "$OUT_DIR/selection.json"; do
  [ -f "$f" ] || { echo "ERROR: $f was not created!" >&2; exit 1; }
  echo "OK: $f"
done
N_LAYERS="$(ls "$FT_DIR"/hidden_layer*_embeddings.npy 2>/dev/null | wc -l)"
[ "$N_LAYERS" -eq 12 ] || { echo "ERROR: expected 12 layer files, found $N_LAYERS" >&2; exit 1; }
echo "OK: $N_LAYERS layers in $FT_DIR"
cat "$OUT_DIR/selection.json"

echo "=== GPU accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,AllocTRES%50,TotalCPU || true

echo "=== Done. Next: sbatch slurm/resubmit/finetune_lata_ceiling_eval.sbatch ==="
