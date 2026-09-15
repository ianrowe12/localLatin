#!/bin/bash
# Issue #194: supervised fine-tuning REFERENCE CEILING for Qwen3-0.6B -- GPU half.
#
# The same recipe #123 ran on LaTa, on a model that was never pre-trained on
# Latin: symmetric InfoNCE with in-batch negatives, temperature 0.05, AdamW
# 2e-5 / wd 0.01, 16 pairs per batch, 8 epochs with patience 3 on dev
# directory accuracy@1, warmup 10% then linear decay, grad clip 1.0, seed 42,
# and the same 28-directory dev carve of the same 190 multi-witness train
# directories, which is what makes the 499 training pairs identical.
#
# POOLING PARITY. The paper's Qwen3-0.6B rows are hidden-state MEAN pooling
# with the `tokenizer_empty` filter at max_length 512
# (runs/active/resubmit_bases/phase9_bases/Qwen_Qwen3-Embedding-0.6B/
#  hidden_mean_tokempty/config.json), not last-token pooling. Training and
# extraction here use exactly that, and --parity_check proves it by
# re-extracting with the PRE-TRAINED weights and diffing against that cache.
#
# RECIPE DELTA. Gradient checkpointing is on. Qwen3-0.6B is 596M parameters
# against LaTa's 110M encoder, and 32 sequences of 512 tokens through 28 blocks
# does not fit beside fp32 AdamW state on one A100 without it. The gradients
# are identical -- activations are recomputed rather than stored -- so nothing
# about the objective, the schedule or the batch changes. It is recorded in
# run_info.json as grad_checkpointing.
#
# Scoring only reads .npy files, so per the repo's GPU-budget rule it runs in
# the companion CPU job:
#
#     sbatch slurm/resubmit/finetune_qwen_ceiling_eval.sbatch
#
# Budget: SLURM charges the reserved wall time, not the elapsed time. LaTa's
# equivalent job measured 1m29s; Qwen3-0.6B is 5.4x the parameters, 28 blocks
# instead of 12, and extracts 28 layers instead of 12, so 30 minutes is the
# reservation. GPU use for this issue was approved by Ian on 2026-09-15.
#
#SBATCH --job-name=ft_qwen_ceiling
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/finetune_qwen_ceiling_%j.out
#SBATCH --error=slurm/logs/finetune_qwen_ceiling_%j.err

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
OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune/qwen3_0.6b"
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
conda run --no-capture-output -n localLatin python scripts/resubmit/finetune_ceiling.py \
  --split_csv "$SPLIT_CSV" \
  --data_root "$CODE_ROOT" \
  --model_name "Qwen/Qwen3-Embedding-0.6B" \
  --ft_model_label "Qwen/Qwen3-Embedding-0.6B-ft" \
  --display_name "Qwen3-0.6B" \
  --results_prefix "finetune_qwen3_0.6b" \
  --out_dir "$OUT_DIR" \
  --bases_root "$BASES_ROOT" \
  --baseline_bases_root "$BASELINE_BASES" \
  --baseline_results_csv "$BASELINE_RESULTS" \
  --results_dir "$RESULTS_DIR" \
  --stages train,extract \
  --parity_check \
  --trust_remote_code \
  --grad_checkpointing \
  --seed 42 \
  --dev_dir_frac 0.15 \
  --epochs 8 \
  --lr 2e-5 \
  --batch_pairs 16 \
  --temperature 0.05 \
  --patience 3 \
  --max_length 512 \
  --eval_batch_size 16 \
  --layers "1-28"

echo "=== Verifying outputs ==="
FT_DIR="$BASES_ROOT/phase9_bases/Qwen_Qwen3-Embedding-0.6B-ft/hidden_mean_tokempty"
for f in "$OUT_DIR/encoder_best.pt" "$OUT_DIR/dev_curve.csv" "$OUT_DIR/selection.json"; do
  [ -f "$f" ] || { echo "ERROR: $f was not created!" >&2; exit 1; }
  echo "OK: $f"
done
N_LAYERS="$(ls "$FT_DIR"/hidden_layer*_embeddings.npy 2>/dev/null | wc -l)"
[ "$N_LAYERS" -eq 28 ] || { echo "ERROR: expected 28 layer files, found $N_LAYERS" >&2; exit 1; }
echo "OK: $N_LAYERS layers in $FT_DIR"
cat "$OUT_DIR/selection.json"

echo "=== GPU accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true

echo "=== Done. Next: sbatch slurm/resubmit/finetune_qwen_ceiling_eval.sbatch ==="
