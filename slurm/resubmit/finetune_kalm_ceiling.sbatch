#!/bin/bash
# Issue #210: supervised fine-tuning REFERENCE CEILING for KaLM-mini -- GPU half.
#
# The third run of the recipe #123 ran on LaTa and #194 ran on Qwen3-0.6B:
# symmetric InfoNCE with in-batch negatives, temperature 0.05, AdamW 2e-5 /
# wd 0.01, 16 pairs per batch, 8 epochs with patience 3 on dev directory
# accuracy@1, warmup 10% then linear decay, grad clip 1.0, seed 42, and the
# same 28-directory dev carve of the same 190 multi-witness train directories,
# which is what makes the 499 training pairs identical across all three models.
#
# POOLING PARITY. The paper's KaLM-mini rows are hidden-state MEAN pooling with
# the `tokenizer_empty` filter at max_length 512
# (runs/active/resubmit_bases/phase9_bases/
#  KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5/
#  hidden_mean_tokempty/config.json), not the last-token pooling a decoder
# embedding model is normally used with. Training and extraction here use
# exactly that, and --parity_check proves it by re-extracting with the
# PRE-TRAINED weights and diffing against that cache.
#
# --trust_remote_code is REQUIRED and is not a convenience. The checkpoint's
# config maps AutoModel to its own `modeling.Qwen2Model` with is_causal=false,
# so the attention is bidirectional; loading it without the remote code would
# give a causal stack and a different representation from the one the paper's
# KaLM-mini rows were extracted with. src/extract_encoder_cli.py is invoked
# with the same flag for those rows (slurm/resubmit/resubmit_extract_kalm.sbatch).
#
# RECIPE DELTA. Gradient checkpointing is on, as it is for Qwen3-0.6B. At 494M
# parameters KaLM-mini is in the same size class as Qwen3-0.6B's 596M, so the
# same memory argument applies to 32 sequences of 512 tokens through 24 blocks
# beside fp32 AdamW state. The gradients are identical -- activations are
# recomputed rather than stored -- so nothing about the objective, the schedule
# or the batch changes. It is recorded in run_info.json as grad_checkpointing.
#
# Scoring only reads .npy files, so per the repo's GPU-budget rule it runs in
# the companion CPU job:
#
#     sbatch slurm/resubmit/finetune_kalm_ceiling_eval.sbatch
#
# Budget: SLURM charges the reserved wall time, not the elapsed time.
# Qwen3-0.6B measured 00:05:07 for 6 epochs plus both full extraction passes at
# 596M parameters and 28 blocks; KaLM-mini is smaller on both counts, so 15
# minutes is the reservation, a ~3x margin on that measurement. GPU use for this
# issue was approved by Ian on 2026-09-15.
#
#SBATCH --job-name=ft_kalm_ceiling
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/finetune_kalm_ceiling_%j.out
#SBATCH --error=slurm/logs/finetune_kalm_ceiling_%j.err

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
OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune/kalm_mini"
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
  --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --ft_model_label "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5-ft" \
  --display_name "KaLM-mini" \
  --results_prefix "finetune_kalm_mini" \
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
  --layers "1-24"

echo "=== Verifying outputs ==="
FT_DIR="$BASES_ROOT/phase9_bases/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5-ft/hidden_mean_tokempty"
for f in "$OUT_DIR/encoder_best.pt" "$OUT_DIR/dev_curve.csv" "$OUT_DIR/selection.json"; do
  [ -f "$f" ] || { echo "ERROR: $f was not created!" >&2; exit 1; }
  echo "OK: $f"
done
N_LAYERS="$(ls "$FT_DIR"/hidden_layer*_embeddings.npy 2>/dev/null | wc -l)"
[ "$N_LAYERS" -eq 24 ] || { echo "ERROR: expected 24 layer files, found $N_LAYERS" >&2; exit 1; }
echo "OK: $N_LAYERS layers in $FT_DIR"
cat "$OUT_DIR/selection.json"

echo "=== GPU accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true

echo "=== Done. Next: sbatch slurm/resubmit/finetune_kalm_ceiling_eval.sbatch ==="
