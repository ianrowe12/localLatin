#!/bin/bash
# Issue #194: supervised fine-tuning REFERENCE CEILING for Qwen3-0.6B -- scoring half.
#
# Reads the .npy embeddings written by slurm/resubmit/finetune_qwen_ceiling.sbatch
# and scores them with the paper's own evaluator: Task A (AUROC, cosine gap),
# Task B (assignment accuracy, dir_acc@1 with tau learned on train), and the
# 5-seed Task B protocol at the selected layers for both the fine-tuned and the
# pre-trained encoder.
#
# This job is the SINGLE WRITER of overleaf_drafts/tables/finetune_ceiling.tex,
# which since #194 carries both ceilings. --tex_extra_run reads LaTa's
# comparison CSV and its saved caption facts back off disk and prints them
# above Qwen's rows, so neither model inherits the other's prose and neither
# has to be re-scored to refresh the table.
#
# No GPU: this only reads cached embeddings, which the repo's budget rule sends
# to the CPU partition. The runtime is dominated by the per-layer ABTT D sweep;
# LaTa's 12 layers measured 2m46s, and this sweeps 28, so 1 hour is reserved.
#
#SBATCH --job-name=ft_qwen_eval
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/finetune_qwen_ceiling_eval_%j.out
#SBATCH --error=slurm/logs/finetune_qwen_ceiling_eval_%j.err

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:-/u/irowerojas/localLatin}"
REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"

export PYTHONUNBUFFERED=1

cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
BASELINE_BASES="$REPO_ROOT/runs/active/resubmit_bases"
BASELINE_RESULTS="$REPO_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv"
OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune/qwen3_0.6b"
LATA_OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune"
BASES_ROOT="$REPO_ROOT/runs/active/resubmit_finetune_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results/finetune"
TEX_OUT="$CODE_ROOT/overleaf_drafts/tables/finetune_ceiling.tex"

mkdir -p "$RESULTS_DIR"

echo "=== Scoring the Qwen3-0.6B fine-tuning ceiling (CPU) ==="
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
  --tex_out "$TEX_OUT" \
  --tex_extra_run "LaTa:finetune_lata:$LATA_OUT_DIR" \
  --stages evaluate,report,mseed \
  --seed 42 \
  --dev_dir_frac 0.15 \
  --layers "1-28" \
  --D_values "1,2,3,5,7,10" \
  --mseed_M 5 --mseed_base_seed 42 \
  --cpu

echo "=== Verifying outputs ==="
for f in finetune_qwen3_0.6b_layer_results.csv \
         finetune_qwen3_0.6b_ceiling_comparison.csv \
         finetune_qwen3_0.6b_mseed_aggregated.csv; do
  [ -f "$RESULTS_DIR/$f" ] || { echo "ERROR: $RESULTS_DIR/$f was not created!" >&2; exit 1; }
  echo "OK: $f ($(wc -l < "$RESULTS_DIR/$f") lines)"
done
grep -q "Qwen3-0.6B (fine-tuned)" "$TEX_OUT" || {
  echo "ERROR: $TEX_OUT has no Qwen3-0.6B rows" >&2; exit 1; }
grep -q "LaTa (fine-tuned)" "$TEX_OUT" || {
  echo "ERROR: $TEX_OUT lost the LaTa rows" >&2; exit 1; }
echo "OK: $(basename "$TEX_OUT") carries both ceilings"

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true

echo "=== Done ==="
