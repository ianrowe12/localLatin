#!/bin/bash
# Issue #194: supervised fine-tuning REFERENCE CEILING for Qwen3-0.6B -- scoring half.
#
# Reads the .npy embeddings written by slurm/resubmit/finetune_qwen_ceiling.sbatch
# and scores them with the paper's own evaluator: Task A (AUROC, cosine gap),
# Task B (assignment accuracy, dir_acc@1 with tau learned on train), and the
# 5-seed Task B protocol at the selected layers for both the fine-tuned and the
# pre-trained encoder.
#
# It does NOT write overleaf_drafts/tables/finetune_ceiling.tex. That table has
# exactly one writer, and since #210 it carries three ceilings, so the writer is
# the job for the last model added: slurm/resubmit/finetune_kalm_ceiling_eval.sbatch,
# which reads this run's comparison CSV and caption facts back off disk through
# --tex_extra_run. A Qwen-only re-run therefore refreshes its own CSVs and leaves
# the table to that job, which is what stops it from silently dropping KaLM-mini.
# Pass TEX_OUT plus --tex_extra_run for both other models if you ever need this
# job to render the table instead.
#
# No GPU: this only reads cached embeddings, which the repo's budget rule sends
# to the CPU partition. The runtime is dominated by the per-layer ABTT D sweep
# over 28 layers: 00:09:36 on one cpu node, 2.5x slower on another (a 20-minute
# reservation timed out at layer 18 of 28). The stage writes its CSV only when
# all 28 layers are done, so a timeout loses the whole sweep; 40 minutes is the
# reservation, which is 4x the fast node and 1.6x the slow one.
#
#SBATCH --job-name=ft_qwen_eval
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=00:40:00
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
BASES_ROOT="$REPO_ROOT/runs/active/resubmit_finetune_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results/finetune"

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
echo "NOTE: the generated table is written by finetune_kalm_ceiling_eval.sbatch;"
echo "      re-run that job to fold these CSVs back into tables/finetune_ceiling.tex."

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true

echo "=== Done ==="
