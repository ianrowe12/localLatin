#!/bin/bash
# Issue #123: supervised fine-tuning REFERENCE CEILING for LaTa -- scoring half.
#
# Reads the .npy embeddings written by slurm/resubmit/finetune_lata_ceiling.sbatch
# and scores them with the paper's own evaluator: Task A (AUROC, cosine gap),
# Task B (assignment accuracy, dir_acc@1 with tau learned on train), and the
# 5-seed Task B protocol at the selected layers for both the fine-tuned and the
# pre-trained encoder. It also emits the comparison CSV and the LaTeX rows.
#
# No GPU: this only reads cached embeddings, which the repo's budget rule sends
# to the CPU partition. The runtime is dominated by the per-layer ABTT D sweep,
# whose directory-assignment scoring is a Python loop over 847 train files x 513
# directories, repeated for six D values. Measured at 3m18s on a cpu node.
#
#SBATCH --job-name=ft_lata_eval
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/finetune_lata_ceiling_eval_%j.out
#SBATCH --error=slurm/logs/finetune_lata_ceiling_eval_%j.err

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
OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune"
BASES_ROOT="$REPO_ROOT/runs/active/resubmit_finetune_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results/finetune"
TEX_OUT="$CODE_ROOT/overleaf_drafts/tables/finetune_ceiling.tex"

mkdir -p "$RESULTS_DIR"

echo "=== Scoring the fine-tuning ceiling (CPU) ==="
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
  --tex_out "$TEX_OUT" \
  --stages evaluate,report,mseed \
  --seed 42 \
  --dev_dir_frac 0.15 \
  --layers "1-12" \
  --D_values "1,2,3,5,7,10" \
  --mseed_M 5 --mseed_base_seed 42 \
  --cpu

echo "=== Verifying outputs ==="
for f in finetune_lata_layer_results.csv \
         finetune_lata_ceiling_comparison.csv \
         finetune_lata_mseed_aggregated.csv; do
  [ -f "$RESULTS_DIR/$f" ] || { echo "ERROR: $RESULTS_DIR/$f was not created!" >&2; exit 1; }
  echo "OK: $f ($(wc -l < "$RESULTS_DIR/$f") lines)"
done
[ -f "$TEX_OUT" ] && echo "OK: $(basename "$TEX_OUT")"

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,AllocTRES%50,TotalCPU || true

echo "=== Done ==="
