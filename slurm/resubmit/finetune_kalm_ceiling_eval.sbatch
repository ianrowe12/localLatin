#!/bin/bash
# Issue #210: supervised fine-tuning REFERENCE CEILING for KaLM-mini -- scoring half.
#
# Reads the .npy embeddings written by slurm/resubmit/finetune_kalm_ceiling.sbatch
# and scores them with the paper's own evaluator: Task A (AUROC, cosine gap),
# Task B (assignment accuracy, dir_acc@1 with tau learned on train), and the
# 5-seed Task B protocol at the selected layers for both the fine-tuned and the
# pre-trained encoder.
#
# This job is the SINGLE WRITER of overleaf_drafts/tables/finetune_ceiling.tex,
# which since #210 carries three ceilings. The role moved here from the Qwen
# eval job for the same reason it moved there from the LaTa one: the last model
# added renders the table, so a re-run of an earlier model cannot silently drop
# a later one. --tex_extra_run reads LaTa's and Qwen3-0.6B's comparison CSVs and
# their saved caption facts back off disk and prints them above KaLM's rows, so
# no model inherits another's prose and none has to be re-scored to refresh the
# table.
#
# No GPU: this only reads cached embeddings, which the repo's budget rule sends
# to the CPU partition. The runtime is dominated by the per-layer ABTT D sweep,
# here over 24 layers. The same sweep over Qwen's 28 layers measured 00:05:07 on
# one cpu node and was still at layer 18 of 28 after 13 minutes on another. The
# stage writes its CSV only when every layer is done, so a timeout loses the
# whole sweep; 40 minutes is the reservation, sized for the slow node.
#
#SBATCH --job-name=ft_kalm_eval
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/finetune_kalm_ceiling_eval_%j.out
#SBATCH --error=slurm/logs/finetune_kalm_ceiling_eval_%j.err

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
OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune/kalm_mini"
LATA_OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune"
QWEN_OUT_DIR="$REPO_ROOT/runs/active/resubmit/finetune/qwen3_0.6b"
BASES_ROOT="$REPO_ROOT/runs/active/resubmit_finetune_bases"
RESULTS_DIR="$REPO_ROOT/runs/active/resubmit/results/finetune"
TEX_OUT="${TEX_OUT:-$CODE_ROOT/overleaf_drafts/tables/finetune_ceiling.tex}"

mkdir -p "$RESULTS_DIR"

echo "=== Scoring the KaLM-mini fine-tuning ceiling (CPU) ==="
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
  --tex_out "$TEX_OUT" \
  --tex_extra_run "LaTa:finetune_lata:$LATA_OUT_DIR" \
  --tex_extra_run "Qwen3-0.6B:finetune_qwen3_0.6b:$QWEN_OUT_DIR" \
  --stages evaluate,report,mseed \
  --seed 42 \
  --dev_dir_frac 0.15 \
  --layers "1-24" \
  --D_values "1,2,3,5,7,10" \
  --mseed_M 5 --mseed_base_seed 42 \
  --cpu

echo "=== Verifying outputs ==="
for f in finetune_kalm_mini_layer_results.csv \
         finetune_kalm_mini_ceiling_comparison.csv \
         finetune_kalm_mini_mseed_aggregated.csv; do
  [ -f "$RESULTS_DIR/$f" ] || { echo "ERROR: $RESULTS_DIR/$f was not created!" >&2; exit 1; }
  echo "OK: $f ($(wc -l < "$RESULTS_DIR/$f") lines)"
done
for system in "LaTa (fine-tuned)" "Qwen3-0.6B (fine-tuned)" "KaLM-mini (fine-tuned)"; do
  grep -q "$system" "$TEX_OUT" || {
    echo "ERROR: $TEX_OUT has no '$system' rows" >&2; exit 1; }
done
echo "OK: $(basename "$TEX_OUT") carries all three ceilings"

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true

echo "=== Done ==="
