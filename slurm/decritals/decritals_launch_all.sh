#!/bin/bash
set -euo pipefail

echo "=== Submitting decritals pipeline ==="

# --- Step 1: 6 parallel labelled extraction jobs ---
LATA_ID=$(sbatch --parsable slurm/decritals/decritals_extract_lata.sbatch)
echo "Labelled extract LaTa: $LATA_ID"

PHILTA_ID=$(sbatch --parsable slurm/decritals/decritals_extract_philta.sbatch)
echo "Labelled extract PhilTa: $PHILTA_ID"

MT5_ID=$(sbatch --parsable slurm/decritals/decritals_extract_mt5.sbatch)
echo "Labelled extract mt5-base: $MT5_ID"

LABSE_ID=$(sbatch --parsable slurm/decritals/decritals_extract_labse.sbatch)
echo "Labelled extract LaBSE: $LABSE_ID"

QWEN_ID=$(sbatch --parsable slurm/decritals/decritals_extract_qwen.sbatch)
echo "Labelled extract Qwen3-0.6B: $QWEN_ID"

KALM_ID=$(sbatch --parsable slurm/decritals/decritals_extract_kalm.sbatch)
echo "Labelled extract KaLM-mini: $KALM_ID"

ALL_LABELLED="$LATA_ID:$PHILTA_ID:$MT5_ID:$LABSE_ID:$QWEN_ID:$KALM_ID"

# --- Step 2: 6 parallel unlabelled extraction jobs (independent of labelled) ---
U_LATA_ID=$(sbatch --parsable slurm/decritals/decritals_unlabelled_extract_lata.sbatch)
echo "Unlabelled extract LaTa: $U_LATA_ID"

U_PHILTA_ID=$(sbatch --parsable slurm/decritals/decritals_unlabelled_extract_philta.sbatch)
echo "Unlabelled extract PhilTa: $U_PHILTA_ID"

U_MT5_ID=$(sbatch --parsable slurm/decritals/decritals_unlabelled_extract_mt5.sbatch)
echo "Unlabelled extract mt5-base: $U_MT5_ID"

U_LABSE_ID=$(sbatch --parsable slurm/decritals/decritals_unlabelled_extract_labse.sbatch)
echo "Unlabelled extract LaBSE: $U_LABSE_ID"

U_QWEN_ID=$(sbatch --parsable slurm/decritals/decritals_unlabelled_extract_qwen.sbatch)
echo "Unlabelled extract Qwen3-0.6B: $U_QWEN_ID"

U_KALM_ID=$(sbatch --parsable slurm/decritals/decritals_unlabelled_extract_kalm.sbatch)
echo "Unlabelled extract KaLM-mini: $U_KALM_ID"

ALL_UNLABELLED="$U_LATA_ID:$U_PHILTA_ID:$U_MT5_ID:$U_LABSE_ID:$U_QWEN_ID:$U_KALM_ID"

# --- Step 3: Task A evaluation (depends only on labelled extracts) ---
EVAL_ID=$(sbatch --parsable --dependency=afterok:$ALL_LABELLED slurm/decritals/decritals_evaluate.sbatch)
echo "Task A evaluation: $EVAL_ID  (waits for labelled extracts)"

# --- Step 4: Task B m-seed (parallel to eval; depends only on labelled extracts) ---
TASKB_ID=$(sbatch --parsable --dependency=afterok:$ALL_LABELLED slurm/decritals/decritals_taskb_mseed.sbatch)
echo "Task B m-seed: $TASKB_ID  (waits for labelled extracts; runs in parallel with eval)"

echo ""
echo "=== Submission complete ==="
echo "Labelled extracts: $ALL_LABELLED"
echo "Unlabelled extracts: $ALL_UNLABELLED  (independent branch)"
echo "Task A eval:        $EVAL_ID  (afterok labelled)"
echo "Task B m-seed:      $TASKB_ID  (afterok labelled, parallel with eval)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "After all complete, run: /u/irowerojas/.conda/envs/localLatin/bin/python scripts/decritals/build_decritals_summary.py"
