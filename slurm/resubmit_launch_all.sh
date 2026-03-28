#!/bin/bash
set -euo pipefail

echo "=== Submitting resubmit pipeline ==="

# Step 1: 6 parallel extraction jobs
LATA_ID=$(sbatch --parsable slurm/resubmit_extract_lata.sbatch)
echo "Extract LaTa: $LATA_ID"

PHILTA_ID=$(sbatch --parsable slurm/resubmit_extract_philta.sbatch)
echo "Extract PhilTa: $PHILTA_ID"

MT5_ID=$(sbatch --parsable slurm/resubmit_extract_mt5.sbatch)
echo "Extract mt5-base: $MT5_ID"

LABSE_ID=$(sbatch --parsable slurm/resubmit_extract_labse.sbatch)
echo "Extract LaBSE: $LABSE_ID"

QWEN_ID=$(sbatch --parsable slurm/resubmit_extract_qwen.sbatch)
echo "Extract Qwen3-0.6B: $QWEN_ID"

KALM_ID=$(sbatch --parsable slurm/resubmit_extract_kalm.sbatch)
echo "Extract KaLM-mini: $KALM_ID"

ALL_EXTRACT="$LATA_ID:$PHILTA_ID:$MT5_ID:$LABSE_ID:$QWEN_ID:$KALM_ID"

# Step 2: Evaluation (depends on all extractions)
EVAL_ID=$(sbatch --parsable --dependency=afterok:$ALL_EXTRACT slurm/resubmit_evaluate.sbatch)
echo "Evaluation: $EVAL_ID"

# Step 3: Figures (depends on evaluation)
FIG_ID=$(sbatch --parsable --dependency=afterok:$EVAL_ID slurm/resubmit_figures.sbatch)
echo "Figures: $FIG_ID"

echo "=== Submission complete ==="
echo "Pipeline: Extract(${ALL_EXTRACT}) -> Eval(${EVAL_ID}) -> Figs(${FIG_ID})"
