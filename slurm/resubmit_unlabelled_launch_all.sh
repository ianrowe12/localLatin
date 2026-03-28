#!/bin/bash
# Launch all unlabelled extraction jobs in parallel,
# then submit retrieval job after all extractions complete.
set -euo pipefail

cd /u/irowerojas/localLatin
mkdir -p slurm/logs

echo "Submitting 6 extraction jobs..."

JOB_LATA=$(sbatch --parsable slurm/resubmit_unlabelled_extract_lata.sbatch)
echo "  LaTa:   $JOB_LATA"

JOB_PHILTA=$(sbatch --parsable slurm/resubmit_unlabelled_extract_philta.sbatch)
echo "  PhilTa: $JOB_PHILTA"

JOB_MT5=$(sbatch --parsable slurm/resubmit_unlabelled_extract_mt5.sbatch)
echo "  MT5:    $JOB_MT5"

JOB_LABSE=$(sbatch --parsable slurm/resubmit_unlabelled_extract_labse.sbatch)
echo "  LaBSE:  $JOB_LABSE"

JOB_QWEN=$(sbatch --parsable slurm/resubmit_unlabelled_extract_qwen.sbatch)
echo "  Qwen:   $JOB_QWEN"

JOB_KALM=$(sbatch --parsable slurm/resubmit_unlabelled_extract_kalm.sbatch)
echo "  KaLM:   $JOB_KALM"

# Submit retrieval after all extractions complete
DEPS="afterok:${JOB_LATA}:${JOB_PHILTA}:${JOB_MT5}:${JOB_LABSE}:${JOB_QWEN}:${JOB_KALM}"
JOB_RET=$(sbatch --parsable --dependency="$DEPS" slurm/resubmit_unlabelled_retrieval.sbatch)
echo "  Retrieval: $JOB_RET (depends on all extractions)"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
