#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG=slurm/logs/lasttok_launch_$(date +%Y%m%d_%H%M%S).log
mkdir -p slurm/logs

echo "== Submitting 5 fast extractions in parallel =="
JID_LATA=$(sbatch --parsable slurm/resubmit/lasttok_extract_lata.sbatch)
JID_PHILTA=$(sbatch --parsable slurm/resubmit/lasttok_extract_philta.sbatch)
JID_LABSE=$(sbatch --parsable slurm/resubmit/lasttok_extract_labse.sbatch)
JID_QWEN06=$(sbatch --parsable slurm/resubmit/lasttok_extract_qwen06.sbatch)
JID_KALM=$(sbatch --parsable slurm/resubmit/lasttok_extract_kalm.sbatch)

echo "LaTa:   $JID_LATA"
echo "PhilTa: $JID_PHILTA"
echo "LaBSE:  $JID_LABSE"
echo "Qwen06: $JID_QWEN06"
echo "KaLM:   $JID_KALM"

echo "== Submitting Qwen3-8B mean extraction =="
JID_8B_MEAN=$(sbatch --parsable slurm/resubmit/lasttok_extract_qwen8b_mean.sbatch)
echo "Qwen8B-mean:    $JID_8B_MEAN"

echo "== Submitting Qwen3-8B lasttok extraction (after 8B-mean finishes) =="
JID_8B_LAST=$(sbatch --parsable --dependency=afterany:$JID_8B_MEAN slurm/resubmit/lasttok_extract_qwen8b.sbatch)
echo "Qwen8B-lasttok: $JID_8B_LAST (depends on $JID_8B_MEAN)"

DEPS="afterok:$JID_LATA:$JID_PHILTA:$JID_LABSE:$JID_QWEN06:$JID_KALM:$JID_8B_MEAN:$JID_8B_LAST"
echo "== Submitting evaluation with dependency $DEPS =="
JID_EVAL=$(sbatch --parsable --dependency=$DEPS slurm/resubmit/lasttok_evaluate.sbatch)
echo "Eval: $JID_EVAL"

echo ""
echo "All job IDs submitted. Monitor with: squeue -u \$USER"
echo "Extraction IDs: $JID_LATA $JID_PHILTA $JID_LABSE $JID_QWEN06 $JID_KALM $JID_8B_MEAN $JID_8B_LAST"
echo "Eval ID:        $JID_EVAL"
