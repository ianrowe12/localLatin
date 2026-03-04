#!/bin/bash
# Launch all Phase 12 jobs with dependency chaining.
# Usage: bash slurm/phase12_launch_all.sh

set -e

echo "=== Submitting Phase 12 pipeline ==="

# Step 1: Prepare PCs (must finish before attribution jobs)
PREP_ID=$(sbatch --parsable slurm/phase12_prepare.sbatch)
echo "Prepare PCs: job $PREP_ID"

# Step 2: Attribution jobs (all depend on prepare, run in parallel)
LATA_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID slurm/phase12_attr_lata.sbatch)
echo "LaTa attribution: job $LATA_ID (after $PREP_ID)"

PHILTA_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID slurm/phase12_attr_philta.sbatch)
echo "PhilTa attribution: job $PHILTA_ID (after $PREP_ID)"

LABSE_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID slurm/phase12_attr_labse.sbatch)
echo "LaBSE attribution: job $LABSE_ID (after $PREP_ID)"

QWEN_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID slurm/phase12_attr_qwen.sbatch)
echo "Qwen3-0.6B attribution: job $QWEN_ID (after $PREP_ID)"

KALM_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID slurm/phase12_attr_kalm.sbatch)
echo "KaLM-mini attribution: job $KALM_ID (after $PREP_ID)"

# Step 3: Aggregate (depends on ALL attribution jobs finishing)
AGG_ID=$(sbatch --parsable --dependency=afterok:$LATA_ID:$PHILTA_ID:$LABSE_ID:$QWEN_ID:$KALM_ID slurm/phase12_aggregate.sbatch)
echo "Aggregate: job $AGG_ID (after all attribution jobs)"

echo ""
echo "=== All jobs submitted ==="
echo "Pipeline: $PREP_ID → {$LATA_ID, $PHILTA_ID, $LABSE_ID, $QWEN_ID, $KALM_ID} → $AGG_ID"
echo "Monitor with: squeue -u \$USER"
