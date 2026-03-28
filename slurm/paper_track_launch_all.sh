#!/bin/bash
set -euo pipefail

echo "=== Submitting paper-track filtered pipeline ==="

PH11_ID=$(sbatch --parsable slurm/paper_phase11_filtered.sbatch)
echo "Phase 11 filtered: $PH11_ID"

PH12D_ID=$(sbatch --parsable slurm/paper_phase12d_filtering.sbatch)
echo "Phase 12d filtering: $PH12D_ID"

P12PREP_ID=$(sbatch --parsable --dependency=afterok:$PH11_ID slurm/paper_phase12_prepare_filtered.sbatch)
echo "Phase 12 prepare filtered: $P12PREP_ID"

LATA_ID=$(sbatch --parsable --dependency=afterok:$P12PREP_ID slurm/paper_phase12c_attr_lata_filtered.sbatch)
PHILTA_ID=$(sbatch --parsable --dependency=afterok:$P12PREP_ID slurm/paper_phase12c_attr_philta_filtered.sbatch)
LABSE_ID=$(sbatch --parsable --dependency=afterok:$P12PREP_ID slurm/paper_phase12c_attr_labse_filtered.sbatch)
QWEN_ID=$(sbatch --parsable --dependency=afterok:$P12PREP_ID slurm/paper_phase12c_attr_qwen_filtered.sbatch)
KALM_ID=$(sbatch --parsable --dependency=afterok:$P12PREP_ID slurm/paper_phase12c_attr_kalm_filtered.sbatch)
MT5_ID=$(sbatch --parsable --dependency=afterok:$P12PREP_ID slurm/paper_phase12c_attr_mt5_filtered.sbatch)
echo "Phase 12c attrs: $LATA_ID $PHILTA_ID $LABSE_ID $QWEN_ID $KALM_ID $MT5_ID"

P12C_ANA_ID=$(sbatch --parsable --dependency=afterok:$LATA_ID:$PHILTA_ID:$LABSE_ID:$QWEN_ID:$KALM_ID:$MT5_ID slurm/paper_phase12c_analysis_filtered.sbatch)
echo "Phase 12c analysis: $P12C_ANA_ID"

PACK_ID=$(sbatch --parsable --dependency=afterok:$PH11_ID:$PH12D_ID:$P12C_ANA_ID slurm/paper_package_assets.sbatch)
echo "Paper packaging: $PACK_ID"

echo "=== Submission complete ==="
