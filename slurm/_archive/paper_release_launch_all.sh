#!/bin/bash
set -euo pipefail

echo "=== Submitting revised paper-release pipeline ==="

PH11_ID=$(sbatch --parsable slurm/paper_release_phase11_tokempty.sbatch)
echo "Phase 11 tokempty release: $PH11_ID"

P12E_ID=$(sbatch --parsable --dependency=afterok:$PH11_ID slurm/paper_release_phase12e.sbatch)
echo "Phase 12e release: $P12E_ID"

PACK_ID=$(sbatch --parsable --dependency=afterok:$PH11_ID:$P12E_ID slurm/paper_release_package_assets.sbatch)
echo "Paper release packaging: $PACK_ID"

echo "=== Submission complete ==="
