#!/bin/bash
#SBATCH --job-name=persist_sif_tokens
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/persist_sif_tokens_%j.out
#SBATCH --error=slurm/logs/persist_sif_tokens_%j.err

# Issue #46 (B3) + #47.2 -- CPU-only post-processing of the highlight artifacts.
# Both passes are tokenizer-only and idempotent, so this can be rerun any time
# the baseline/abtt matrices change. The GPU job
# slurm/resubmit/ig_artifacts_mt5_kalm.sbatch runs the same two steps inline
# after generating the mT5-base / KaLM-mini NPZs.

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
mkdir -p "$REPO_ROOT/slurm/logs"
cd "$REPO_ROOT" || exit 1

EXAMPLES_CSV="runs/active/ig_examples/phase12f_examples.csv"
ARTIFACTS_DIR="runs/active/ig_examples/artifacts"

echo "=== Decoded token strings for all models ==="
conda run -n localLatin python scripts/ig/persist_decoded_tokens.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --overwrite

echo "=== sif / sif_abtt attribution variants for all models ==="
conda run -n localLatin python scripts/resubmit/persist_sif_attribution.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
    --token_probs_cache runs/active/ig_examples/sif_token_probs

echo "=== Done. NPZs now carry 4 variants x 7 methods plus token strings. ==="
