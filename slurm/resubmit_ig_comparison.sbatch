#!/bin/bash
#SBATCH --job-name=ig_compare
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/ig_comparison_%j.out
#SBATCH --error=slurm/logs/ig_comparison_%j.err

set -euo pipefail
cd /u/irowerojas/localLatin

# Step 1: Select examples
conda run -n localLatin python scripts/run_resubmit_select_ig_examples.py \
    --input_csv runs/phase12e_release/phase12e_examples.csv \
    --out_dir runs/phase_resubmit/ig_comparison

# Step 2: Generate comparison figures (IDF-BERTScore + contrastive)
conda run -n localLatin python scripts/run_resubmit_ig_comparison.py \
    --examples_csv runs/phase_resubmit/ig_comparison/selected_examples.csv \
    --full_examples_csv runs/phase12e_release/phase12e_examples.csv \
    --artifacts_dir runs/phase12e_release/artifacts \
    --out_dir runs/phase_resubmit/ig_comparison \
    --max_tokens 18

echo "Done. Figures saved to runs/phase_resubmit/ig_comparison/"
