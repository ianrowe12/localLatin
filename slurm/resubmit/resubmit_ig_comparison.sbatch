#!/bin/bash
#SBATCH --job-name=ig_compare_f
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/ig_comparison_f_%j.out
#SBATCH --error=slurm/logs/ig_comparison_f_%j.err

set -euo pipefail
cd /u/irowerojas/localLatin
mkdir -p slurm/logs

module reset
module load miniforge3-python

echo "=== Phase 12f IG Comparison — 6 methods, all examples (4 models) ==="
conda run -n localLatin python scripts/resubmit/run_resubmit_ig_comparison.py \
    --examples_csv runs/active/ig_examples/phase12f_examples.csv \
    --artifacts_dir runs/active/ig_examples/artifacts \
    --out_dir runs/active/resubmit/ig_comparison \
    --max_tokens 20 \
    --max_examples 100

echo "=== Done. Outputs in runs/active/resubmit/ig_comparison/ ==="
