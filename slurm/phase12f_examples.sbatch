#!/bin/bash
#SBATCH --job-name=phase12f_4model
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/phase12f_4model_%j.out
#SBATCH --error=slurm/logs/phase12f_4model_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/projects/beto/irowerojas/localLatin"
export RUNS_ROOT="$REPO_ROOT/runs"
mkdir -p "$REPO_ROOT/slurm/logs"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12f: 4-model artifact generation (LaTa, PhilTa, LaBSE, Qwen3-0.6B) ==="

# Step 1-3: Select examples, compute IG artifacts, create visualizations
bash scripts/run_phase12f_examples_pipeline.sh

echo "=== Artifacts done. Running 6-method comparison on all examples ==="

# Step 4: Generate 6-method comparison figures + metrics + summary table
conda run -n localLatin python scripts/run_resubmit_ig_comparison.py \
    --examples_csv runs/phase12f_examples/phase12f_examples.csv \
    --artifacts_dir runs/phase12f_examples/artifacts \
    --out_dir runs/phase_resubmit/ig_comparison \
    --max_tokens 20 \
    --max_examples 100

echo "=== Done. Check runs/phase12f_examples/ and runs/phase_resubmit/ig_comparison/ ==="
