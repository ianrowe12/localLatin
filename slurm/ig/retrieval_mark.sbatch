#!/bin/bash
#SBATCH --job-name=retrieval_mark
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=03:30:00
#SBATCH --output=slurm/logs/retrieval_mark_%j.out
#SBATCH --error=slurm/logs/retrieval_mark_%j.err

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="/projects/beto/irowerojas/worktrees/run-2-agent-1"
mkdir -p "$REPO_ROOT/slurm/logs"
cd "$REPO_ROOT"

echo "=== retrieval_mark: start $(date) ==="
echo "Host: $(hostname)"
nvidia-smi || true

echo ""
echo "=== Phase 1: mask optimization sweep (GPU) ==="
conda run -n localLatin python scripts/ig/run_retrieval_mark_pair_examples.py \
    --examples_csv      runs/active/ig_examples/phase12f_examples.csv \
    --artifacts_in_dir  runs/active/ig_examples/artifacts \
    --artifacts_out_dir runs/active/ig_examples/retrieval_mark/artifacts \
    --steps 200 \
    --lr 0.1 \
    --lambda_sparsity 0.01 \
    --gamma_tv 0.001 \
    --init_mask_logit 2.197 \
    --early_stop_thresh 0.01 \
    --early_stop_min_steps 50 \
    --token_filter tokenizer_empty \
    --max_length 256 \
    --seed 0 \
    --trust_remote_code \
    --skip_existing

echo ""
echo "=== Phase 2: merge sidecars into canonical NPZs (CPU, seconds) ==="
conda run -n localLatin python scripts/ig/merge_retrieval_mark_into_canonical.py \
    --examples_csv    runs/active/ig_examples/phase12f_examples.csv \
    --canonical_dir   runs/active/ig_examples/artifacts \
    --sidecar_dir     runs/active/ig_examples/retrieval_mark/artifacts \
    --update_methods_available

echo ""
echo "=== retrieval_mark: done $(date) ==="
