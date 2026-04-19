#!/bin/bash
#SBATCH --job-name=attr_metrics
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/attr_metrics_%j.out
#SBATCH --error=slurm/logs/attr_metrics_%j.err

set -euo pipefail
cd /projects/beto/irowerojas/worktrees/run-2-agent-2

mkdir -p slurm/logs runs/active/ig_examples/attribution_metrics

PY=/u/irowerojas/.conda/envs/localLatin/bin/python

echo "=== smoke: 1 pair on bowphs/LaTa ==="
$PY scripts/ig/run_attribution_metrics.py \
    --models "bowphs/LaTa" \
    --max_pairs_per_model 1

echo "=== full run: 4 models x 20 pairs ==="
$PY scripts/ig/run_attribution_metrics.py
