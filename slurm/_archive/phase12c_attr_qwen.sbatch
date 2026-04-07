#!/bin/bash
#SBATCH --job-name=ph12c_qwen
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm/logs/phase12c_qwen_%j.out
#SBATCH -e slurm/logs/phase12c_qwen_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12c: Qwen3-0.6B Retrieval Attribution ==="
conda run -n localLatin python scripts/run_phase12c_retrieval_attribution.py \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --model_type decoder \
    --pc_dir "$REPO_ROOT/runs/phase12/pcs/Qwen_Qwen3-Embedding-0.6B" \
    --pairs_csv "$REPO_ROOT/runs/phase12c/retrieval_pairs.csv" \
    --layers "1,4,14,28" \
    --out_dir "$REPO_ROOT/runs/phase12c/attributions/Qwen_Qwen3-Embedding-0.6B" \
    --D 10 \
    --n_steps 50 \
    --trust_remote_code

echo "=== Done ==="
