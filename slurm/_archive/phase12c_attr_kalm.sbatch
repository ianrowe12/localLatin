#!/bin/bash
#SBATCH --job-name=ph12c_kalm
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH -o slurm/logs/phase12c_kalm_%j.out
#SBATCH -e slurm/logs/phase12c_kalm_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12c: KaLM-mini Retrieval Attribution ==="
conda run -n localLatin python scripts/run_phase12c_retrieval_attribution.py \
    --model_name KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 \
    --model_type decoder \
    --pc_dir "$REPO_ROOT/runs/phase12/pcs/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --pairs_csv "$REPO_ROOT/runs/phase12c/retrieval_pairs.csv" \
    --layers "1,5,13,23" \
    --out_dir "$REPO_ROOT/runs/phase12c/attributions/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --D 10 \
    --n_steps 50 \
    --trust_remote_code

echo "=== Done ==="
