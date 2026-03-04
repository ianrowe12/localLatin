#!/bin/bash
#SBATCH --job-name=ph12_kalm
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12: KaLM-mini Attribution ==="
conda run -n localLatin python scripts/run_phase12_attribution.py \
    --model_name KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 \
    --model_type decoder \
    --pc_dir "$REPO_ROOT/runs/phase12/pcs/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --meta_csv "$REPO_ROOT/runs/phase9_bases/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5/hidden_mean/meta.csv" \
    --sample_csv "$REPO_ROOT/runs/phase12/phase12_samples.csv" \
    --layers "1,5,13,23" \
    --out_dir "$REPO_ROOT/runs/phase12/attributions/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --D 10 \
    --n_steps 50 \
    --half_precision \
    --trust_remote_code

echo "=== Done ==="
