#!/bin/bash
#SBATCH --job-name=paper_p12c_kalm
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/paper_phase12c_kalm_%j.out
#SBATCH --error=slurm/logs/paper_phase12c_kalm_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/phase12c_filtered/attributions/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/run_phase12c_retrieval_attribution.py \
  --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --model_type decoder \
  --pc_dir "$REPO_ROOT/runs/phase12_filtered/pcs/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --pairs_csv "$REPO_ROOT/runs/phase12c/retrieval_pairs.csv" \
  --layers "1,5,13,23" \
  --out_dir "$REPO_ROOT/runs/phase12c_filtered/attributions/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --D 10 \
  --n_steps 50 \
  --require_all_layers \
  --trust_remote_code \
  --token_filter no_empty
