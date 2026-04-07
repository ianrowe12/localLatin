#!/bin/bash
#SBATCH --job-name=paper_p12d
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/paper_phase12d_%j.out
#SBATCH --error=slurm/logs/paper_phase12d_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/phase12d_filtered"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name "bowphs/LaTa" --model_type t5 --layers "1,4,8,12" --D 10 --out_dir "$REPO_ROOT/runs/phase12d_filtered"
conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name "bowphs/PhilTa" --model_type t5 --layers "1,6,9,12" --D 10 --out_dir "$REPO_ROOT/runs/phase12d_filtered"
conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name "sentence-transformers/LaBSE" --model_type bert --layers "1,6,9,11" --D 10 --out_dir "$REPO_ROOT/runs/phase12d_filtered"
conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name "Qwen/Qwen3-Embedding-0.6B" --model_type decoder --layers "1,4,14,28" --D 10 --trust_remote_code --out_dir "$REPO_ROOT/runs/phase12d_filtered"
conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" --model_type decoder --layers "1,5,13,23" --D 10 --trust_remote_code --out_dir "$REPO_ROOT/runs/phase12d_filtered"
conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
  --model_name "google/mt5-base" --model_type t5 --layers "1,4,8,12" --D 10 --out_dir "$REPO_ROOT/runs/phase12d_filtered"

conda run -n localLatin python scripts/combine_phase12d_results.py \
  --input_dir "$REPO_ROOT/runs/phase12d_filtered" \
  --output_csv "$REPO_ROOT/runs/phase12d_filtered/filtering_results_all.csv"

conda run -n localLatin python scripts/plot_phase12d_filtering.py \
  --results_csv "$REPO_ROOT/runs/phase12d_filtered/filtering_results_all.csv" \
  --out_dir "$REPO_ROOT/runs/phase12d_filtered"
