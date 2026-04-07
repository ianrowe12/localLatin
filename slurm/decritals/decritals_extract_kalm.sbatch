#!/bin/bash
#SBATCH --job-name=decr_kalm
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/decritals_extract_kalm_%j.out
#SBATCH --error=slurm/logs/decritals_extract_kalm_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

META_CSV="$REPO_ROOT/runs/active/decritals/data/meta.csv"
SPLIT_CSV="$REPO_ROOT/runs/active/decritals/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/active/decritals_bases/phase9_bases"
SLUG="KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5"

echo "=== Extracting KaLM-mini on decritals (mean pooling) ==="
conda run -n localLatin python src/extract_encoder_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_mean_tokempty" \
  --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --repr hidden \
  --layers "1-24" \
  --pooling mean \
  --token_filter tokenizer_empty \
  --max_length 512 \
  --batch_size 8 \
  --trust_remote_code

echo "=== Extracting KaLM-mini on decritals (SIF pooling) ==="
conda run -n localLatin python src/extract_encoder_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_sif_tokempty" \
  --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
  --repr hidden \
  --layers "1-24" \
  --pooling sif \
  --token_filter tokenizer_empty \
  --split_csv "$SPLIT_CSV" \
  --max_length 512 \
  --batch_size 8 \
  --trust_remote_code

echo "=== Done ==="
