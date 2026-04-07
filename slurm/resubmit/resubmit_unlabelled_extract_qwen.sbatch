#!/bin/bash
#SBATCH --job-name=unlab_qwen
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/resubmit_unlabelled_extract_qwen_%j.out
#SBATCH --error=slurm/logs/resubmit_unlabelled_extract_qwen_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

META_CSV="$REPO_ROOT/runs/active/resubmit/data/unlabelled_meta.csv"
SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/active/resubmit/unlabelled/bases"
SLUG="Qwen_Qwen3-Embedding-0.6B"

echo "=== Extracting Qwen3-0.6B unlabelled (mean pooling) ==="
conda run -n localLatin python src/extract_encoder_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_mean_tokempty" \
  --model_name "Qwen/Qwen3-Embedding-0.6B" \
  --repr hidden \
  --layers "1-28" \
  --pooling mean \
  --token_filter tokenizer_empty \
  --max_length 512 \
  --batch_size 8 \
  --half_precision \
  --trust_remote_code

echo "=== Extracting Qwen3-0.6B unlabelled (SIF pooling) ==="
conda run -n localLatin python src/extract_encoder_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_sif_tokempty" \
  --model_name "Qwen/Qwen3-Embedding-0.6B" \
  --repr hidden \
  --layers "1-28" \
  --pooling sif \
  --token_filter tokenizer_empty \
  --split_csv "$SPLIT_CSV" \
  --max_length 512 \
  --batch_size 8 \
  --half_precision \
  --trust_remote_code

echo "=== Done ==="
