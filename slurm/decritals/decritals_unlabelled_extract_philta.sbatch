#!/bin/bash
#SBATCH --job-name=decr_unlab_philta
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/decritals_unlabelled_extract_philta_%j.out
#SBATCH --error=slurm/logs/decritals_unlabelled_extract_philta_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

META_CSV="$REPO_ROOT/runs/active/decritals/unlabelled/meta_unlabelled.csv"
SPLIT_CSV="$REPO_ROOT/runs/active/decritals/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/active/decritals/unlabelled/bases"
SLUG="bowphs_PhilTa"

echo "=== Extracting bowphs/PhilTa decritals unlabelled (mean pooling) ==="
conda run -n localLatin python src/extract_hidden_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_mean_tokempty" \
  --model_name "bowphs/PhilTa" \
  --layers "1-12" \
  --pooling mean \
  --token_filter tokenizer_empty \
  --max_length 512 \
  --batch_size 8

echo "=== Extracting bowphs/PhilTa decritals unlabelled (SIF pooling) ==="
conda run -n localLatin python src/extract_hidden_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_sif_tokempty" \
  --model_name "bowphs/PhilTa" \
  --layers "1-12" \
  --pooling sif \
  --token_filter tokenizer_empty \
  --split_csv "$SPLIT_CSV" \
  --max_length 512 \
  --batch_size 8

echo "=== Done ==="
