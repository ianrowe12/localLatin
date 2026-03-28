#!/bin/bash
#SBATCH --job-name=unlab_philta
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/resubmit_unlabelled_extract_philta_%j.out
#SBATCH --error=slurm/logs/resubmit_unlabelled_extract_philta_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

META_CSV="$REPO_ROOT/runs/phase_resubmit/data/unlabelled_meta.csv"
SPLIT_CSV="$REPO_ROOT/runs/phase_resubmit/data/phase_resubmit_split.csv"
BASES="$REPO_ROOT/runs/phase_resubmit/unlabelled/bases"
SLUG="bowphs_PhilTa"

echo "=== Extracting bowphs/PhilTa unlabelled (mean pooling) ==="
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

echo "=== Extracting bowphs/PhilTa unlabelled (SIF pooling) ==="
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
