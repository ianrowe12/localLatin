#!/bin/bash
#SBATCH --job-name=hidden_lasttok
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
# module load python

export REPO_ROOT="$WORK/localLatin"
export CANON_ROOT="$WORK/canon"
export RUNS_ROOT="$WORK/runs/ff1_lata_postact"

export HF_HOME="$WORK/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/.cache/huggingface"
export HF_DATASETS_CACHE="$WORK/.cache/huggingface"

cd "$REPO_ROOT" || exit 1

python src/extract_hidden_cli.py \
  --meta_csv "$RUNS_ROOT/meta.csv" \
  --runs_root "$RUNS_ROOT" \
  --model_name "bowphs/LaTa" \
  --layers "0-12" \
  --pooling lasttok \
  --max_length 512 \
  --batch_size 12
