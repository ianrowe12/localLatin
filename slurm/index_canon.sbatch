#!/bin/bash
#SBATCH --job-name=canon_index
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-cpu

module reset
# module load python

export REPO_ROOT="$WORK/localLatin"
export CANON_ROOT="$WORK/canon"
export RUNS_ROOT="$WORK/runs/ff1_lata_postact"

export HF_HOME="$WORK/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/.cache/huggingface"
export HF_DATASETS_CACHE="$WORK/.cache/huggingface"

cd "$REPO_ROOT" || exit 1

python src/index_canon_cli.py \
  --canon_root "$CANON_ROOT" \
  --runs_root "$RUNS_ROOT" \
  --model_name "bowphs/LaTa" \
  --max_length 512 \
  --batch_size 128
