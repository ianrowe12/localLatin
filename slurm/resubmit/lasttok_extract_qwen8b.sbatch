#!/bin/bash
#SBATCH --job-name=resub_lasttok_qwen8b
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=07:00:00
#SBATCH --output=slurm/logs/lasttok_extract_qwen8b_%j.out
#SBATCH --error=slurm/logs/lasttok_extract_qwen8b_%j.err

set -euo pipefail

module reset
module load miniforge3-python

export REPO_ROOT="/projects/beto/irowerojas/worktrees/run-1-agent-3"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

META_CSV="$REPO_ROOT/runs/active/resubmit/data/meta.csv"
BASES="$REPO_ROOT/runs/active/resubmit_bases/phase9_bases"
SLUG="Qwen_Qwen3-Embedding-8B"

echo "=== Extracting Qwen/Qwen3-Embedding-8B (lasttok pooling, layers 0-50, half precision) ==="
conda run -n localLatin python src/extract_encoder_cli.py \
  --meta_csv "$META_CSV" \
  --runs_root "$BASES" \
  --run_dir "$BASES/$SLUG/hidden_lasttok_tokempty" \
  --model_name "Qwen/Qwen3-Embedding-8B" \
  --repr hidden \
  --layers "0-50" \
  --pooling lasttok \
  --token_filter tokenizer_empty \
  --max_length 512 \
  --batch_size 2 \
  --half_precision \
  --trust_remote_code

echo "=== Done ==="
