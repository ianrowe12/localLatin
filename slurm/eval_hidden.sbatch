#!/bin/bash
#SBATCH --job-name=hidden_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-cpu

module reset
# module load python

export REPO_ROOT="$WORK/localLatin"
export RUNS_ROOT="$WORK/runs/ff1_lata_postact"

cd "$REPO_ROOT" || exit 1

RUN_DIR="$RUNS_ROOT/run_YYYYMMDD_HHMMSS"

python src/eval_retrieval_cli.py \
  --run_dir "$RUN_DIR" \
  --repr hidden \
  --pooling mean \
  --layers "0-12"
