#!/bin/bash
#SBATCH --job-name=phase5_eval_h12
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-cpu

module reset
# module load python

export REPO_ROOT="$WORK/localLatin"
export PHASE5_RUN_DIR="$REPO_ROOT/runs/phase5_hidden12_mean/run_YYYYMMDD_HHMMSS"

cd "$REPO_ROOT" || exit 1

python src/phase5_full_eval_cli.py \
  --screening_csv "$PHASE5_RUN_DIR/phase5_screening_scoreboard.csv" \
  --top_n 5 \
  --layer 12 \
  --output_csv "$PHASE5_RUN_DIR/phase5_full_eval_scoreboard.csv"
