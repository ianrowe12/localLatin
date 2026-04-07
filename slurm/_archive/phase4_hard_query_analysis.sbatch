#!/bin/bash
#SBATCH --job-name=phase4_hard_queries
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

export REPO_ROOT="/u/irowerojas/localLatin"
export PHASE3_RUN_DIR="/u/irowerojas/localLatin/runs/phase3_hidden12_mean/run_20260126_153356"
export OUT_ROOT_DIR="/u/irowerojas/localLatin/runs/phase4_hard_queries"

cd "$REPO_ROOT" || exit 1

python src/phase4_hard_query_analysis_cli.py \
  --phase3_run_dir "$PHASE3_RUN_DIR" \
  --out_root_dir "$OUT_ROOT_DIR" \
  --layer 12 \
  --k_list "1,3,5"
