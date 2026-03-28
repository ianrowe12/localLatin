#!/bin/bash
#SBATCH --job-name=unlab_retrieval
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/resubmit_unlabelled_retrieval_%j.out
#SBATCH --error=slurm/logs/resubmit_unlabelled_retrieval_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

echo "=== Unlabelled retrieval ==="
conda run -n localLatin python scripts/run_resubmit_unlabelled_retrieval.py \
  --split_csv "$REPO_ROOT/runs/phase_resubmit/data/phase_resubmit_split.csv" \
  --labelled_bases "$REPO_ROOT/runs/phase_resubmit_bases/phase9_bases" \
  --unlabelled_bases "$REPO_ROOT/runs/phase_resubmit/unlabelled/bases" \
  --unlabelled_meta "$REPO_ROOT/runs/phase_resubmit/data/unlabelled_meta.csv" \
  --results_csv "$REPO_ROOT/runs/phase_resubmit/results/phase_resubmit_results.csv" \
  --out_dir "$REPO_ROOT/runs/phase_resubmit/unlabelled"

echo "=== Done ==="
