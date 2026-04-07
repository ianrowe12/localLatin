#!/bin/bash
#SBATCH --job-name=persist_attribution_methods
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/persist_attribution_methods_%j.out
#SBATCH --error=slurm/logs/persist_attribution_methods_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/projects/beto/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs"
cd "$REPO_ROOT" || exit 1

echo "=== Persisting 6 attribution methods x {baseline, abtt} per pair ==="

conda run -n localLatin python scripts/resubmit/persist_attribution_methods.py \
    --examples_csv runs/active/ig_examples/phase12f_examples.csv \
    --artifacts_dir runs/active/ig_examples/artifacts

echo "=== Done. Check phase12f_examples.csv methods_available column and NPZ keys. ==="
