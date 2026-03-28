#!/bin/bash
#SBATCH --job-name=phase6_generalize
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python
conda run -n localLatin python -m pip install -r /u/irowerojas/localLatin/requirements.txt
conda run -n localLatin python -c "import sklearn; print(sklearn.__version__)"

export REPO_ROOT="/u/irowerojas/localLatin"
export CANON_ROOT="/u/irowerojas/localLatin/canon"
export RUNS_ROOT="/u/irowerojas/localLatin/runs"

# REQUIRED: update to the Phase 5 run you want to generalize.
export PHASE5_RUN_DIR="/u/irowerojas/localLatin/runs/phase5_hidden12_mean/run_20260127_193832"

export PHASE6_ROOT="$RUNS_ROOT/phase6_generalization"

cd "$REPO_ROOT" || exit 1

conda run -n localLatin bash "$REPO_ROOT/scripts/run_phase6.sh"
