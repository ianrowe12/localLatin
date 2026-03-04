#!/bin/bash
#SBATCH --job-name=ph12_philta
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12: PhilTa Attribution ==="
conda run -n localLatin python scripts/run_phase12_attribution.py \
    --model_name bowphs/PhilTa \
    --model_type t5 \
    --pc_dir "$REPO_ROOT/runs/phase12/pcs/bowphs_PhilTa" \
    --meta_csv "$REPO_ROOT/runs/phase9_bases/bowphs_PhilTa/hidden_mean/meta.csv" \
    --layers "12" \
    --out_dir "$REPO_ROOT/runs/phase12/attributions/bowphs_PhilTa" \
    --D 10 \
    --n_steps 50

echo "=== Done ==="
