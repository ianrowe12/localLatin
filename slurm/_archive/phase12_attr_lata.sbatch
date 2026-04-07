#!/bin/bash
#SBATCH --job-name=ph12_lata
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12: LaTa Attribution ==="
conda run -n localLatin python scripts/run_phase12_attribution.py \
    --model_name bowphs/LaTa \
    --model_type t5 \
    --pc_dir "$REPO_ROOT/runs/phase12/pcs/bowphs_LaTa" \
    --meta_csv "$REPO_ROOT/runs/phase9_bases/bowphs_LaTa/hidden_mean/meta.csv" \
    --layers "11,12" \
    --out_dir "$REPO_ROOT/runs/phase12/attributions/bowphs_LaTa" \
    --D 10 \
    --n_steps 50

echo "=== Done ==="
