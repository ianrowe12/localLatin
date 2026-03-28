#!/bin/bash
#SBATCH --job-name=ph12c_philta
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH -o slurm/logs/phase12c_philta_%j.out
#SBATCH -e slurm/logs/phase12c_philta_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1

echo "=== Phase 12c: PhilTa Retrieval Attribution ==="
conda run -n localLatin python scripts/run_phase12c_retrieval_attribution.py \
    --model_name bowphs/PhilTa \
    --model_type t5 \
    --pc_dir "$REPO_ROOT/runs/phase12/pcs/bowphs_PhilTa" \
    --pairs_csv "$REPO_ROOT/runs/phase12c/retrieval_pairs.csv" \
    --layers "1,6,9,12" \
    --out_dir "$REPO_ROOT/runs/phase12c/attributions/bowphs_PhilTa" \
    --D 10 \
    --n_steps 50

echo "=== Done ==="
