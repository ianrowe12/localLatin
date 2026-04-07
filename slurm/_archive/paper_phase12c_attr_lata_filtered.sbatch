#!/bin/bash
#SBATCH --job-name=paper_p12c_lata
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/paper_phase12c_lata_%j.out
#SBATCH --error=slurm/logs/paper_phase12c_lata_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/phase12c_filtered/attributions/bowphs_LaTa"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/run_phase12c_retrieval_attribution.py \
  --model_name "bowphs/LaTa" \
  --model_type t5 \
  --pc_dir "$REPO_ROOT/runs/phase12_filtered/pcs/bowphs_LaTa" \
  --pairs_csv "$REPO_ROOT/runs/phase12c/retrieval_pairs.csv" \
  --layers "1,4,8,12" \
  --out_dir "$REPO_ROOT/runs/phase12c_filtered/attributions/bowphs_LaTa" \
  --D 10 \
  --n_steps 50 \
  --require_all_layers \
  --token_filter no_empty
