#!/bin/bash
#SBATCH --job-name=paper_relpkg
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_release_package_%j.out
#SBATCH --error=slurm/logs/paper_release_package_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/paper_release_assets"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/package_paper_release_assets.py \
  --phase11_csv "$REPO_ROOT/runs/phase11_release/phase11_results.csv" \
  --phase11_fig_dir "$REPO_ROOT/runs/phase11_release/figures" \
  --dist_dir "$REPO_ROOT/runs/phase11_release/distributions" \
  --taskb_dir "$REPO_ROOT/runs/phase12_release/taskb" \
  --phase12e_fig_dir "$REPO_ROOT/runs/phase12e_release/figures" \
  --out_dir "$REPO_ROOT/runs/paper_release_assets"
