#!/bin/bash
#SBATCH --job-name=paper_pack
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_package_%j.out
#SBATCH --error=slurm/logs/paper_package_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/runs/paper_assets"
cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/package_paper_track_assets.py \
  --phase11_csv "$REPO_ROOT/runs/phase11_filtered/phase11_results.csv" \
  --dist_dir "$REPO_ROOT/runs/phase11_filtered/distributions" \
  --phase12c_figures_dir "$REPO_ROOT/runs/phase12c_filtered/figures" \
  --phase12d_dir "$REPO_ROOT/runs/phase12d_filtered" \
  --out_dir "$REPO_ROOT/runs/paper_assets"
