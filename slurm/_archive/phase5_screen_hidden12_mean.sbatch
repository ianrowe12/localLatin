#!/bin/bash
#SBATCH --job-name=phase5_screen_h12
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

export REPO_ROOT="$WORK/localLatin"
export BASE_RUN_DIR="$REPO_ROOT/runs/ff1_lata_postact/run_20260125_154356"
export PHASE5_ROOT="$REPO_ROOT/runs/phase5_hidden12_mean"

cd "$REPO_ROOT" || exit 1

python src/phase5_hidden12_cli.py \
  --base_run_dir "$BASE_RUN_DIR" \
  --out_root_dir "$PHASE5_ROOT" \
  --layer 12 \
  --variance_drop_percents "25,50" \
  --corr_thresholds "0.90,0.85" \
  --corr_target_dims "576" \
  --pc_remove_components "1,3,5,10" \
  --pc_remove_center \
  --include_center \
  --include_standardize \
  --chain_variance_drop_percents "25" \
  --chain_pc_remove_components "3,5" \
  --chain_center_pc_remove
