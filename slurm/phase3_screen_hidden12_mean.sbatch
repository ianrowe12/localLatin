#!/bin/bash
#SBATCH --job-name=phase3_screen_h12
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
export BASE_RUN_DIR="$WORK/runs/ff1_lata_postact/run_20260125_154356"
export PHASE3_ROOT="$WORK/runs/phase3_hidden12_mean"

cd "$REPO_ROOT" || exit 1

python src/phase3_hidden12_cli.py \
  --base_run_dir "$BASE_RUN_DIR" \
  --out_root_dir "$PHASE3_ROOT" \
  --layer 12 \
  --variance_drop_percents "10,25,50,75,90" \
  --corr_thresholds "0.95,0.975,0.99" \
  --pca_components "32,64,96,128,256,384,512,640" \
  --random_drop_percents "10,25,50,75,90" \
  --random_trials 10 \
  --random_seed 13
