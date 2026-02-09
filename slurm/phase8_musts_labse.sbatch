#!/bin/bash
#SBATCH --job-name=phase8_musts_labse
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
conda run -n localLatin python -m pip install -r /u/irowerojas/localLatin/requirements.txt

export REPO_ROOT="/u/irowerojas/localLatin"
export OUT_DIR="$REPO_ROOT/runs/phase8_results"

cd "$REPO_ROOT" || exit 1

conda run -n localLatin python scripts/run_phase8_musts_sweep.py \
    --models "sentence-transformers/LaBSE" \
    --languages "english,french,sinhala,tamil" \
    --layers "0-12" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001 \
    --out_dir "$OUT_DIR" \
    --batch_size 32 \
    --max_length 128
