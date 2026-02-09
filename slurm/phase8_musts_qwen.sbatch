#!/bin/bash
#SBATCH --job-name=phase8_musts_qwen
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
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
    --models "Qwen/Qwen2-7B" \
    --languages "english,french,sinhala,tamil" \
    --layers "0-28" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001 \
    --out_dir "$OUT_DIR" \
    --batch_size 8 \
    --max_length 128 \
    --half_precision \
    --trust_remote_code
