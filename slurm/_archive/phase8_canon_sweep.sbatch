#!/bin/bash
#SBATCH --job-name=phase8_canon
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python
conda run -n localLatin python -m pip install -r /u/irowerojas/localLatin/requirements.txt

export REPO_ROOT="/u/irowerojas/localLatin"
export RUNS_ROOT="$REPO_ROOT/runs"
export OUT_DIR="$RUNS_ROOT/phase8_results"
export SPLIT_CSV="$OUT_DIR/meta_split.csv"

cd "$REPO_ROOT" || exit 1

# Create split if not exists
if [ ! -f "$SPLIT_CSV" ]; then
    conda run -n localLatin python scripts/create_canon_split.py \
        --canon_root "$REPO_ROOT/canon" \
        --output_csv "$SPLIT_CSV" \
        --test_fraction 0.2 \
        --random_seed 42
fi

# Canon sweep: LaTa, PhilTa, LaBSE
conda run -n localLatin python scripts/run_phase8_canon_sweep.py \
    --split_csv "$SPLIT_CSV" \
    --runs_root "$RUNS_ROOT" \
    --out_dir "$OUT_DIR" \
    --models "bowphs/LaTa,bowphs/PhilTa" \
    --reprs "hidden,ff1" \
    --layers_hidden "0-12" \
    --layers_ff1 "1-12" \
    --D 10 \
    --sif_a 0.001 \
    --encoder_models "sentence-transformers/LaBSE" \
    --encoder_layers "0-12"
