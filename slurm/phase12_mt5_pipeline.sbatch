#!/bin/bash
#SBATCH --job-name=ph12_mt5
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH -o slurm/logs/phase12_mt5_%j.out
#SBATCH -e slurm/logs/phase12_mt5_%j.err

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

MODEL="google/mt5-base"
SLUG="google_mt5-base"
LAYERS="1,4,8,12"

echo "========================================="
echo "MT5-base Full Pipeline"
echo "========================================="

# ── Step 1: Extract hidden-state embeddings ──
# Output goes to phase9_bases/{slug}/hidden_mean/ to match PCA script expectations
EMB_DIR="$REPO_ROOT/runs/phase9_bases/${SLUG}/hidden_mean"
if [ ! -f "$EMB_DIR/hidden_layer12_embeddings.npy" ]; then
    echo ">>> Step 1: Extract embeddings (layers 1-12)"
    conda run -n localLatin python src/extract_hidden_cli.py \
        --meta_csv "$REPO_ROOT/runs/phase9_bases/bowphs_LaTa/hidden_mean/meta.csv" \
        --runs_root "$REPO_ROOT/runs/phase9_bases" \
        --run_dir "$EMB_DIR" \
        --model_name "$MODEL" \
        --layers "1-12" \
        --pooling mean \
        --max_length 512 \
        --batch_size 8
else
    echo ">>> Step 1: Embeddings already exist, skipping"
fi

# ── Step 2: Compute PCs ──
PC_DIR="$REPO_ROOT/runs/phase12/pcs/${SLUG}"
if [ ! -f "$PC_DIR/layer12_pcs.npz" ]; then
    echo ">>> Step 2: Compute PCs from train embeddings"
    conda run -n localLatin python scripts/run_phase12_prepare_pcs.py \
        --split_csv "$REPO_ROOT/runs/phase9/phase9_split.csv" \
        --runs_root "$REPO_ROOT/runs" \
        --out_dir "$REPO_ROOT/runs/phase12/pcs" \
        --D_max 10 \
        --models "$MODEL"
else
    echo ">>> Step 2: PCs already exist, skipping"
fi

# ── Step 3: Phase 12c Attribution ──
ATTR_DIR="$REPO_ROOT/runs/phase12c/attributions/${SLUG}"
if [ ! -f "$ATTR_DIR/layer12_retrieval_attr.npz" ]; then
    echo ">>> Step 3: Run Phase 12c retrieval attribution"
    conda run -n localLatin python scripts/run_phase12c_retrieval_attribution.py \
        --model_name "$MODEL" \
        --model_type t5 \
        --pc_dir "$PC_DIR" \
        --pairs_csv "$REPO_ROOT/runs/phase12c/retrieval_pairs.csv" \
        --layers "$LAYERS" \
        --out_dir "$ATTR_DIR" \
        --D 10 \
        --n_steps 50
else
    echo ">>> Step 3: Attribution already done, skipping"
fi

# ── Step 4: Phase 12d Filtering Check ──
echo ">>> Step 4: Run Phase 12d filtering sanity check"
conda run -n localLatin python scripts/run_phase12d_filtering_sanity_check.py \
    --model_name "$MODEL" \
    --model_type t5 \
    --layers "$LAYERS" \
    --D 10

echo "========================================="
echo "MT5-base pipeline complete!"
echo "========================================="
