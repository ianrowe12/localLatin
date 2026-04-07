#!/bin/bash
#SBATCH --job-name=phase9_exp1
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export RUNS_ROOT="$REPO_ROOT/runs"
export PHASE9_DIR="$RUNS_ROOT/phase9"
export SPLIT_CSV="$PHASE9_DIR/phase9_split.csv"
export META_CSV="$PHASE9_DIR/phase9_split.csv"

cd "$REPO_ROOT" || exit 1

# ─────────────────────────────────────────────────────────
# Step 1: Data prep — generate split + TSVs
# ─────────────────────────────────────────────────────────
echo "=== Step 1: Data Prep ==="
conda run -n localLatin python scripts/data_prep.py \
    --canon_root "$REPO_ROOT/canon" \
    --out_dir "$PHASE9_DIR" \
    --seed 42

# ─────────────────────────────────────────────────────────
# Step 2: Extract embeddings — LaTa (hidden 0-12, ff1 1-12, mean + sif)
# ─────────────────────────────────────────────────────────
echo "=== Step 2: LaTa Extraction ==="
LATA_BASE="$RUNS_ROOT/phase9_bases/bowphs_LaTa"

for POOLING in mean sif; do
    LATA_DIR="$LATA_BASE/hidden_${POOLING}"
    conda run -n localLatin python src/extract_hidden_cli.py \
        --meta_csv "$META_CSV" \
        --runs_root "$RUNS_ROOT" \
        --model_name "bowphs/LaTa" \
        --layers "0-12" \
        --pooling "$POOLING" \
        --sif_a 0.001 \
        --split_csv "$SPLIT_CSV" \
        --run_dir "$LATA_DIR"

    LATA_FF1_DIR="$LATA_BASE/ff1_${POOLING}"
    conda run -n localLatin python src/extract_ff1_cli.py \
        --meta_csv "$META_CSV" \
        --runs_root "$RUNS_ROOT" \
        --model_name "bowphs/LaTa" \
        --layers "1-12" \
        --pooling "$POOLING" \
        --sif_a 0.001 \
        --split_csv "$SPLIT_CSV" \
        --run_dir "$LATA_FF1_DIR"
done

# ─────────────────────────────────────────────────────────
# Step 3: Extract embeddings — PhilTa (hidden 0-12, ff1 1-12, mean + sif)
# ─────────────────────────────────────────────────────────
echo "=== Step 3: PhilTa Extraction ==="
PHILTA_BASE="$RUNS_ROOT/phase9_bases/bowphs_PhilTa"

for POOLING in mean sif; do
    PHILTA_DIR="$PHILTA_BASE/hidden_${POOLING}"
    conda run -n localLatin python src/extract_hidden_cli.py \
        --meta_csv "$META_CSV" \
        --runs_root "$RUNS_ROOT" \
        --model_name "bowphs/PhilTa" \
        --layers "0-12" \
        --pooling "$POOLING" \
        --sif_a 0.001 \
        --split_csv "$SPLIT_CSV" \
        --run_dir "$PHILTA_DIR"

    PHILTA_FF1_DIR="$PHILTA_BASE/ff1_${POOLING}"
    conda run -n localLatin python src/extract_ff1_cli.py \
        --meta_csv "$META_CSV" \
        --runs_root "$RUNS_ROOT" \
        --model_name "bowphs/PhilTa" \
        --layers "1-12" \
        --pooling "$POOLING" \
        --sif_a 0.001 \
        --split_csv "$SPLIT_CSV" \
        --run_dir "$PHILTA_FF1_DIR"
done

# ─────────────────────────────────────────────────────────
# Step 4: Extract embeddings — LaBSE (hidden 0-12, mean + sif)
# ─────────────────────────────────────────────────────────
echo "=== Step 4: LaBSE Extraction ==="
LABSE_BASE="$RUNS_ROOT/phase9_bases/sentence-transformers_LaBSE"

for POOLING in mean sif; do
    LABSE_DIR="$LABSE_BASE/hidden_${POOLING}"
    conda run -n localLatin python src/extract_encoder_cli.py \
        --meta_csv "$META_CSV" \
        --runs_root "$RUNS_ROOT" \
        --model_name "sentence-transformers/LaBSE" \
        --repr hidden \
        --layers "0-12" \
        --pooling "$POOLING" \
        --sif_a 0.001 \
        --split_csv "$SPLIT_CSV" \
        --run_dir "$LABSE_DIR"
done

# ─────────────────────────────────────────────────────────
# Step 5: Evaluate all embeddings
# ─────────────────────────────────────────────────────────
echo "=== Step 5: Evaluation ==="
conda run -n localLatin python scripts/evaluate_vectors.py \
    --split_csv "$SPLIT_CSV" \
    --runs_root "$RUNS_ROOT" \
    --out_csv "$PHASE9_DIR/phase9_experiment1_results.csv" \
    --models "bowphs/LaTa,bowphs/PhilTa" \
    --encoder_models "sentence-transformers/LaBSE" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001

# ─────────────────────────────────────────────────────────
# Step 6: Visualize
# ─────────────────────────────────────────────────────────
echo "=== Step 6: Visualization ==="
conda run -n localLatin python scripts/visualize_experiment_1.py \
    --results_csv "$PHASE9_DIR/phase9_experiment1_results.csv" \
    --out_dir "$PHASE9_DIR/figures" \
    --split_csv "$SPLIT_CSV" \
    --emb_dir "$RUNS_ROOT/phase9_bases"

echo "=== Phase 9 Experiment 1 Complete ==="
