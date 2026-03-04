#!/bin/bash
#SBATCH --job-name=p10_qwen8b
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export P10_DIR="$REPO_ROOT/runs/phase10/experiment2"
export P9_DIR="$REPO_ROOT/runs/phase9/experiment2"
export MODEL="Qwen/Qwen3-Embedding-8B"
export LANGUAGES="english,french,spanish,sinhala,tamil,serbian"

cd "$REPO_ROOT" || exit 1

# =============================================================
# Step 1: Data prep — Spanish only
# =============================================================
echo "=== Step 1: Data prep (Spanish) ==="
conda run -n localLatin python scripts/run_phase10_exp2_data_prep.py \
    --out_dir "$P10_DIR" \
    --languages "spanish" \
    --seed 42

# =============================================================
# Step 2: Copy Phase 9 split CSVs for existing 5 languages
# =============================================================
echo "=== Step 2: Copy Phase 9 split CSVs ==="
for lang in english french sinhala tamil serbian; do
    cp -n "$P9_DIR/${lang}_split.csv" "$P10_DIR/${lang}_split.csv" \
        && echo "  Copied ${lang}_split.csv" \
        || echo "  Skipped ${lang}_split.csv (already exists)"
done

# =============================================================
# Step 3: Evaluate Qwen3-8B (layer-wise, all 6 languages)
# =============================================================
echo "=== Step 3: Evaluate $MODEL ==="
conda run -n localLatin python scripts/run_phase10_exp2_evaluate.py \
    --data_dir "$P10_DIR" \
    --out_dir "$P10_DIR" \
    --models "$MODEL" \
    --languages "$LANGUAGES" \
    --layers "0-50" \
    --half_precision \
    --batch_size 16 \
    --trust_remote_code \
    --phase9_results_csv "$P9_DIR/results.csv"

# =============================================================
# Step 4: Contingency (last-token + OT at best layer)
# =============================================================
echo "=== Step 4: Contingency ==="
pip install pot -q
conda run -n localLatin python scripts/run_phase10_exp2_contingency.py \
    --data_dir "$P10_DIR" \
    --results_csv "$P10_DIR/results.csv" \
    --out_dir "$P10_DIR" \
    --half_precision \
    --batch_size 16 \
    --trust_remote_code \
    --phase9_results_csv "$P9_DIR/results.csv"

# =============================================================
# Step 5: Diagnostics (ROUGE-L + BLEU-1, 6 languages)
# =============================================================
echo "=== Step 5: Diagnostics ==="
conda run -n localLatin python scripts/run_phase9_exp2_diagnostics.py \
    --data_dir "$P10_DIR" \
    --out_dir "$P10_DIR" \
    --languages "$LANGUAGES"

# =============================================================
# Step 6: Bias analysis
# =============================================================
echo "=== Step 6: Bias analysis ==="
conda run -n localLatin python scripts/run_phase10_exp2_bias.py \
    --results_csv "$P10_DIR/results.csv" \
    --data_dir "$P10_DIR" \
    --cache_dir "$P10_DIR/cache" \
    --out_dir "$P10_DIR" \
    --languages "$LANGUAGES" \
    --methods "baseline_mean,sif_abtt_optimal,whitening"

# =============================================================
# Step 7: Visualization (Figs 1-10)
# =============================================================
echo "=== Step 7: Visualization ==="
conda run -n localLatin python scripts/run_phase10_exp2_visualize.py \
    --results_csv "$P10_DIR/results.csv" \
    --diagnostics_csv "$P10_DIR/diagnostics.csv" \
    --contingency_csv "$P10_DIR/contingency_results.csv" \
    --bias_csv "$P10_DIR/bias_results.csv" \
    --cache_dir "$P10_DIR/cache" \
    --out_dir "$P10_DIR" \
    --dpi 300

echo "=== Phase 10 Exp 2 (Qwen3-8B) Complete ==="
