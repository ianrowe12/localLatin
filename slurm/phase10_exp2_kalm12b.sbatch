#!/bin/bash
#SBATCH --job-name=p10_kalm12b
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --account=beto-delta-gpu

# NOTE: Submit ONLY after the Qwen3-8B job (phase10_exp2_qwen8b.sbatch) has
# completed its evaluate step, to avoid race conditions on shared CSVs.
# Recommended:  sbatch --dependency=afterok:<qwen8b_jobid> phase10_exp2_kalm12b.sbatch

module reset
module load miniforge3-python

export REPO_ROOT="/u/irowerojas/localLatin"
export P10_DIR="$REPO_ROOT/runs/phase10/experiment2"
export P9_DIR="$REPO_ROOT/runs/phase9/experiment2"
export MODEL="tencent/KaLM-Embedding-Gemma3-12B-2511"
export LANGUAGES="english,french,spanish,sinhala,tamil,serbian"

cd "$REPO_ROOT" || exit 1

# Steps 1-2 (data prep + cp) are skipped: CSVs already exist from Qwen8B job.
# Guard against re-running in case of accidental double-submit:
if [ ! -f "$P10_DIR/spanish_split.csv" ]; then
    echo "=== Step 1 (guard): Data prep (Spanish) ==="
    conda run -n localLatin python scripts/run_phase10_exp2_data_prep.py \
        --out_dir "$P10_DIR" \
        --languages "spanish" \
        --seed 42
fi

for lang in english french sinhala tamil serbian; do
    cp -n "$P9_DIR/${lang}_split.csv" "$P10_DIR/${lang}_split.csv" 2>/dev/null || true
done

# =============================================================
# Step 3: Evaluate KaLM-Gemma3-12B (layer-wise, all 6 languages)
# The dedup guard in evaluate.py prevents re-running Qwen3-8B rows.
# =============================================================
echo "=== Step 3: Evaluate $MODEL ==="
conda run -n localLatin python scripts/run_phase10_exp2_evaluate.py \
    --data_dir "$P10_DIR" \
    --out_dir "$P10_DIR" \
    --models "$MODEL" \
    --languages "$LANGUAGES" \
    --layers "0-50" \
    --half_precision \
    --batch_size 8 \
    --trust_remote_code \
    --phase9_results_csv "$P9_DIR/results.csv"

# =============================================================
# Step 4: Contingency
# =============================================================
echo "=== Step 4: Contingency ==="
pip install pot -q
conda run -n localLatin python scripts/run_phase10_exp2_contingency.py \
    --data_dir "$P10_DIR" \
    --results_csv "$P10_DIR/results.csv" \
    --out_dir "$P10_DIR" \
    --half_precision \
    --batch_size 8 \
    --trust_remote_code \
    --phase9_results_csv "$P9_DIR/results.csv"

# Step 5 (diagnostics) skipped: already run in Qwen8B job.
# Guard re-runs in case of need:
if [ ! -f "$P10_DIR/diagnostics.csv" ]; then
    echo "=== Step 5 (guard): Diagnostics ==="
    conda run -n localLatin python scripts/run_phase9_exp2_diagnostics.py \
        --data_dir "$P10_DIR" \
        --out_dir "$P10_DIR" \
        --languages "$LANGUAGES"
fi

# =============================================================
# Step 6: Bias analysis (re-run to include both models)
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
# Step 7: Visualization (re-run to include both models)
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

echo "=== Phase 10 Exp 2 (KaLM-Gemma3-12B) Complete ==="
