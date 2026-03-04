#!/bin/bash
#SBATCH --job-name=phase10
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
export P8_OUT="$RUNS_ROOT/phase8_results"

cd "$REPO_ROOT" || exit 1

# =============================================================
# Part A: Phase 8 Canon Sweep for new models
# SKIPPED — already completed in job 15974425
# =============================================================

# =============================================================
# Part B: Phase 9 Exp 1 — Extract embeddings for new models
# =============================================================

echo "=== Part B: Qwen3 Extraction ==="
for POOLING in mean sif; do
    conda run -n localLatin python src/extract_encoder_cli.py \
        --meta_csv "$SPLIT_CSV" --runs_root "$RUNS_ROOT" \
        --model_name "Qwen/Qwen3-Embedding-0.6B" \
        --repr hidden --layers "0-36" --pooling "$POOLING" \
        --sif_a 0.001 --split_csv "$SPLIT_CSV" \
        --run_dir "$RUNS_ROOT/phase9_bases/Qwen_Qwen3-Embedding-0.6B/hidden_${POOLING}" \
        --trust_remote_code

    conda run -n localLatin python src/extract_encoder_cli.py \
        --meta_csv "$SPLIT_CSV" --runs_root "$RUNS_ROOT" \
        --model_name "Qwen/Qwen3-Embedding-0.6B" \
        --repr ffn_intermediate --layers "0-35" --pooling "$POOLING" \
        --sif_a 0.001 --split_csv "$SPLIT_CSV" \
        --run_dir "$RUNS_ROOT/phase9_bases/Qwen_Qwen3-Embedding-0.6B/ffn_int_${POOLING}" \
        --trust_remote_code
done

echo "=== Part B: KaLM Extraction ==="
for POOLING in mean sif; do
    conda run -n localLatin python src/extract_encoder_cli.py \
        --meta_csv "$SPLIT_CSV" --runs_root "$RUNS_ROOT" \
        --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
        --repr hidden --layers "0-24" --pooling "$POOLING" \
        --sif_a 0.001 --split_csv "$SPLIT_CSV" \
        --run_dir "$RUNS_ROOT/phase9_bases/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5/hidden_${POOLING}" \
        --trust_remote_code

    conda run -n localLatin python src/extract_encoder_cli.py \
        --meta_csv "$SPLIT_CSV" --runs_root "$RUNS_ROOT" \
        --model_name "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
        --repr ffn_intermediate --layers "0-23" --pooling "$POOLING" \
        --sif_a 0.001 --split_csv "$SPLIT_CSV" \
        --run_dir "$RUNS_ROOT/phase9_bases/KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5/ffn_int_${POOLING}" \
        --trust_remote_code
done

# =============================================================
# Part C: Phase 9 Exp 1 — Evaluate ALL models (new D optimization)
# =============================================================

echo "=== Part C: Evaluation (all 5 models, Assignment Accuracy D optimization) ==="
conda run -n localLatin python scripts/evaluate_vectors.py \
    --split_csv "$SPLIT_CSV" \
    --runs_root "$RUNS_ROOT" \
    --out_csv "$PHASE9_DIR/phase10_experiment1_results.csv" \
    --models "bowphs/LaTa,bowphs/PhilTa" \
    --encoder_models "sentence-transformers/LaBSE,Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5" \
    --D_values "1,2,3,5,7,10" \
    --sif_a 0.001

echo "=== Phase 10 Complete ==="
