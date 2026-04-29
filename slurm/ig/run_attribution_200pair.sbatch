#!/bin/bash
#SBATCH --job-name=attr_200pair
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --output=slurm/logs/attr_200pair_%j.out
#SBATCH --error=slurm/logs/attr_200pair_%j.err

# Larger-sample attribution metrics rerun for LaTa + PhilTa (200 random test pairs each).
#
# Stages (all run sequentially in this one sbatch to keep the dependency simple):
#   1) IG + companion artifact NPZs via scripts/_archive/run_phase12e_pair_explanations.py
#   2) MaRC mask optimization + merge-into-canonical via scripts/ig/*
#   3) Attribution-quality metrics (IG, MaRC, etc.) via scripts/ig/run_attribution_metrics.py
#
# Expected wall-clock (extrapolated from the 80-pair full pipeline: 15+31+17 = ~63 min):
#   Stage 1: ~75 min  (IG is ~4x work: 400 pairs vs 80)
#   Stage 2: ~150 min (MaRC is the dominant cost)
#   Stage 3: ~85 min  (metrics)
#   Total:   ~5.2h    (7h SBATCH limit has modest headroom)

set -euo pipefail

module reset
module load miniforge3-python

cd /projects/beto/irowerojas/localLatin
mkdir -p slurm/logs

PY=/u/irowerojas/.conda/envs/localLatin/bin/python
# scripts/_archive/run_phase12e_pair_explanations.py uses sys.path.append(parents[1]/"src"),
# which resolves to scripts/src (nonexistent) now that the script lives under _archive/.
# Fix by putting the real src/ on PYTHONPATH.
export PYTHONPATH="/projects/beto/irowerojas/localLatin/src:${PYTHONPATH:-}"

EXAMPLES_CSV=runs/active/ig_examples_200pair/random200_examples.csv
ARTIFACTS_DIR=runs/active/ig_examples_200pair/artifacts
MARC_OUT_DIR=runs/active/ig_examples_200pair/retrieval_mark/artifacts
METRICS_OUT_DIR=runs/active/ig_examples_200pair/attribution_metrics
PC_ROOT=runs/phase12_release/pcs
TEX_OUT=overleaf_drafts/tables/attribution_metrics_200pair.tex

echo "=== Preflight: GPU + CSV sanity ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=',torch.cuda.get_device_name(0))"
test -f "$EXAMPLES_CSV" || { echo "missing $EXAMPLES_CSV"; exit 1; }
wc -l "$EXAMPLES_CSV"

echo ""
echo "=== Stage 1/3: IG artifact NPZs ($(date)) ==="
$PY scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$(dirname "$ARTIFACTS_DIR")" \
    --token_filter tokenizer_empty \
    --trust_remote_code \
    --skip_existing

echo ""
echo "=== Stage 2/3: MaRC mask optimization + merge ($(date)) ==="
$PY -u scripts/ig/run_retrieval_mark_pair_examples.py \
    --examples_csv       "$EXAMPLES_CSV" \
    --artifacts_in_dir   "$ARTIFACTS_DIR" \
    --artifacts_out_dir  "$MARC_OUT_DIR" \
    --steps 200 \
    --lr 0.1 \
    --lambda_sparsity 0.01 \
    --gamma_tv 0.001 \
    --init_mask_logit 2.197 \
    --early_stop_thresh 0.01 \
    --early_stop_min_steps 50 \
    --token_filter tokenizer_empty \
    --max_length 256 \
    --seed 0 \
    --trust_remote_code \
    --require_cuda \
    --skip_existing

$PY -u scripts/ig/merge_retrieval_mark_into_canonical.py \
    --examples_csv    "$EXAMPLES_CSV" \
    --canonical_dir   "$ARTIFACTS_DIR" \
    --sidecar_dir     "$MARC_OUT_DIR" \
    --update_methods_available

echo ""
echo "=== Stage 3/3: attribution-quality metrics ($(date)) ==="
mkdir -p "$METRICS_OUT_DIR"
$PY scripts/ig/run_attribution_metrics.py \
    --examples_csv   "$EXAMPLES_CSV" \
    --artifacts_root "$ARTIFACTS_DIR" \
    --out_root       "$METRICS_OUT_DIR" \
    --tex_out        "$TEX_OUT"

echo ""
echo "=== Done ($(date)) ==="
echo "Summary CSV: $METRICS_OUT_DIR/summary.csv"
echo "Paper table: $TEX_OUT"
