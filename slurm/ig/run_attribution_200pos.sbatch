#!/bin/bash
#SBATCH --job-name=attr_200pos
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/attr_200pos_%j.out
#SBATCH --error=slurm/logs/attr_200pos_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# 200-positive (gold-similar only) attribution-metrics rerun for LaTa + PhilTa.
# Companion to scripts/ig/sample_positive_test_pairs.py.
#
# Stages (chained sequentially in one sbatch):
#   1) IG + companion artifact NPZs        scripts/_archive/run_phase12e_pair_explanations.py
#   2) MaRC mask optimization + merge      scripts/ig/run_retrieval_mark_pair_examples.py
#                                          scripts/ig/merge_retrieval_mark_into_canonical.py
#   3) Persist 6 attribution methods       scripts/resubmit/persist_attribution_methods.py
#   4) Attribution-quality metrics         scripts/ig/run_attribution_metrics.py
#
# Wall-clock budget (extrapolated from slurm/logs/attr_200pair_17773865.out + the
# follow-up persist+metrics rerun on the same 400 pairs):
#   Stage 1: ~25 min   Stage 2: ~60 min   Stage 3: ~5 min   Stage 4: ~55 min
#   Total:   ~2.5-3h   (4h SBATCH limit gives ~1h headroom)

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

EXAMPLES_CSV=runs/active/ig_examples_200pos/positive200_examples.csv
ARTIFACTS_DIR=runs/active/ig_examples_200pos/artifacts
MARC_OUT_DIR=runs/active/ig_examples_200pos/retrieval_mark/artifacts
METRICS_OUT_DIR=runs/active/ig_examples_200pos/attribution_metrics
PC_ROOT=runs/phase12_release/pcs
TEX_OUT=overleaf_drafts/tables/attribution_metrics_200pos.tex

echo "=== Preflight: GPU + CSV sanity ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=',torch.cuda.get_device_name(0))"
test -f "$EXAMPLES_CSV" || { echo "missing $EXAMPLES_CSV"; exit 1; }
wc -l "$EXAMPLES_CSV"

echo ""
echo "=== Stage 1/4: IG artifact NPZs ($(date)) ==="
$PY scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$(dirname "$ARTIFACTS_DIR")" \
    --token_filter tokenizer_empty \
    --trust_remote_code \
    --skip_existing

echo ""
echo "=== Stage 2/4: MaRC mask optimization + merge ($(date)) ==="
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
echo "=== Stage 3/4: persist 6 attribution methods into NPZs ($(date)) ==="
$PY scripts/resubmit/persist_attribution_methods.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== Stage 4/4: attribution-quality metrics ($(date)) ==="
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
