#!/bin/bash
#SBATCH --job-name=ig_demo_unlab
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/ig_demo_unlab_%j.out
#SBATCH --error=slurm/logs/ig_demo_unlab_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# Issue #53 (B10) -- attribution artifacts for the Abigail demo queries.
#
# docs/research/abigail_demo_script.md section 5 has to tell the reviewer that
# the Attribution view will not appear, because all 120 artifacts describe
# labelled canon pairs and none lines up with a live unlabelled query. This job
# generates the 8 missing (query, directory) pairs -- the four demo queries x
# {raw top-1, corrected top-1} -- so the Evidence view has real integrated
# gradients instead of the lexical fallback.
#
# Everything is keyed off the deployed variant prediction CSVs: layer, D, the
# ABTT principal components and the candidate file inside each directory are
# all reproduced from them, so the attribution explains the ranking the
# reviewer is actually looking at (checked to 5 dp in step 1).
#
# Steps 5-7 rerun over ALL artifacts, not just the new ones: adding the demo
# pairs changes which documents are excluded from the leak-free SIF corpus, so
# the token probabilities are refreshed and every artifact's sif / sif_abtt
# variant is recomputed against the same corpus.

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:-/projects/beto/irowerojas/localLatin/.claude/worktrees/agent-a53720ddc94ab2ec7}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$CODE_ROOT"
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"

IG_ROOT="$DATA_ROOT/runs/active/ig_examples"
EXAMPLES_CSV="$IG_ROOT/phase12f_examples.csv"
SUBSET_CSV="$IG_ROOT/phase12f_examples_unlabelled.csv"
ARTIFACTS_DIR="$IG_ROOT/artifacts"
PC_ROOT="$DATA_ROOT/runs/phase12_release/pcs"
UNLAB_ROOT="$DATA_ROOT/runs/active/resubmit/unlabelled"
LAB_BASES="$DATA_ROOT/runs/active/resubmit_bases/phase9_bases"
SPLIT_CSV="$DATA_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
RESULTS_CSV="$DATA_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv"
PROBS_CACHE="$IG_ROOT/sif_token_probs"

echo "=== Environment ==="
echo "code_root=$CODE_ROOT"
echo "data_root=$DATA_ROOT"
date
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== [1/7] Select the demo (query, directory) pairs ($(date)) ==="
$PY scripts/ig/select_unlabelled_demo_pairs.py \
    --examples_csv "$EXAMPLES_CSV" \
    --subset_csv "$SUBSET_CSV" \
    --unlabelled_root "$UNLAB_ROOT" \
    --labelled_bases "$LAB_BASES" \
    --split_csv "$SPLIT_CSV" \
    --results_csv "$RESULTS_CSV" \
    --data_root "$DATA_ROOT/data" \
    --pc_root "$PC_ROOT"

echo ""
echo "=== [2/7] Dry-run IG paths / PC availability ($(date)) ==="
$PY scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$SUBSET_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$IG_ROOT" \
    --token_filter tokenizer_empty \
    --dry_run \
    --dry_run_require_pcs

echo ""
echo "=== [3/7] GPU: IG artifact NPZs for the 8 demo pairs ($(date)) ==="
$PY -u scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$SUBSET_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$IG_ROOT" \
    --token_filter tokenizer_empty \
    --n_steps 40 \
    --trust_remote_code \
    --skip_existing

echo ""
echo "=== [4/7] Persist baseline/abtt method matrices (all artifacts) ($(date)) ==="
$PY -u scripts/resubmit/persist_attribution_methods.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== [5/7] Persist decoded token strings ($(date)) ==="
$PY -u scripts/ig/persist_decoded_tokens.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --overwrite

echo ""
echo "=== [6/7] Persist sif / sif_abtt variants, refreshed SIF corpus ($(date)) ==="
$PY -u scripts/resubmit/persist_sif_attribution.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --split_csv "$SPLIT_CSV" \
    --repo_root "$DATA_ROOT" \
    --token_probs_cache "$PROBS_CACHE" \
    --refresh_probs

echo ""
echo "=== [7/7] Verify the demo pairs ($(date)) ==="
$PY -u scripts/ig/verify_demo_artifacts.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== Done ($(date)) ==="
