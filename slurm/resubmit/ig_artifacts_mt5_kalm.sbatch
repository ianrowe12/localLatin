#!/bin/bash
#SBATCH --job-name=ig_mt5_kalm
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/ig_mt5_kalm_%j.out
#SBATCH --error=slurm/logs/ig_mt5_kalm_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# Issue #47 (B4) -- the single authorised GPU job for this work.
#
# Generates the two missing IG example artifact sets for the webapp highlight
# view (google/mt5-base and KaLM-mini, 20 pair examples each, 4 buckets x 5),
# then runs the three CPU post-processing passes over ALL SIX models:
#
#   1. persist_attribution_methods.py  -> baseline/abtt matrices (new models only)
#   2. persist_decoded_tokens.py       -> real token strings (all models, #47.2)
#   3. persist_sif_attribution.py      -> sif/sif_abtt variants (all models, #46)
#
# Layer contract (scripts/ig/attribution_model_config.py): mT5-base L1,
# KaLM-mini L1 (train-selected operational layer), D=10.
#
# Code comes from the issue-46-47 worktree; data/artifacts are read from and
# written to the canonical checkout so the webapp picks them up directly.

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
ARTIFACTS_DIR="$IG_ROOT/artifacts"
PC_ROOT="$DATA_ROOT/runs/phase12_release/pcs"
BASES_ROOT="$DATA_ROOT/runs/active/encoder_bases"
SPLIT_CSV="$DATA_ROOT/runs/phase9/phase9_split.csv"
SIF_SPLIT_CSV="$DATA_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
PROBS_CACHE="$IG_ROOT/sif_token_probs"

MT5="google/mt5-base"
KALM="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"

echo "=== Environment ==="
echo "code_root=$CODE_ROOT"
echo "data_root=$DATA_ROOT"
echo "examples_csv=$EXAMPLES_CSV"
date
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== [1/7] Ensure example rows exist for mT5-base + KaLM-mini ($(date)) ==="
$PY scripts/ig/select_extra_pair_examples.py \
    --models "$MT5" "$KALM" \
    --examples_csv "$EXAMPLES_CSV" \
    --split_csv "$SPLIT_CSV" \
    --bases_root "$BASES_ROOT" \
    --data_root "$DATA_ROOT/data/canon"

echo ""
echo "=== [2/7] Refit D=10 ABTT PCs for the two new models ($(date)) ==="
$PY scripts/ig/refit_pcs_for_attribution.py \
    --slugs google_mt5-base KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5 \
    --bases_root "$BASES_ROOT" \
    --pooling hidden_mean_tokempty \
    --split_csv "$SPLIT_CSV" \
    --pc_root "$PC_ROOT" \
    --d 10

echo ""
echo "=== [3/7] Dry-run IG paths / PC availability ($(date)) ==="
$PY scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$IG_ROOT" \
    --models "$MT5" "$KALM" \
    --token_filter tokenizer_empty \
    --dry_run \
    --dry_run_require_pcs

echo ""
echo "=== [4/7] GPU: IG artifact NPZs for mT5-base + KaLM-mini ($(date)) ==="
$PY -u scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$IG_ROOT" \
    --models "$MT5" "$KALM" \
    --token_filter tokenizer_empty \
    --n_steps 40 \
    --trust_remote_code \
    --skip_existing

echo ""
echo "=== [5/7] Persist baseline/abtt method matrices for the new models ($(date)) ==="
$PY -u scripts/resubmit/persist_attribution_methods.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --models "$MT5" "$KALM"

echo ""
echo "=== [6/7] Persist decoded token strings for ALL models ($(date)) ==="
$PY -u scripts/ig/persist_decoded_tokens.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --overwrite

echo ""
echo "=== [7/7] Persist sif / sif_abtt variants for ALL models ($(date)) ==="
$PY -u scripts/resubmit/persist_sif_attribution.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --split_csv "$SIF_SPLIT_CSV" \
    --repo_root "$DATA_ROOT" \
    --token_probs_cache "$PROBS_CACHE"

echo ""
echo "=== Coverage check ($(date)) ==="
$PY - "$ARTIFACTS_DIR" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
variants = ("baseline", "abtt", "sif", "sif_abtt")
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    npzs = sorted(d.glob("example*_pair_example.npz"))
    if not npzs:
        continue
    data = np.load(npzs[0], allow_pickle=False)
    have = [v for v in variants if f"pair_matrix_ig_{v}" in data]
    toks = "query_token_strings" in data
    sample = ""
    if toks:
        sample = " ".join(repr(t) for t in data["query_token_strings"][:5].tolist())
    print(f"{d.name:62s} n={len(npzs):3d} variants={have} tokens={toks} {sample}")
PYEOF

echo ""
echo "=== Done ($(date)) ==="
