#!/bin/bash
#SBATCH --job-name=reframe_e3h1w
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=00:12:00
#SBATCH --output=slurm/logs/reframe_e3_h1_whiten_%j.out
#SBATCH --error=slurm/logs/reframe_e3_h1_whiten_%j.err

# Reframe experiments E3 (subspace split), H1 (ABTT D ablation) and WHITEN
# (reduced-rank PCA whitening), issue #232. CPU only: reads the cached pooled
# vectors, fits everything on the train split, writes small CSVs under
# runs/active/reframe/ and the generated tables/figure under overleaf_drafts/.
#
# Submit from the repo root (sbatch runs a spooled copy of this file, so the
# script cannot locate the repo from its own path):
#   sbatch slurm/paper/reframe_e3_h1_whiten.sbatch
# The embedding caches are gitignored; DATA_ROOT points at a checkout that has
# them (default: the HPC working copy).
#
# Sizing (pilot on 2026-09-28, 4 BLAS threads): H1 ~23 s per model-layer
# (12 metric blocks), E3 ~7 s, WHITEN ~8 s; 100 + 164 + 100 model-layers is
# ~75 CPU-minutes. Job 22534981 (first full run, no pc1 step) took 5:27 on
# 16 workers; the E3 control columns and the pc1 step add under a minute.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT" || exit 1
test -f scripts/paper/reframe/abtt_subspace_whiten.py || {
  echo "run from the repo root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
DATA_ROOT="${DATA_ROOT:-/u/irowerojas/localLatin}"
BASES="$DATA_ROOT/runs/active/resubmit_bases"
FT_BASES="$DATA_ROOT/runs/active/resubmit_finetune_bases"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"
SCRIPT=scripts/paper/reframe/abtt_subspace_whiten.py

# One BLAS thread per worker; parallelism comes from the process pool.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

echo "=== Alignment gate ==="
$PY scripts/resubmit/verify_embedding_alignment.py \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --runs_root "$BASES/phase9_bases" | tail -5

echo "=== H1: ABTT D ablation ==="
$PY "$SCRIPT" h1 --bases_root "$BASES" --workers "$WORKERS"

echo "=== E3: subspace split ==="
$PY "$SCRIPT" e3 --bases_root "$BASES" --ft_bases_root "$FT_BASES" --workers "$WORKERS"

echo "=== WHITEN: reduced-rank whitening ==="
$PY "$SCRIPT" whiten --bases_root "$BASES" --workers "$WORKERS"

echo "=== H1 robustness: train- vs test-fitted PC1 at the T5 layers ==="
$PY "$SCRIPT" pc1 --bases_root "$BASES" --workers "$WORKERS"

echo "=== Render tables, figure, facts ==="
FT_RES="$DATA_ROOT/runs/active/resubmit/results/finetune"
$PY "$SCRIPT" render \
  --ft_csv "$FT_RES/finetune_lata_layer_results.csv" \
  --ft_csv "$FT_RES/finetune_qwen3_0.6b_layer_results.csv" \
  --ft_csv "$FT_RES/finetune_kalm_mini_layer_results.csv"

echo "=== All done ==="
