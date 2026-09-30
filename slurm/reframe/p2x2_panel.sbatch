#!/bin/bash
# Issue #248 (reframe P2x2): the architecture-by-objective panel, per layer, on CPU.
#
#   p2x2_panel.py --stage compute  Task A test AUROC, train top-PC share and effective rank at
#                                  layers 1-12 of ten encoders, from cached mean-pooled vectors;
#                                  hard reproduction gate against the published cells
#   p2x2_panel.py --stage render   tab:panel_2x2, tab:p2x2_layerwise and p2x2_facts.md
#
# No GPU: the job only reads cached vectors (120 matrices of 1,705 x 768). Four models took
# 2 min 30 s on 8 cores (job 22573180, about 2 s per layer after a 50 s start-up), so ten take
# about 5 min; 15 min reserved. The vectors of the six new models come from
# slurm/reframe/p2x2_extract.sbatch, which must have finished.
#
# Submit from the checkout that holds the code (CODE_ROOT). The caches are read from REPO_ROOT;
# the split and the published reference CSVs are tracked, so they are read from CODE_ROOT, and
# every output is written into CODE_ROOT so it can be committed:
#   runs/active/reframe/p2x2/{p2x2_layers.csv,p2x2_repro.csv,p2x2_facts.md}  (git add -f)
#   overleaf_drafts/tables/{panel_2x2.tex,p2x2_layerwise.tex}
# Create slurm/logs before submitting. MODELS="LaTa PhilTa" restricts the run (a partial table);
# PY overrides the interpreter, e.g. PY=/path/to/env/bin/python.
#
#SBATCH --job-name=p2x2_panel_248
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/p2x2_panel_%j.out
#SBATCH --error=slurm/logs/p2x2_panel_%j.err

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:-/u/irowerojas/localLatin}"
REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"
PY="${PY:-conda run --no-capture-output -n localLatin python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

MODEL_ARGS=()
if [ -n "${MODELS:-}" ]; then
  # shellcheck disable=SC2206
  MODEL_ARGS=(--models $MODELS)
fi

echo "=== compute (reproduction gate) ==="
$PY scripts/paper/reframe/p2x2_panel.py --stage compute \
  --bases_root "$REPO_ROOT/runs/active/resubmit_bases" \
  --p2x2_bases "$REPO_ROOT/runs/active/reframe/p2x2/bases" \
  "${MODEL_ARGS[@]}"

echo "=== render ==="
$PY scripts/paper/reframe/p2x2_panel.py --stage render

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true
echo "=== Done ==="
