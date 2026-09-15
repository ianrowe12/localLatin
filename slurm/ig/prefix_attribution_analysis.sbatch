#!/bin/bash
#SBATCH --job-name=prefix_attr_211
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/prefix_attr_211_%j.out
#SBATCH --error=slurm/logs/prefix_attr_211_%j.err

# Issue #211, part 2. How much attribution mass lands on Latin prefix pieces
# and on high-frequency subwords, per model / variant / view, and what changes
# when the pieces are aggregated to whole words.
#
# CPU only: every number comes from per-token vectors already stored in the
# NPZs plus the tokenizers, so no model is loaded and no GPU is needed.
# A full pass is ~27k artifact reads and six corpus tokenisations; the timed
# run took well under the reservation below.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/ig:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

OUT_DIR="${OUT_DIR:-docs/research/data}"
PER_PAIR_DIR="${PER_PAIR_DIR:-runs/active/ig_examples/prefix_attribution}"

$PY scripts/ig/run_prefix_attribution_analysis.py \
    --run both \
    --artifacts_root "${ARTIFACTS_ROOT:-runs/active/ig_examples}" \
    --pos200_root "${POS200_ROOT:-runs/active/ig_examples_200pos_v1}" \
    --corpus "${CORPUS:-data/canon_labelled}" \
    --variants raw sif_abtt \
    --out_dir "$OUT_DIR" \
    --per_pair_dir "$PER_PAIR_DIR"

echo "done: $(date -Is)"
