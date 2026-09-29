#!/bin/bash
# Issue #234 (reframe GEN + FT): label-free layer geometry, model x text, on CPU.
#
#   1. gen_english_sample.py  length-matched English sample (CAP court opinions via
#                             Common Pile, pinned revision, seed 42); skipped when the CSV
#                             exists unless FORCE_SAMPLE=1
#   2. gen_extract.py         mean-pooled hidden states, layers 1-12, for mT5-base, PhilTa
#                             and T5-v1.1-base on the Latin corpus and the English sample
#                             (mT5-base / PhilTa on Latin re-extracted as the reproduction check)
#   3. gen_ft_geometry.py     top-PC share / effective rank per layer, the Latin reproduction
#                             check, the fine-tuned LaTa layerwise row, tables and figure
#
# No GPU: T5-base encoders on 1,705 passages take about two minutes each on 16 cores.
# Pilot (job 22534999, 16 cores): 96 Latin passages in 4.8 s (mT5-base) and 6.4 s
# (T5-v1.1-base) of forward time plus 5-20 s model load. Full job estimate: about 25 min;
# 50 min reserved. The HF cache must already hold the three models and the seven CAP shards
# (download them on a login node: compute nodes run with HF_HUB_OFFLINE=1).
#
# Submit from the checkout that holds the code (CODE_ROOT); data and outputs live under
# REPO_ROOT (outputs: runs/active/reframe/gen/, gitignored).
#
#SBATCH --job-name=gen_ft_234
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/gen_ft_geometry_%j.out
#SBATCH --error=slurm/logs/gen_ft_geometry_%j.err

set -euo pipefail

module reset
module load miniforge3-python

CODE_ROOT="${CODE_ROOT:-/u/irowerojas/localLatin}"
REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv"
OUT="$REPO_ROOT/runs/active/reframe/gen"
ENG_CSV="$OUT/english_sample.csv"
PY="conda run --no-capture-output -n localLatin python"
mkdir -p "$OUT"

if [ ! -f "$ENG_CSV" ] || [ "${FORCE_SAMPLE:-0}" = "1" ]; then
  echo "=== English sample ==="
  $PY scripts/paper/reframe/gen_english_sample.py \
    --split_csv "$SPLIT_CSV" --data_root "$REPO_ROOT" --out_csv "$ENG_CSV" --seed 42
fi

for MODEL in google/mt5-base bowphs/PhilTa google/t5-v1_1-base; do
  for CORPUS in latin english; do
    echo "=== extract $MODEL $CORPUS ==="
    $PY scripts/paper/reframe/gen_extract.py \
      --model_name "$MODEL" --corpus "$CORPUS" --split_csv "$SPLIT_CSV" \
      --data_root "$REPO_ROOT" --english_csv "$ENG_CSV" --out_root "$OUT/bases" \
      --batch_size 8 --max_length 512 --token_filter tokenizer_empty --threads "$THREADS"
  done
done

echo "=== geometry, reproduction check, FT row ==="
$PY scripts/paper/reframe/gen_ft_geometry.py --stage compute \
  --runs_root "$REPO_ROOT/runs/active" --split_csv "$SPLIT_CSV" --out_dir "$OUT"
$PY scripts/paper/reframe/gen_ft_geometry.py --stage render --out_dir "$OUT"

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true
echo "=== Done ==="
