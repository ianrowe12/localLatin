#!/bin/bash
#SBATCH --job-name=bv1_lexical
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/benchmark_v1_lexical_%j.out
#SBATCH --error=slurm/logs/benchmark_v1_lexical_%j.err

# Lexical baselines on the corrected split (issue #113, folds in the #130 run).
# Pure text retrieval, no embeddings, so CPU only.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"

$PY scripts/resubmit/lexical_baselines.py \
  --split_csv "$REPO_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv" \
  --data_root "$REPO_ROOT" \
  --out_csv "$REPO_ROOT/runs/active/resubmit/results/lexical_baselines.csv" \
  --mseed_csv "$REPO_ROOT/runs/active/resubmit/results/lexical_baselines_mseed.csv" \
  --out_tex "$REPO_ROOT/overleaf_drafts/tables/lexical_baselines.tex" \
  --base_seed 42 --M 5 --top_k 5 --workers -1

echo "=== All done ==="
