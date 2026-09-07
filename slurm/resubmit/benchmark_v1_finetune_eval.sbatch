#!/bin/bash
#SBATCH --job-name=bv1_ft_eval
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/benchmark_v1_finetune_eval_%j.out
#SBATCH --error=slurm/logs/benchmark_v1_finetune_eval_%j.err

# Rescore the fine-tuning reference ceiling (#123) on the corrected benchmark v1
# split (#113). Scoring only: the fine-tuned encoder and its cached embeddings
# are unchanged, so no GPU and no retraining. Measured at 3m18s on a cpu node.

set -euo pipefail

# CODE_ROOT holds the scripts (a worktree before merge); DATA_ROOT holds runs/,
# which is gitignored and therefore only ever complete in the main checkout.
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$CODE_ROOT}"
cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export PYTHONUNBUFFERED=1

$PY scripts/resubmit/finetune_lata_ceiling.py \
  --split_csv "$DATA_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv" \
  --data_root "$CODE_ROOT" \
  --model_name "bowphs/LaTa" \
  --ft_model_label "bowphs/LaTa-ft" \
  --out_dir "$DATA_ROOT/runs/active/resubmit/finetune" \
  --bases_root "$DATA_ROOT/runs/active/resubmit_finetune_bases" \
  --baseline_bases_root "$DATA_ROOT/runs/active/resubmit_bases" \
  --baseline_results_csv "$DATA_ROOT/runs/active/resubmit/results/phase_resubmit_results.csv" \
  --results_dir "$DATA_ROOT/runs/active/resubmit/results/finetune" \
  --tex_out "$CODE_ROOT/overleaf_drafts/tables/finetune_ceiling.tex" \
  --stages evaluate,report,mseed \
  --seed 42 \
  --dev_dir_frac 0.15 \
  --layers "1-12" \
  --D_values "1,2,3,5,7,10" \
  --mseed_M 5 --mseed_base_seed 42 \
  --cpu

echo "=== All done ==="
