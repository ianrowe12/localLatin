#!/bin/bash
#SBATCH --job-name=attr_200pair_metrics
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/attr_200pair_metrics_%j.out
#SBATCH --error=slurm/logs/attr_200pair_metrics_%j.err

# NPZs for the 200-pair run now contain pair_matrix keys for all 7 attribution
# methods (persist_attribution_methods completed on the login node). This job
# just re-runs scripts/ig/run_attribution_metrics.py on a GPU so the metrics
# computation (many forward passes per pair) finishes in reasonable time.

set -euo pipefail

module reset
module load miniforge3-python

cd /projects/beto/irowerojas/localLatin
mkdir -p slurm/logs

PY=/u/irowerojas/.conda/envs/localLatin/bin/python
export PYTHONPATH="/projects/beto/irowerojas/localLatin/src:${PYTHONPATH:-}"

echo "=== Metrics rerun on 400 pairs with all 7 attribution methods ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=',torch.cuda.get_device_name(0))"

$PY scripts/ig/run_attribution_metrics.py \
    --examples_csv   runs/active/ig_examples_200pair/random200_examples.csv \
    --artifacts_root runs/active/ig_examples_200pair/artifacts \
    --out_root       runs/active/ig_examples_200pair/attribution_metrics \
    --tex_out        overleaf_drafts/tables/attribution_metrics_200pair.tex

echo ""
echo "=== Done ($(date)) ==="
