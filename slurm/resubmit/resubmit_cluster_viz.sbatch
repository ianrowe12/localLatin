#!/bin/bash
#SBATCH --job-name=cluster_viz
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/resubmit_cluster_viz_%j.out
#SBATCH --error=slurm/logs/resubmit_cluster_viz_%j.err

module reset
module load miniforge3-python

# Script location: the directory where sbatch was submitted.
# After merge into main, this resolves to /u/irowerojas/localLatin.
# Before merge (worktree execution), it resolves to the worktree path.
export SCRIPT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Data + figure output always live in the main repo (runs/ is gitignored, so
# worktrees don't have copies; the main repo is authoritative).
export DATA_ROOT="/u/irowerojas/localLatin"
cd "$SCRIPT_ROOT" || exit 1
mkdir -p "$SCRIPT_ROOT/slurm/logs"

echo "SCRIPT_ROOT=$SCRIPT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"

echo "=== Install UMAP if missing ==="
conda run -n localLatin pip install --quiet "umap-learn>=0.5.5"

echo "=== 2D cluster projections (t-SNE + UMAP) ==="
conda run -n localLatin python "$SCRIPT_ROOT/scripts/resubmit/visualize_clusters_2d.py" \
  --split_csv "$DATA_ROOT/runs/active/resubmit/data/phase_resubmit_split.csv" \
  --bases_root "$DATA_ROOT/runs/active/resubmit_bases/phase9_bases" \
  --best_layer_csv "$DATA_ROOT/runs/active/resubmit/results/taskA_per_model_summary.csv" \
  --intermediate_dir "$DATA_ROOT/runs/active/resubmit/cluster_viz" \
  --figures_dir "$DATA_ROOT/overleaf_drafts/figures"

echo "=== Done ==="
