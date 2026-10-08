#!/bin/bash
# Spine D2 (approved 2026-10-08): t5-efficient-base through p2x2_panel.py, and the projection
# (ABTT, train-only fit) on the four controls. CPU only.
#
#   d2_controls.py --stage config      pinned config / weights / card / gin of the three T5
#                                      checkpoints; exit 2 if t5-efficient-base is not the
#                                      original layout pretrained on C4 only
#   d2_controls.py --stage extract     five models on CPU, one pass for all 12 layers
#   d2_controls.py --stage panel       p2x2_panel.py --stage compute (gates a-d) and the
#                                      comparability check against the committed P2x2 rows
#   d2_controls.py --stage projection  baseline / ABTT D=10 / ABTT D on train, layers 1-12;
#                                      hard gate against the published panel cells
#   d2_controls.py --stage render      d2_facts.md and the two appendix tables
#
# Timing: pilot job 22753558 (16 cores, 2:30) measured about 10 s per layer for the three
# projection methods. Full run, job 22753661 (16 cores): 39:44 elapsed, MaxRSS 5.4 GB
# (extraction 14 min, panel 3, tokenizer sensitivity 5, projection 19); 70 min reserved, which
# can drop to 50.
# The HF cache must already hold the five pinned snapshots (download on a login node; the job
# runs with HF_HUB_OFFLINE=1).
#
# Submit from the checkout that holds the code and the corpus (CODE_ROOT); the paper's panel
# caches are read from REPO_ROOT. Every output goes under CODE_ROOT:
#   runs/active/reframe/d2/{d2_config_check.csv,p2x2_layers.csv,p2x2_repro.csv,d2_repro.csv,
#                           d2_projection.csv,d2_facts.md}   (git add -f)
#   runs/active/reframe/d2/bases/                             (vectors, not committed)
#   overleaf_drafts/tables/{d2_t5_efficient.tex,d2_controls_abtt.tex}
# PILOT=1 runs the config stage, a 64-passage extraction of one T5 and one encoder-only model,
# and the gated projection at layers 6 and 12 of the four gate models, to time the stages.
#
#SBATCH --job-name=d2_controls
#SBATCH --partition=cpu
#SBATCH --account=beto-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:10:00
#SBATCH --output=slurm/logs/d2_controls_%j.out
#SBATCH --error=slurm/logs/d2_controls_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/u/irowerojas/localLatin}"
REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"
PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
BASES_ROOT="$REPO_ROOT/runs/active/resubmit_bases"
D2="scripts/paper/reframe/d2_controls.py"

cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs
echo "code: $CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse --short HEAD)) caches: $REPO_ROOT"
$PY -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'threads', torch.get_num_threads())"

stamp () { echo "=== $1: $(date +%T) (${SECONDS}s) ==="; }

if [ "${PILOT:-0}" = "1" ]; then
  stamp "pilot config"
  $PY "$D2" --stage config
  stamp "pilot extract T5-base, 64 passages"
  $PY "$D2" --stage extract --models T5-base --limit 64 --threads "$THREADS"
  stamp "pilot extract LaBERTa, 64 passages"
  $PY "$D2" --stage extract --models LaBERTa --limit 64 --threads "$THREADS"
  stamp "pilot projection, the four gate models at layers 6 and 12, gated"
  $PY "$D2" --stage projection --bases_root "$BASES_ROOT" --models --layers 6 12 \
    --out_dir runs/active/reframe/d2/pilot
  stamp "pilot done"
else
  stamp "config"
  $PY "$D2" --stage config
  stamp "extract"
  $PY "$D2" --stage extract --threads "$THREADS"
  stamp "panel"
  $PY "$D2" --stage panel --bases_root "$BASES_ROOT"
  stamp "tokenizer sensitivity: T5-efficient-base read with the T5-base tokenizer"
  $PY "$D2" --stage extract --models T5-efficient-base --tokenizer_from T5-base \
    --bases runs/active/reframe/d2/bases_t5tok --threads "$THREADS"
  $PY scripts/paper/reframe/p2x2_panel.py --stage compute --models T5-efficient-base \
    --p2x2_bases runs/active/reframe/d2/bases_t5tok --out_dir runs/active/reframe/d2/t5tok
  stamp "projection"
  $PY "$D2" --stage projection --bases_root "$BASES_ROOT"
  stamp "render"
  $PY "$D2" --stage render
  stamp "done"
fi

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true
