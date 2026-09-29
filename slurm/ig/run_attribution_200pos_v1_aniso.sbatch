#!/bin/bash
#SBATCH --job-name=attr_200pos_aniso
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=01:25:00
#SBATCH --output=slurm/logs/attr_200pos_aniso_%j.out
#SBATCH --error=slurm/logs/attr_200pos_aniso_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# Issue #227. The benchmark v1 attribution run of record
# (slurm/ig/run_attribution_200pos_v1.sbatch, job 22070707) repeated at the most
# anisotropic layers instead of the operational ones:
#
#   LaTa 8, PhilTa 6, mT5-base 5   (was LaTa 7, PhilTa 1, mT5-base 1)
#
# These are the label-free mechanism-check layers the paper names: the argmax
# of the top-PC variance share per model
# (runs/active/resubmit/layer_diagnostics/layer_rule_candidates.csv,
# diagnostic_layer_by_max_pc1_dominance). Ian approved this one GPU run on
# 2026-09-22.
#
# Everything else is the v1 job, command for command: the same 600 pairs (the
# examples CSV is the v1 sampler rerun with the same seed and
# --layer_overrides, and differs from v1's only in its ``layer`` column; the
# pair-manifest digest is v1's), IG at 40 steps, MaRC with the same
# hyperparameters and seed, token filter ``tokenizer_empty``, D = 10, three
# concurrent per-model streams on one A100. The layer enters through the
# examples CSV's ``layer`` column (IG target, stored hidden states, MaRC
# optimiser) and through the PC file, which is refit train-only at the new
# layer before submission:
#
#   python scripts/ig/refit_pcs_for_attribution.py \
#     --bases_root runs/active/resubmit_bases/phase9_bases \
#     --pooling hidden_mean_tokempty \
#     --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
#     --pc_root runs/active/ig_examples_200pos_v1_aniso/pcs --d 10 \
#     --layer_overrides bowphs_LaTa=8 bowphs_PhilTa=6 google_mt5-base=5
#
# The metrics stage's hidden backend reads the layer, the hidden states and the
# PCs from each NPZ, so it follows automatically.
#
# Output goes to runs/active/ig_examples_200pos_v1_aniso/. The v1 run of record
# is never written to. Time: v1 took 58:45, of which MaRC was about 41 to 43
# minutes per stream. Per-step cost does not depend on the layer (every step
# runs the full encoder), but early stopping does. LaTa, the slowest stream,
# used 69% of its step budget at layer 7; at 100% its stream would take about
# 77 minutes, so the v1 reservation of 1:25:00 is kept rather than cut.
#
# CODE_ROOT is where the pipeline code is read from (pass the worktree the job
# was prepared in to pin it); REPO_ROOT is where data and runs live.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
CODE_ROOT="${CODE_ROOT:-$REPO_ROOT}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts/ig:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-runs/active/ig_examples_200pos_v1_aniso}"
PC_ROOT="${PC_ROOT:-$OUT_ROOT/pcs}"
EXAMPLES_CSV="${EXAMPLES_CSV:-$OUT_ROOT/positive200_examples.csv}"
ARTIFACTS_DIR="$OUT_ROOT/artifacts"
MARC_OUT_DIR="$OUT_ROOT/retrieval_mark/artifacts"
STREAM_LOG_DIR="$OUT_ROOT/stream_logs"
TOKEN_FILTER="${TOKEN_FILTER:-tokenizer_empty}"
IG_STEPS="${IG_STEPS:-40}"
MARC_STEPS="${MARC_STEPS:-200}"
LAYER_OVERRIDES=("bowphs/LaTa=8" "bowphs/PhilTa=6" "google/mt5-base=5")

MODELS=("bowphs/LaTa" "bowphs/PhilTa" "google/mt5-base")

case "$OUT_ROOT" in
    *ig_examples_200pos_v1|*ig_examples_200pos_v1/)
        echo "FAIL: OUT_ROOT is the v1 run of record; this job never writes there" >&2
        exit 1 ;;
esac

mkdir -p "$ARTIFACTS_DIR" "$MARC_OUT_DIR" "$STREAM_LOG_DIR"

echo "=== Issue #227 attribution environment ==="
echo "repo=$REPO_ROOT"
echo "code=$CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown))"
echo "out_root=$OUT_ROOT"
echo "pc_root=$PC_ROOT"
echo "examples_csv=$EXAMPLES_CSV"
echo "token_filter=$TOKEN_FILTER"
echo "layer_overrides=${LAYER_OVERRIDES[*]}"
echo "start=$(date)"
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== Same pairs as v1 ($(date)) ==="
$PY "$CODE_ROOT/scripts/ig/pair_manifest_digest.py" \
    --examples_csv "$EXAMPLES_CSV" \
    --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
    --expect 4fcdaa2be99ebcb279c16d47ce2af158411727df27c5b86414404780a43591c8

echo ""
echo "=== Dry-run IG paths and PC availability ($(date)) ==="
$PY "$CODE_ROOT/scripts/_archive/run_phase12e_pair_explanations.py" \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$OUT_ROOT" \
    --token_filter "$TOKEN_FILTER" \
    --models "${MODELS[@]}" \
    --dry_run \
    --dry_run_require_pcs | tail -n 3

run_stream () {
    local model="$1"
    local slug="${model//\//_}"
    local log="$STREAM_LOG_DIR/${slug}.log"

    {
        echo "=== [$slug] IG artifact NPZs ($(date)) ==="
        $PY "$CODE_ROOT/scripts/_archive/run_phase12e_pair_explanations.py" \
            --examples_csv "$EXAMPLES_CSV" \
            --pc_root "$PC_ROOT" \
            --out_dir "$OUT_ROOT" \
            --token_filter "$TOKEN_FILTER" \
            --models "$model" \
            --n_steps "$IG_STEPS" \
            --trust_remote_code \
            --skip_existing

        echo "=== [$slug] MaRC mask optimization ($(date)) ==="
        $PY -u "$CODE_ROOT/scripts/ig/run_retrieval_mark_pair_examples.py" \
            --examples_csv       "$EXAMPLES_CSV" \
            --artifacts_in_dir   "$ARTIFACTS_DIR" \
            --artifacts_out_dir  "$MARC_OUT_DIR" \
            --models "$model" \
            --steps "$MARC_STEPS" \
            --lr 0.1 \
            --lambda_sparsity 0.01 \
            --gamma_tv 0.001 \
            --init_mask_logit 2.197 \
            --early_stop_thresh 0.01 \
            --early_stop_min_steps 50 \
            --token_filter "$TOKEN_FILTER" \
            --max_length 256 \
            --seed 0 \
            --trust_remote_code \
            --require_cuda \
            --skip_existing

        echo "=== [$slug] done ($(date)) ==="
    } >"$log" 2>&1
}

echo ""
echo "=== Launching ${#MODELS[@]} per-model streams ($(date)) ==="
pids=()
for model in "${MODELS[@]}"; do
    run_stream "$model" &
    pids+=("$!")
    echo "  stream $model pid ${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo ""
echo "=== Stream logs ($(date)) ==="
for model in "${MODELS[@]}"; do
    slug="${model//\//_}"
    echo "----- $slug -----"
    tail -n 25 "$STREAM_LOG_DIR/${slug}.log"
done

if [[ "$status" != "0" ]]; then
    echo "FAIL: at least one stream exited non-zero; see $STREAM_LOG_DIR" >&2
    exit 1
fi

echo ""
echo "=== Merge MaRC sidecars into canonical NPZs ($(date)) ==="
$PY -u "$CODE_ROOT/scripts/ig/merge_retrieval_mark_into_canonical.py" \
    --examples_csv    "$EXAMPLES_CSV" \
    --canonical_dir   "$ARTIFACTS_DIR" \
    --sidecar_dir     "$MARC_OUT_DIR" \
    --update_methods_available

echo ""
echo "=== Persist attribution method matrices ($(date)) ==="
$PY "$CODE_ROOT/scripts/resubmit/persist_attribution_methods.py" \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== Build manifest ($(date)) ==="
$PY "$CODE_ROOT/scripts/ig/build_attribution_run_manifest.py" \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --artifacts_root "$ARTIFACTS_DIR" \
    --retrieval_mark_root "$MARC_OUT_DIR" \
    --metrics_root "$OUT_ROOT/attribution_metrics_draws20" \
    --out_json "$OUT_ROOT/manifest.json" \
    --out_inventory_csv "$OUT_ROOT/artifact_inventory.csv" \
    --expected_n_per_model 200 \
    --expected_models "${MODELS[@]}" \
    --layer_overrides "${LAYER_OVERRIDES[@]}"

echo ""
echo "=== Assert every pair has a complete artifact ($(date)) ==="
$PY "$CODE_ROOT/scripts/ig/verify_attribution_artifacts.py" \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_root "$ARTIFACTS_DIR"

echo ""
echo "=== Done ($(date)) ==="
