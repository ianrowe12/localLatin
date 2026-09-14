#!/bin/bash
#SBATCH --job-name=attr_200pos_v1
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=01:25:00
#SBATCH --output=slurm/logs/attr_200pos_v1_%j.out
#SBATCH --error=slurm/logs/attr_200pos_v1_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# Issue #141 (G6). Re-run of the 200-positive-pair attribution bundle on
# benchmark v1 (data/canon_labelled, phase_resubmit_split.csv) instead of the
# legacy phase-9 split over data/canon that run 3 sampled from.
#
# Layer contract is unchanged: LaTa L7, PhilTa L1, mT5-base L1, D = 10.
#
# Two differences from slurm/ig/run_attribution_200pos_run3_operational.sbatch,
# both forced by the corpus change and both recorded in
# docs/research/attribution_v1_resample.md:
#
#   * token_filter is ``tokenizer_empty`` rather than ``all``. canon_labelled is
#     only cached at ``hidden_mean_tokempty``, and the PCs have to be fit
#     through the same pooling the artifacts use or they clean a distribution
#     they were not fit on (the failure this pipeline already hit once, see
#     docs/analyses/cosine_inflation_investigation.md).
#   * The three models run as concurrent streams rather than one sequential
#     pass. Run 3 took 4 h 17 m in one stream, of which IG and MaRC were 101 m;
#     this job is capped at 01:25:00 and the metrics stage moved to CPU, so the
#     GPU stages are overlapped. Each stream owns its own model slug, its own
#     artifact subdirectory and its own log, so they never touch the same file.
#
# The attribution-quality metrics are NOT run here. They are CPU work under
# ``--backend hidden`` and run separately (see the doc above).

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/localLatin}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-conda run -n localLatin python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/ig:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-runs/active/ig_examples_200pos_v1}"
PC_ROOT="${PC_ROOT:-$OUT_ROOT/pcs}"
EXAMPLES_CSV="${EXAMPLES_CSV:-$OUT_ROOT/positive200_examples.csv}"
ARTIFACTS_DIR="$OUT_ROOT/artifacts"
MARC_OUT_DIR="$OUT_ROOT/retrieval_mark/artifacts"
STREAM_LOG_DIR="$OUT_ROOT/stream_logs"
TOKEN_FILTER="${TOKEN_FILTER:-tokenizer_empty}"
IG_STEPS="${IG_STEPS:-40}"
MARC_STEPS="${MARC_STEPS:-200}"

MODELS=("bowphs/LaTa" "bowphs/PhilTa" "google/mt5-base")

mkdir -p "$ARTIFACTS_DIR" "$MARC_OUT_DIR" "$STREAM_LOG_DIR"

echo "=== Issue #141 attribution environment ==="
echo "repo=$REPO_ROOT"
echo "out_root=$OUT_ROOT"
echo "pc_root=$PC_ROOT"
echo "examples_csv=$EXAMPLES_CSV"
echo "token_filter=$TOKEN_FILTER"
echo "start=$(date)"
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== Dry-run IG paths and PC availability ($(date)) ==="
$PY scripts/_archive/run_phase12e_pair_explanations.py \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --out_dir "$OUT_ROOT" \
    --token_filter "$TOKEN_FILTER" \
    --models "${MODELS[@]}" \
    --dry_run \
    --dry_run_require_pcs

# One stream per model: IG first, then MaRC on that model's own artifacts.
# MaRC only reads the NPZs its own IG pass wrote, so the chain is per model and
# the three chains are independent.
run_stream () {
    local model="$1"
    local slug="${model//\//_}"
    local log="$STREAM_LOG_DIR/${slug}.log"

    {
        echo "=== [$slug] IG artifact NPZs ($(date)) ==="
        $PY scripts/_archive/run_phase12e_pair_explanations.py \
            --examples_csv "$EXAMPLES_CSV" \
            --pc_root "$PC_ROOT" \
            --out_dir "$OUT_ROOT" \
            --token_filter "$TOKEN_FILTER" \
            --models "$model" \
            --n_steps "$IG_STEPS" \
            --trust_remote_code \
            --skip_existing

        echo "=== [$slug] MaRC mask optimization ($(date)) ==="
        $PY -u scripts/ig/run_retrieval_mark_pair_examples.py \
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
$PY -u scripts/ig/merge_retrieval_mark_into_canonical.py \
    --examples_csv    "$EXAMPLES_CSV" \
    --canonical_dir   "$ARTIFACTS_DIR" \
    --sidecar_dir     "$MARC_OUT_DIR" \
    --update_methods_available

echo ""
echo "=== Persist attribution method matrices ($(date)) ==="
$PY scripts/resubmit/persist_attribution_methods.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo ""
echo "=== Build manifest ($(date)) ==="
$PY scripts/ig/build_attribution_run_manifest.py \
    --examples_csv "$EXAMPLES_CSV" \
    --pc_root "$PC_ROOT" \
    --artifacts_root "$ARTIFACTS_DIR" \
    --retrieval_mark_root "$MARC_OUT_DIR" \
    --metrics_root "$OUT_ROOT/attribution_metrics" \
    --out_json "$OUT_ROOT/manifest.json" \
    --out_inventory_csv "$OUT_ROOT/artifact_inventory.csv" \
    --expected_n_per_model 200 \
    --expected_models "${MODELS[@]}"

# build_attribution_run_manifest.py --require_complete also demands the
# metrics summary.csv, which this run deliberately never produces: the
# model-backend metrics pass cost run 3 two and a half GPU hours and the paper
# reads summary_v2.csv from the CPU hidden-backend pass instead. The artifact
# side of completeness is therefore asserted directly.
echo ""
echo "=== Assert every pair has a complete artifact ($(date)) ==="
$PY scripts/ig/verify_attribution_artifacts.py \
    --examples_csv "$EXAMPLES_CSV" \
    --artifacts_root "$ARTIFACTS_DIR"

echo ""
echo "=== Done ($(date)) ==="
