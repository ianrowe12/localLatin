#!/bin/bash
#SBATCH --job-name=attr_200pos_r3
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/attr_200pos_r3_%j.out
#SBATCH --error=slurm/logs/attr_200pos_r3_%j.err
#SBATCH --mail-user=ian.rowe.rojas@gmail.com
#SBATCH --mail-type=END,FAIL

# Run 3 operational 200-positive attribution bundle.
# Layer contract: LaTa L7, PhilTa L1, mT5-base L1, D=10.
# Outputs live under runs/active/ig_examples_200pos_run3_operational by default.

set -euo pipefail

module reset
module load miniforge3-python

REPO_ROOT="${REPO_ROOT:-/projects/beto/irowerojas/worktrees/run-3-agent-1}"
cd "$REPO_ROOT"
mkdir -p slurm/logs

PY="${PY:-/u/irowerojas/.conda/envs/localLatin/bin/python}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-runs/active/ig_examples_200pos_run3_operational}"
SMOKE_ROOT="${SMOKE_ROOT:-$OUT_ROOT/smoke}"
PC_ROOT="${PC_ROOT:-$OUT_ROOT/pcs}"
BASES_ROOT="${BASES_ROOT:-/projects/beto/irowerojas/localLatin/runs/active/encoder_bases}"
SPLIT_CSV="${SPLIT_CSV:-runs/phase9/phase9_split.csv}"
TOKEN_FILTER="${TOKEN_FILTER:-all}"

FULL_N_PER_MODEL="${FULL_N_PER_MODEL:-200}"
SMOKE_N_PER_MODEL="${SMOKE_N_PER_MODEL:-2}"
IG_STEPS="${IG_STEPS:-40}"
SMOKE_IG_STEPS="${SMOKE_IG_STEPS:-4}"
MARC_STEPS="${MARC_STEPS:-200}"
SMOKE_MARC_STEPS="${SMOKE_MARC_STEPS:-5}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"

MODELS=("bowphs/LaTa" "bowphs/PhilTa" "google/mt5-base")

echo "=== Run 3 attribution environment ==="
echo "repo=$REPO_ROOT"
echo "out_root=$OUT_ROOT"
echo "pc_root=$PC_ROOT"
echo "bases_root=$BASES_ROOT"
echo "split_csv=$SPLIT_CSV"
echo "token_filter=$TOKEN_FILTER"
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(0))"

echo ""
echo "=== Refit PCs through metrics-compatible mean pooling ($(date)) ==="
$PY scripts/ig/refit_pcs_for_attribution.py \
    --slugs bowphs_LaTa bowphs_PhilTa google_mt5-base \
    --bases_root "$BASES_ROOT" \
    --pooling hidden_mean \
    --split_csv "$SPLIT_CSV" \
    --pc_root "$PC_ROOT" \
    --d 10

run_bundle () {
    local root="$1"
    local n_per_model="$2"
    local ig_steps="$3"
    local marc_steps="$4"
    local require_complete="$5"
    local marc_early_stop_min_steps=50
    if (( marc_steps < marc_early_stop_min_steps )); then
        marc_early_stop_min_steps="$marc_steps"
    fi

    local examples_csv="$root/positive${n_per_model}_examples.csv"
    local artifacts_dir="$root/artifacts"
    local marc_out_dir="$root/retrieval_mark/artifacts"
    local metrics_out_dir="$root/attribution_metrics"
    local tex_out="$root/attribution_metrics.tex"
    local manifest_json="$root/manifest.json"
    local inventory_csv="$root/artifact_inventory.csv"

    mkdir -p "$root" "$artifacts_dir" "$marc_out_dir" "$metrics_out_dir"

    echo ""
    echo "=== Sample ${n_per_model} positives/model -> $examples_csv ($(date)) ==="
    $PY scripts/ig/sample_positive_test_pairs.py \
        --out_csv "$examples_csv" \
        --n_per_model "$n_per_model" \
        --models "${MODELS[@]}"

    echo ""
    echo "=== Dry-run IG paths and PC availability ($(date)) ==="
    $PY scripts/_archive/run_phase12e_pair_explanations.py \
        --examples_csv "$examples_csv" \
        --pc_root "$PC_ROOT" \
        --out_dir "$root" \
        --token_filter "$TOKEN_FILTER" \
        --models "${MODELS[@]}" \
        --max_examples "$(( n_per_model < 2 ? n_per_model : 2 ))" \
        --dry_run \
        --dry_run_require_pcs

    echo ""
    echo "=== IG artifact NPZs ($(date)) ==="
    $PY scripts/_archive/run_phase12e_pair_explanations.py \
        --examples_csv "$examples_csv" \
        --pc_root "$PC_ROOT" \
        --out_dir "$root" \
        --token_filter "$TOKEN_FILTER" \
        --n_steps "$ig_steps" \
        --trust_remote_code \
        --skip_existing

    echo ""
    echo "=== MaRC mask optimization + merge ($(date)) ==="
    $PY -u scripts/ig/run_retrieval_mark_pair_examples.py \
        --examples_csv       "$examples_csv" \
        --artifacts_in_dir   "$artifacts_dir" \
        --artifacts_out_dir  "$marc_out_dir" \
        --steps "$marc_steps" \
        --lr 0.1 \
        --lambda_sparsity 0.01 \
        --gamma_tv 0.001 \
        --init_mask_logit 2.197 \
        --early_stop_thresh 0.01 \
        --early_stop_min_steps "$marc_early_stop_min_steps" \
        --token_filter "$TOKEN_FILTER" \
        --max_length 256 \
        --seed 0 \
        --trust_remote_code \
        --require_cuda \
        --skip_existing

    $PY -u scripts/ig/merge_retrieval_mark_into_canonical.py \
        --examples_csv    "$examples_csv" \
        --canonical_dir   "$artifacts_dir" \
        --sidecar_dir     "$marc_out_dir" \
        --update_methods_available

    echo ""
    echo "=== Persist attribution method matrices ($(date)) ==="
    $PY scripts/resubmit/persist_attribution_methods.py \
        --examples_csv "$examples_csv" \
        --artifacts_dir "$artifacts_dir"

    echo ""
    echo "=== Attribution-quality metrics ($(date)) ==="
    $PY scripts/ig/run_attribution_metrics.py \
        --examples_csv   "$examples_csv" \
        --artifacts_root "$artifacts_dir" \
        --out_root       "$metrics_out_dir" \
        --tex_out        "$tex_out" \
        --trust_remote_code

    echo ""
    echo "=== Build manifest ($(date)) ==="
    local manifest_args=()
    if [[ "$require_complete" == "1" ]]; then
        manifest_args+=(--require_complete)
    fi
    $PY scripts/ig/build_attribution_run_manifest.py \
        --examples_csv "$examples_csv" \
        --pc_root "$PC_ROOT" \
        --artifacts_root "$artifacts_dir" \
        --retrieval_mark_root "$marc_out_dir" \
        --metrics_root "$metrics_out_dir" \
        --out_json "$manifest_json" \
        --out_inventory_csv "$inventory_csv" \
        --expected_n_per_model "$n_per_model" \
        --expected_models "${MODELS[@]}" \
        "${manifest_args[@]}"

    echo "Summary CSV: $metrics_out_dir/summary.csv"
    echo "Manifest: $manifest_json"
}

if [[ "$RUN_SMOKE" == "1" ]]; then
    run_bundle "$SMOKE_ROOT" "$SMOKE_N_PER_MODEL" "$SMOKE_IG_STEPS" "$SMOKE_MARC_STEPS" 1
fi

if [[ "$RUN_FULL" == "1" ]]; then
    run_bundle "$OUT_ROOT" "$FULL_N_PER_MODEL" "$IG_STEPS" "$MARC_STEPS" 1
fi

echo ""
echo "=== Done ($(date)) ==="
