#!/bin/bash
# Issue #248 (reframe P2x2): mean-pooled encoder hidden states, layers 1-12, of the 2x2
# panel models on the 1,705 labelled Latin passages, with the repo's extraction CLIs and
# exactly the arguments of the paper's caches (slurm/resubmit/resubmit_extract_*.sbatch:
# --layers 1-12 --pooling mean --token_filter tokenizer_empty --max_length 512 --batch_size 8).
#
#   1. p2x2_checks.py --stage tokenization  what each new tokenizer does to the corpus
#                                           (lengths, truncation, <unk> rate, lowercasing)
#   2. p2x2_checks.py --stage st5_encoder   the hidden CLI's AutoModelForSeq2SeqLM load of
#                                           the encoder-only Sentence-T5 checkpoint gives
#                                           the T5EncoderModel encoder
#   3. extraction, src/extract_hidden_cli.py:   Sentence-T5, T5-base, T5-v1.1-base
#                  src/extract_encoder_cli.py:  LaBERTa, PhilBERTa, SPhilBERTa
#      into $REPO_ROOT/runs/active/reframe/p2x2/bases/<slug>/hidden_mean_tokempty/
#   4. reproduction pair: LaTa (hidden CLI) and LaBSE (encoder CLI) re-extracted into
#      .../p2x2/repro_bases/<slug>/hidden_mean_tokempty/
#   5. p2x2_checks.py --stage verify        shapes, dtype, finiteness, meta.csv coverage,
#                                           and the reproduction pair against the existing
#                                           caches in runs/active/resubmit_bases/phase9_bases
#
# A model whose 12 layer files and meta.csv exist is skipped unless FORCE=1. A failing
# step does not stop the others; the job lists the failures and exits non-zero at the end.
#
# Timing (job 22572972, one A40): 96 to 195 s per model, 21 min in all for the eight
# extractions and the checks; 45 min reserved. The CLIs rerun the encoder once per layer.
# A rerun that skips every extraction takes about one minute (job 22574753). The HF cache
# must already hold the eight models (download on a login node: the job runs with
# HF_HUB_OFFLINE=1).
#
# Submit from the checkout that holds the code and the corpus (CODE_ROOT; the split CSV's
# paths are relative to it); caches are read from and written under REPO_ROOT. The check
# reports (tokenization_report.{csv,md}, st5_encoder_check.csv, extraction_check.csv) go
# to CODE_ROOT/runs/active/reframe/p2x2/ so they can be committed. Overrides:
#   PY        interpreter (default: conda run -n localLatin python)
#   HF_CACHE  HF cache root; exported as HF_HOME, and TRANSFORMERS_CACHE is dropped
#             (transformers 4 prefers it over HF_HOME). Unset: the shell's own settings.
#   FORCE=1   re-extract models whose cache is complete
# Outside the repo owner's checkout, for example:
#   sbatch --account=bimc-delta-gpu --partition=gpuA40x4-interactive \
#     --export=ALL,CODE_ROOT=$PWD,REPO_ROOT=/projects/bimc/swong2/localLatin,PY=/projects/bimc/swong2/envs/localLatin/bin/python,HF_CACHE=/work/nvme/bimc/swong2/hf_cache \
#     slurm/reframe/p2x2_extract.sbatch
#
#SBATCH --job-name=p2x2_extract_248
#SBATCH --partition=gpuA100x4
#SBATCH --account=beto-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/p2x2_extract_%j.out
#SBATCH --error=slurm/logs/p2x2_extract_%j.err

set -uo pipefail

CODE_ROOT="${CODE_ROOT:-/u/irowerojas/localLatin}"
REPO_ROOT="${REPO_ROOT:-/u/irowerojas/localLatin}"
if [ -z "${PY:-}" ]; then
  module reset
  module load miniforge3-python
  PY="conda run --no-capture-output -n localLatin python"
fi
if [ -n "${HF_CACHE:-}" ]; then
  export HF_HOME="$HF_CACHE"
  unset TRANSFORMERS_CACHE
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

cd "$CODE_ROOT" || exit 1
mkdir -p slurm/logs

SPLIT_CSV="${SPLIT_CSV:-runs/active/resubmit/data/phase_resubmit_split.csv}"
P2X2="$REPO_ROOT/runs/active/reframe/p2x2"
BASES="$P2X2/bases"
REPRO="$P2X2/repro_bases"
REF="$REPO_ROOT/runs/active/resubmit_bases/phase9_bases"
CHECKS="scripts/paper/reframe/p2x2_checks.py"
FAILED=()

echo "code: $CODE_ROOT ($(git -C "$CODE_ROOT" rev-parse --short HEAD)) data: $REPO_ROOT"
echo "HF_HOME=${HF_HOME:-unset} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

# extract <hidden|encoder> <HF id> <output root>
extract () {
  local cli="$1" model="$2" root="$3"
  local run_dir="$root/${model//\//_}/hidden_mean_tokempty"
  local n_done
  n_done=$(ls "$run_dir" 2>/dev/null | grep -c -E '^hidden_layer[0-9]+_embeddings\.npy$' || true)
  if [ "${FORCE:-0}" != "1" ] && [ "$n_done" = "12" ] && [ -f "$run_dir/meta.csv" ]; then
    echo "=== $model: complete in $run_dir, skipped (FORCE=1 re-extracts) ==="
    return 0
  fi
  local args=(--meta_csv "$SPLIT_CSV" --runs_root "$root" --run_dir "$run_dir"
              --model_name "$model" --layers 1-12 --pooling mean
              --token_filter tokenizer_empty --max_length 512 --batch_size 8)
  local script=src/extract_hidden_cli.py
  if [ "$cli" = "encoder" ]; then
    script=src/extract_encoder_cli.py
    args+=(--repr hidden)
  fi
  echo "=== extract $model ($cli CLI) -> $run_dir ==="
  local t0=$SECONDS
  if $PY "$script" "${args[@]}"; then
    echo "=== $model done in $((SECONDS - t0))s ==="
  else
    echo "=== $model FAILED after $((SECONDS - t0))s ==="
    FAILED+=("extract:$model")
  fi
}

echo "=== tokenization report ==="
$PY "$CHECKS" --stage tokenization --split_csv "$SPLIT_CSV" || FAILED+=("tokenization")

echo "=== Sentence-T5 encoder load check ==="
$PY "$CHECKS" --stage st5_encoder --split_csv "$SPLIT_CSV" || FAILED+=("st5_encoder")

extract hidden  sentence-transformers/sentence-t5-base "$BASES"
extract hidden  google-t5/t5-base                      "$BASES"
extract hidden  google/t5-v1_1-base                    "$BASES"
extract encoder bowphs/LaBerta                         "$BASES"
extract encoder bowphs/PhilBerta                       "$BASES"
extract encoder bowphs/SPhilBerta                      "$BASES"
extract hidden  bowphs/LaTa                            "$REPRO"
extract encoder sentence-transformers/LaBSE            "$REPRO"

echo "=== verify ==="
$PY "$CHECKS" --stage verify --split_csv "$SPLIT_CSV" \
  --bases_root "$BASES" --repro_root "$REPRO" --ref_root "$REF" || FAILED+=("verify")

echo "=== Accounting ==="
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,JobName%22,State,Elapsed,Timelimit,AllocTRES%50,TotalCPU || true

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== FAILED: ${FAILED[*]} ==="
  exit 1
fi
echo "=== Done ==="
