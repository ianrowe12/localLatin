#!/usr/bin/env bash
#
# submit_bulk_chain.sh -- size and submit the issue #84 chunk chain.
#
# Each chunk's --time is derived, not guessed:
#
#   reserve_minutes = ceil( (remaining_pairs * s_per_pair * MARGIN) / 60 ) + SETUP
#
# MARGIN (default 1.5) covers the length distribution -- a chunk's pairs are not
# the pilot's pairs, and IG cost scales with token count. SETUP (default 10 min)
# covers what the per-pair rate excludes and the pilot measured separately: the
# O(vocab) token-keep lookup (minutes on a 250k-token vocab), the D sweeps, the
# train-corpus SIF estimate, and the registry pass.
#
# CLAUDE.md charges the *reserved* wallclock, so the sizing is also the bill.
# The chunk itself stops its pair loop 8 minutes before the limit and still
# writes its registry, so an under-estimate costs a resumed chunk, not a TIMEOUT.
#
# BUDGET: chunks are submitted in priority order while the projected
# reservations fit inside (balance - floor). The first chunk that does not fit
# is submitted with its --time *capped* at the hours still available rather
# than skipped, because the generator is resume-safe and stops its own pair loop
# cleanly: a capped chunk builds as many of its pairs as the remaining budget
# buys and leaves the rest for a later run, which is strictly better than
# leaving those GPU-hours unspent. --no-partial turns that off and skips
# instead. After a capped chunk nothing further is submitted. Every submitted
# chunk also re-checks the live balance at start and no-ops below the floor, so
# a chain that turns out more expensive than projected still stops itself.
#
# USAGE
#   bash scripts/ig/submit_bulk_chain.sh --rates 'bowphs/LaTa=2.32,...' [--dry-run]
#
# Rates are seconds per pair, measured. --dry-run prints the plan and the sbatch
# commands without submitting anything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}}"
DATA_ROOT="${DATA_ROOT:-/projects/beto/irowerojas/localLatin}"
ACCOUNT="${BUDGET_ACCOUNT:-beto-delta-gpu}"
FLOOR="${BUDGET_FLOOR:-5.0}"
MARGIN="${MARGIN:-1.5}"
SETUP_MIN="${SETUP_MIN:-10}"
RATES=""
DRY_RUN=0
DEP=""
ALLOW_PARTIAL=1
MIN_PARTIAL_H="${MIN_PARTIAL_H:-1.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rates)     RATES="$2"; shift 2 ;;
        --margin)    MARGIN="$2"; shift 2 ;;
        --setup-min) SETUP_MIN="$2"; shift 2 ;;
        --floor)     FLOOR="$2"; shift 2 ;;
        --after)     DEP="$2"; shift 2 ;;
        --no-partial) ALLOW_PARTIAL=0; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${RATES}" ]] || { echo "--rates is required" >&2; exit 2; }

# shellcheck source=../common/gpu_budget_guard.sh
source "${REPO_ROOT}/scripts/common/gpu_budget_guard.sh"

balance="$(budget_balance "${ACCOUNT}")"
available="$(awk -v b="${balance}" -v f="${FLOOR}" 'BEGIN { printf "%.2f", b - f }')"
echo "balance ${balance} h on ${ACCOUNT}; floor ${FLOOR} h; available ${available} h"
echo "margin ${MARGIN}x, setup ${SETUP_MIN} min/chunk"
echo

# Remaining pairs per model, straight from the enumerator, so the sizing counts
# what is actually left rather than the full corpus.
PLAN="$(cd "${CODE_ROOT}" && PYTHONPATH="${CODE_ROOT}/src" python - "${DATA_ROOT}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

data_root = Path(sys.argv[1])
here = Path(__file__).resolve()
sys.path.insert(0, str(Path.cwd() / "scripts" / "ig"))
sys.path.insert(0, str(Path.cwd() / "scripts" / "resubmit"))
from bulk_attribution import (  # noqa: E402
    MODEL_PRIORITY, enumerate_pairs, load_variant_frames, slug_for,
)

unl = data_root / "runs/active/resubmit/unlabelled"
artifacts = data_root / "runs/active/ig_examples/artifacts"
meta = pd.read_csv(unl / "meta_unlabelled.csv")
f2r = {str(r): i for i, r in enumerate(meta["filename"])}
fids = {str(r["filename"]): int(r["file_id"]) for _, r in meta.iterrows()}
frames = load_variant_frames(unl)
for model in MODEL_PRIORITY:
    slug = slug_for(model)
    pairs = enumerate_pairs(frames, model, f2r, fids)
    have = {p.stem for p in (artifacts / slug).glob("*.npz")} if (artifacts / slug).exists() else set()
    todo = sum(1 for p in pairs if p.artifact_path(artifacts, slug).stem not in have)
    print(f"{model}\t{slug}\t{len(pairs)}\t{todo}")
PY
)"

used_h=0
submitted=0
while IFS=$'\t' read -r model slug total todo; do
    rate="$(awk -F, -v m="${model}" '{
        for (i = 1; i <= NF; i++) {
            split($i, kv, "=")
            if (kv[1] == m) { print kv[2]; exit }
        }
    }' <<< "${RATES}")"
    if [[ -z "${rate}" ]]; then
        echo "SKIP ${model}: no rate given in --rates"
        continue
    fi
    if [[ "${todo}" -eq 0 ]]; then
        echo "SKIP ${model}: 0 pairs left to build"
        continue
    fi
    minutes="$(awk -v n="${todo}" -v r="${rate}" -v m="${MARGIN}" -v s="${SETUP_MIN}" \
        'BEGIN { printf "%d", int((n * r * m) / 60 + 0.999) + s }')"
    hours="$(awk -v x="${minutes}" 'BEGIN { printf "%.2f", x / 60 }')"
    fits="$(awk -v u="${used_h}" -v h="${hours}" -v a="${available}" 'BEGIN { print (u + h <= a) ? 1 : 0 }')"

    printf '%-62s pairs=%5d/%5d  rate=%.3f s  ->  %3d min (%.2f h)  ' \
        "${model}" "${todo}" "${total}" "${rate}" "${minutes}" "${hours}"
    last_chunk=0
    if [[ "${fits}" -eq 0 ]]; then
        left="$(awk -v u="${used_h}" -v a="${available}" 'BEGIN { printf "%.2f", a - u }')"
        if [[ "${ALLOW_PARTIAL}" -eq 0 ]] \
           || awk -v l="${left}" -v m="${MIN_PARTIAL_H}" 'BEGIN { exit !(l < m) }'; then
            echo "DOES NOT FIT (${used_h} h planned, ${left} h left) -- not submitted"
            break
        fi
        minutes="$(awk -v l="${left}" 'BEGIN { printf "%d", int(l * 60) }')"
        hours="${left}"
        expected="$(awk -v x="${minutes}" -v s="${SETUP_MIN}" -v r="${rate}" -v m="${MARGIN}" \
            'BEGIN { printf "%d", ((x - s - 8) * 60) / (r * m) }')"
        echo -n "CAPPED to ${minutes} min (~${expected} of ${todo} pairs) "
        last_chunk=1
    fi
    used_h="$(awk -v u="${used_h}" -v h="${hours}" 'BEGIN { printf "%.2f", u + h }')"

    time_arg="$(awk -v x="${minutes}" 'BEGIN { printf "%02d:%02d:00", int(x/60), x%60 }')"
    args=(--time="${time_arg}"
          --job-name="ig_bulk_${slug:0:16}"
          --export="ALL,CODE_ROOT=${CODE_ROOT},DATA_ROOT=${DATA_ROOT},BULK_MODEL=${model},BUDGET_FLOOR=${FLOOR}")
    [[ -n "${DEP}" ]] && args+=(--dependency="afterany:${DEP}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "DRY RUN"
        echo "    sbatch ${args[*]} slurm/ig/bulk_attribution_chunk.sbatch"
        submitted=$((submitted + 1))
        DEP="<job${submitted}>"
        [[ "${last_chunk}" -eq 1 ]] && break
        continue
    fi
    jobid="$(sbatch --parsable "${args[@]}" \
        "${CODE_ROOT}/slurm/ig/bulk_attribution_chunk.sbatch")"
    echo "submitted ${jobid}${DEP:+ (after ${DEP})}"
    DEP="${jobid}"
    submitted=$((submitted + 1))
    [[ "${last_chunk}" -eq 1 ]] && break
done <<< "${PLAN}"

echo
echo "chunks: ${submitted}; reserved ${used_h} h of ${available} h available"
