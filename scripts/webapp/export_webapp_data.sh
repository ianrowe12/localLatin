#!/bin/bash
# Verify and report data files required by the localLatin webapp.
# Usage: bash scripts/webapp/export_webapp_data.sh [repo_root] [--strict]
#
# --strict exits 1 if anything is MISSING. deploy/deploy.sh runs it that way so
# a host with no data payload fails with a readable message instead of an
# opaque uvicorn FileNotFoundError behind the health check.
set -euo pipefail

ROOT="."
STRICT=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        *) ROOT="$arg" ;;
    esac
done

MISSING=0

echo "=== localLatin Webapp Data Contract ==="
echo ""

# check LABEL PATH KIND [COUNT_CMD]
#
# KIND is "file" or "dir". Existence alone is not enough: a truncated download
# or an interrupted extract leaves a zero-byte CSV or an empty artifacts/
# directory, which passes `[ -e ]` and then fails at uvicorn startup with an
# opaque parser error. A file must be non-empty, and a COUNT_CMD (rows, files,
# subdirectories) must come back greater than zero.
check() {
    local label="$1" path="$2" kind="${3:-any}" count_cmd="${4:-}"

    if [ ! -e "$path" ]; then
        echo "  MISSING  $label: $path"
        MISSING=$((MISSING + 1))
        return
    fi

    if [ "$kind" = "file" ] && [ ! -s "$path" ]; then
        echo "  EMPTY    $label: $path (zero bytes)"
        MISSING=$((MISSING + 1))
        return
    fi

    if [ "$kind" = "dir" ] && [ ! -d "$path" ]; then
        echo "  MISSING  $label: $path (not a directory)"
        MISSING=$((MISSING + 1))
        return
    fi

    local count=""
    if [ -n "$count_cmd" ]; then
        count=$(eval "$count_cmd" 2>/dev/null || echo 0)
        case "$count" in
            ''|*[!0-9]*) count=0 ;;
        esac
        if [ "$count" -le 0 ]; then
            echo "  EMPTY    $label: $path (no usable content)"
            MISSING=$((MISSING + 1))
            return
        fi
        echo "  OK  $label ($count)"
        return
    fi

    echo "  OK  $label"
}

check "data/canon_unlabelled/" \
    "$ROOT/data/canon_unlabelled" dir \
    "find '$ROOT/data/canon_unlabelled' -name '*.txt' | wc -l | tr -d ' '"

check "data/canon_labelled/" \
    "$ROOT/data/canon_labelled" dir \
    "find '$ROOT/data/canon_labelled' -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' '"

# One predictions CSV per post-processing variant. The pre-variant frozen file
# (unlabelled_predictions.csv) is stale and is no longer read by the webapp.
# Each must be non-empty AND carry at least one data row: a zero-byte or
# header-only CSV is what a truncated sync leaves behind.
for variant in raw abtt sif sif_abtt; do
    check "predictions CSV ($variant)" \
        "$ROOT/runs/active/resubmit/unlabelled/unlabelled_predictions_${variant}.csv" file \
        "tail -n +2 '$ROOT/runs/active/resubmit/unlabelled/unlabelled_predictions_${variant}.csv' | wc -l | tr -d ' '"
done

# Query-query cosine matrices, one per model. Reviewer-created directories are
# scored from these (issue #95); without them the webapp still runs but serves
# no reviewer-directory candidates, so a missing matrix is worth reporting.
# Checked as a group: the count is the number of models that can score them.
check "q-q matrices" \
    "$ROOT/runs/active/resubmit/unlabelled" dir \
    "find '$ROOT/runs/active/resubmit/unlabelled' -maxdepth 1 -name 'qq_sim_*.npz' -size +0 | wc -l | tr -d ' '"

check "IG examples CSV" \
    "$ROOT/runs/active/ig_examples/phase12f_examples.csv" file \
    "tail -n +2 '$ROOT/runs/active/ig_examples/phase12f_examples.csv' | wc -l | tr -d ' '"

# Counts only non-empty .npz files, so a directory of zero-byte artifacts fails.
check "IG artifacts" \
    "$ROOT/runs/active/ig_examples/artifacts" dir \
    "find '$ROOT/runs/active/ig_examples/artifacts' -name '*.npz' -size +0 | wc -l | tr -d ' '"

echo ""
echo "Set data_root in web/config.yaml to: $(cd "$ROOT" && pwd)"

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "$MISSING required data path(s) missing."
    if [ "$STRICT" -eq 1 ]; then
        echo "Publish and install a data release: scripts/webapp/make_data_release.sh," \
             "then set DATA_RELEASE_TAG."
        exit 1
    fi
fi
