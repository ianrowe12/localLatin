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

check() {
    local label="$1" path="$2" pattern="${3:-}" count_cmd="${4:-}"
    if [ -e "$path" ]; then
        if [ -n "$count_cmd" ]; then
            local count
            count=$(eval "$count_cmd")
            echo "  OK  $label ($count)"
        else
            echo "  OK  $label"
        fi
    else
        echo "  MISSING  $label: $path"
        MISSING=$((MISSING + 1))
    fi
}

check "data/canon_unlabelled/" \
    "$ROOT/data/canon_unlabelled" "" \
    "find '$ROOT/data/canon_unlabelled' -name '*.txt' | wc -l | tr -d ' '"

check "data/canon_labelled/" \
    "$ROOT/data/canon_labelled" "" \
    "find '$ROOT/data/canon_labelled' -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' '"

# One predictions CSV per post-processing variant. The pre-variant frozen file
# (unlabelled_predictions.csv) is stale and is no longer read by the webapp.
for variant in raw abtt sif sif_abtt; do
    check "predictions CSV ($variant)" \
        "$ROOT/runs/active/resubmit/unlabelled/unlabelled_predictions_${variant}.csv" "" \
        "tail -n +2 '$ROOT/runs/active/resubmit/unlabelled/unlabelled_predictions_${variant}.csv' | wc -l | tr -d ' '"
done

check "IG examples CSV" \
    "$ROOT/runs/active/ig_examples/phase12f_examples.csv" "" \
    "tail -n +2 '$ROOT/runs/active/ig_examples/phase12f_examples.csv' | wc -l | tr -d ' '"

check "IG artifacts" \
    "$ROOT/runs/active/ig_examples/artifacts" "" \
    "find '$ROOT/runs/active/ig_examples/artifacts' -name '*.npz' | wc -l | tr -d ' '"

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
