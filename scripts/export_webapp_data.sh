#!/bin/bash
# Verify and report data files required by the localLatin webapp.
# Usage: bash scripts/export_webapp_data.sh [repo_root]
set -euo pipefail

ROOT="${1:-.}"
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
    fi
}

check "canon_unlabelled/" \
    "$ROOT/canon_unlabelled" "" \
    "find '$ROOT/canon_unlabelled' -name '*.txt' | wc -l | tr -d ' '"

check "canon_labelled/" \
    "$ROOT/canon_labelled" "" \
    "find '$ROOT/canon_labelled' -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' '"

check "predictions CSV" \
    "$ROOT/runs/active/resubmit/unlabelled/unlabelled_predictions.csv" "" \
    "tail -n +2 '$ROOT/runs/active/resubmit/unlabelled/unlabelled_predictions.csv' | wc -l | tr -d ' '"

check "IG examples CSV" \
    "$ROOT/runs/active/ig_examples/phase12f_examples.csv" "" \
    "tail -n +2 '$ROOT/runs/active/ig_examples/phase12f_examples.csv' | wc -l | tr -d ' '"

check "IG artifacts" \
    "$ROOT/runs/active/ig_examples/artifacts" "" \
    "find '$ROOT/runs/active/ig_examples/artifacts' -name '*.npz' | wc -l | tr -d ' '"

echo ""
echo "Set data_root in web/config.yaml to: $(cd "$ROOT" && pwd)"
