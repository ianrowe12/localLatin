#!/usr/bin/env bash
#
# watch_chain.sh -- keep slurm_watchdog.sh armed for the lifetime of a job chain.
#
# The issue #84 chain is five or six dependency-linked GPU jobs running for
# hours, so it needs the watchdog on a cadence rather than a one-shot check.
# This is the loop documented in scripts/common/slurm_watchdog.sh, made a file
# so an orchestration session can background it instead of retyping it.
#
# It branches on squeue's OWN exit status, never on whether its output is empty:
# a failed or timed-out squeue also produces no rows, and a loop that cannot
# tell those apart disarms itself during a scheduler outage. The loop exits only
# on a squeue that succeeded and returned nothing.
#
# USAGE
#   bash scripts/common/watch_chain.sh [INTERVAL_SECONDS]
#
# Alerts go to stdout (and to the watchdog's status log). Exit codes from the
# watchdog -- 1 for a job alert, 3 for a blind pass -- are surfaced as ALERT
# lines here rather than ending the loop: the chain is still running and still
# needs watching.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERVAL="${1:-600}"
USER_NAME="${WATCHDOG_USER:-${USER:-irowerojas}}"

echo "[watch_chain] watching ${USER_NAME}, every ${INTERVAL}s, from ${REPO_ROOT}"
while :; do
    if ! rows="$(timeout 30 squeue -u "${USER_NAME}" -h -o '%i')"; then
        echo "[watch_chain] $(date -Is) squeue unavailable -- staying armed"
        sleep 60
        continue
    fi
    if [[ -z "${rows}" ]]; then
        echo "[watch_chain] $(date -Is) queue empty -- chain finished, standing down"
        break
    fi
    bash "${REPO_ROOT}/scripts/common/slurm_watchdog.sh"
    status=$?
    case "${status}" in
        0) echo "[watch_chain] $(date -Is) ok ($(wc -w <<< "${rows}") job(s) queued)" ;;
        1) echo "[watch_chain] $(date -Is) ALERT: watchdog flagged a job" ;;
        3) echo "[watch_chain] $(date -Is) ALERT: watchdog is blind (squeue/sacct failed)" ;;
        *) echo "[watch_chain] $(date -Is) ALERT: watchdog exited ${status}" ;;
    esac
    sleep "${INTERVAL}"
done
