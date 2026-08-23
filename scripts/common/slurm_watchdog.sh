#!/bin/bash
# slurm_watchdog.sh -- single-pass health check for this user's SLURM jobs.
#
# WHAT IT DOES (one pass per invocation; no daemon loop -- cron or a session
# Monitor provides the cadence):
#   1. squeue: flags jobs PENDING longer than WATCHDOG_PENDING_MAX minutes,
#      RUNNING jobs whose StdOut log has not been written to for more than
#      WATCHDOG_STALL_MAX minutes, and RUNNING jobs within
#      WATCHDOG_WALL_WARN minutes of their wallclock limit.
#   2. sacct: flags jobs that finished in the last WATCHDOG_LOOKBACK_HOURS
#      hours with state FAILED, TIMEOUT, OUT_OF_MEMORY or NODE_FAIL.
#
# OUTPUT:
#   - human-readable "ALERT ..." lines on stdout
#   - structured lines appended to the status log:
#       timestamp|jobid|state|reason
#     default: <repo_root>/runs/active/slurm_watchdog_status.log
#   - when nothing is flagged: exactly one no-op line and no status-log write.
#
# EXIT CODES: 0 = nothing to flag, 1 = at least one alert raised (so callers
# can react), 2 = usage/environment error.
#
# USAGE:
#   bash scripts/common/slurm_watchdog.sh
#   WATCHDOG_PENDING_MAX=10 WATCHDOG_STALL_MAX=5 bash scripts/common/slurm_watchdog.sh
#
# CRON (every 10 minutes, log alerts and keep the last run's stdout):
#   */10 * * * * /bin/bash /u/irowerojas/localLatin/scripts/common/slurm_watchdog.sh \
#       >> /u/irowerojas/localLatin/runs/active/slurm_watchdog_cron.out 2>&1
#
# ARMING IT FROM AN ORCHESTRATION SESSION:
#   While an orchestration session is live it can poll this script directly
#   instead of using cron -- run it after each sbatch submission and on a
#   Monitor cadence, e.g.
#       until ! squeue -u irowerojas -h | grep -q .; do \
#         bash scripts/common/slurm_watchdog.sh; sleep 600; done
#   Exit status 1 means new alerts were printed, so the session should read the
#   stdout lines (or tail runs/active/slurm_watchdog_status.log) and react.
#   Tear the loop down once squeue is empty -- the script itself is stateless.
#
# ENV OVERRIDES:
#   WATCHDOG_USER            SLURM user to watch      (default: $USER, else irowerojas)
#   WATCHDOG_PENDING_MAX     minutes                  (default: 30)
#   WATCHDOG_STALL_MAX       minutes                  (default: 20)
#   WATCHDOG_WALL_WARN       minutes                  (default: 15)
#   WATCHDOG_LOOKBACK_HOURS  hours of sacct history   (default: 2)
#   WATCHDOG_STATUS_LOG      status-log path          (default: see above)
#   WATCHDOG_CMD_TIMEOUT     seconds per slurm call   (default: 30)

set -uo pipefail

WATCHDOG_USER="${WATCHDOG_USER:-${USER:-irowerojas}}"
WATCHDOG_PENDING_MAX="${WATCHDOG_PENDING_MAX:-30}"
WATCHDOG_STALL_MAX="${WATCHDOG_STALL_MAX:-20}"
WATCHDOG_WALL_WARN="${WATCHDOG_WALL_WARN:-15}"
WATCHDOG_LOOKBACK_HOURS="${WATCHDOG_LOOKBACK_HOURS:-2}"
WATCHDOG_CMD_TIMEOUT="${WATCHDOG_CMD_TIMEOUT:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATUS_LOG="${WATCHDOG_STATUS_LOG:-${REPO_ROOT}/runs/active/slurm_watchdog_status.log}"

command -v squeue >/dev/null 2>&1 || { echo "slurm_watchdog: squeue not found (not on a SLURM host?)" >&2; exit 2; }

ALERT_COUNT=0
NOW_EPOCH="$(date +%s)"

# Append one structured record; create the log directory on first use.
log_alert() {  # jobid, state, reason
    local jobid="$1" state="$2" reason="$3"
    mkdir -p "$(dirname "$STATUS_LOG")" 2>/dev/null || true
    printf '%s|%s|%s|%s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$jobid" "$state" "$reason" >> "$STATUS_LOG" 2>/dev/null || \
        echo "slurm_watchdog: WARNING could not write $STATUS_LOG" >&2
    echo "ALERT [$state] job $jobid: $reason"
    ALERT_COUNT=$((ALERT_COUNT + 1))
}

# Convert a SLURM duration ([DD-]HH:MM:SS | MM:SS | UNLIMITED) to seconds.
# Prints -1 for values that carry no finite duration.
slurm_dur_to_sec() {
    local raw="${1:-}" days=0 rest secs=0
    case "$raw" in
        ""|UNLIMITED|INVALID|NOT_SET|N/A|Unknown) echo -1; return 0 ;;
    esac
    if [[ "$raw" == *-* ]]; then
        days="${raw%%-*}"
        rest="${raw#*-}"
    else
        rest="$raw"
    fi
    local IFS=':'
    read -r -a parts <<< "$rest"
    case "${#parts[@]}" in
        3) secs=$((10#${parts[0]} * 3600 + 10#${parts[1]} * 60 + 10#${parts[2]})) ;;
        2) secs=$((10#${parts[0]} * 60 + 10#${parts[1]})) ;;
        1) secs=$((10#${parts[0]})) ;;
        *) echo -1; return 0 ;;
    esac
    echo $((secs + 10#${days} * 86400))
}

# ---------------------------------------------------------------- squeue pass
QUEUE_ROWS=""
if ! QUEUE_ROWS="$(timeout "$WATCHDOG_CMD_TIMEOUT" squeue -u "$WATCHDOG_USER" -h -o '%i|%T|%j|%M|%l|%V|%r' 2>/dev/null)"; then
    echo "slurm_watchdog: WARNING squeue failed or timed out after ${WATCHDOG_CMD_TIMEOUT}s" >&2
    QUEUE_ROWS=""
fi

QUEUE_JOBS=0
while IFS='|' read -r jobid state name used limit submit reason; do
    [ -n "${jobid:-}" ] || continue
    QUEUE_JOBS=$((QUEUE_JOBS + 1))

    case "$state" in
    PENDING)
        submit_epoch="$(date -d "$submit" +%s 2>/dev/null || echo 0)"
        if [ "$submit_epoch" -gt 0 ]; then
            pending_s=$((NOW_EPOCH - submit_epoch))
            [ "$pending_s" -lt 0 ] && pending_s=0
            if [ "$pending_s" -ge $((WATCHDOG_PENDING_MAX * 60)) ]; then
                log_alert "$jobid" "PENDING" \
                    "$name pending $((pending_s / 60))m (threshold ${WATCHDOG_PENDING_MAX}m), reason=${reason:-unknown}"
            fi
        fi
        ;;
    RUNNING)
        used_s="$(slurm_dur_to_sec "$used")"
        limit_s="$(slurm_dur_to_sec "$limit")"

        # Near wallclock limit?
        if [ "$used_s" -ge 0 ] && [ "$limit_s" -gt 0 ]; then
            remaining_s=$((limit_s - used_s))
            if [ "$remaining_s" -le $((WATCHDOG_WALL_WARN * 60)) ]; then
                rem_min=$((remaining_s / 60))
                [ "$rem_min" -lt 0 ] && rem_min=0
                log_alert "$jobid" "NEAR_WALLCLOCK" \
                    "$name has ${rem_min}m left of ${limit} wallclock (warn <= ${WATCHDOG_WALL_WARN}m)"
            fi
        fi

        # Stalled log?
        stdout_path=""
        if job_detail="$(timeout "$WATCHDOG_CMD_TIMEOUT" scontrol show job "$jobid" 2>/dev/null)"; then
            stdout_path="$(printf '%s\n' "$job_detail" | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -n1)"
        fi
        if [ -n "$stdout_path" ] && [ -f "$stdout_path" ]; then
            mtime="$(stat -c %Y "$stdout_path" 2>/dev/null || echo 0)"
            if [ "$mtime" -gt 0 ]; then
                silent_s=$((NOW_EPOCH - mtime))
                [ "$silent_s" -lt 0 ] && silent_s=0
                if [ "$silent_s" -ge $((WATCHDOG_STALL_MAX * 60)) ]; then
                    log_alert "$jobid" "STALLED" \
                        "$name log silent $((silent_s / 60))m (threshold ${WATCHDOG_STALL_MAX}m): $stdout_path"
                fi
            fi
        fi
        ;;
    esac
done <<< "$QUEUE_ROWS"

# ----------------------------------------------------------------- sacct pass
SINCE="$(date -d "${WATCHDOG_LOOKBACK_HOURS} hours ago" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || echo "")"
SACCT_ROWS=""
if [ -n "$SINCE" ] && command -v sacct >/dev/null 2>&1; then
    if ! SACCT_ROWS="$(timeout "$WATCHDOG_CMD_TIMEOUT" sacct -u "$WATCHDOG_USER" -S "$SINCE" -X -n -P \
            -o JobID,JobName,State,ExitCode,End 2>/dev/null)"; then
        echo "slurm_watchdog: WARNING sacct failed or timed out after ${WATCHDOG_CMD_TIMEOUT}s" >&2
        SACCT_ROWS=""
    fi
fi

FAILED_JOBS=0
while IFS='|' read -r jobid name state exitcode endtime; do
    [ -n "${jobid:-}" ] || continue
    # sacct states can carry a suffix, e.g. "CANCELLED by 12345".
    base_state="${state%% *}"
    case "$base_state" in
        FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            FAILED_JOBS=$((FAILED_JOBS + 1))
            log_alert "$jobid" "$base_state" \
                "$name finished ${endtime:-?} with exit ${exitcode:-?} (last ${WATCHDOG_LOOKBACK_HOURS}h)"
            ;;
    esac
done <<< "$SACCT_ROWS"

# ---------------------------------------------------------------------- result
if [ "$ALERT_COUNT" -eq 0 ]; then
    if [ "$QUEUE_JOBS" -eq 0 ]; then
        echo "OK: queue empty, no recent failures (user=$WATCHDOG_USER, last ${WATCHDOG_LOOKBACK_HOURS}h)"
    else
        echo "OK: ${QUEUE_JOBS} job(s) in queue healthy, no recent failures (user=$WATCHDOG_USER)"
    fi
    exit 0
fi

echo "slurm_watchdog: ${ALERT_COUNT} alert(s) written to $STATUS_LOG"
exit 1
