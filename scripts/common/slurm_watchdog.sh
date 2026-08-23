#!/bin/bash
# slurm_watchdog.sh -- single-pass health check for this user's SLURM jobs.
#
# WHAT IT DOES (one pass per invocation; no daemon loop -- cron or a session
# Monitor provides the cadence):
#   1. squeue: flags jobs PENDING longer than WATCHDOG_PENDING_MAX minutes,
#      RUNNING jobs whose StdOut log has not been written to for more than
#      WATCHDOG_STALL_MAX minutes, and RUNNING jobs close to their wallclock
#      limit (see NEAR_WALLCLOCK below).
#   2. sacct: flags jobs that finished in the last WATCHDOG_LOOKBACK_HOURS
#      hours with state FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, BOOT_FAIL
#      or DEADLINE.
#
# OUTPUT:
#   - human-readable "ALERT ..." lines on stdout
#   - structured lines appended to the status log:
#       timestamp|jobid|state|reason
#     default: <repo_root>/runs/active/slurm_watchdog_status.log
#   - when nothing is flagged: exactly one no-op line and no status-log write.
#
# EXIT CODES:
#   0  nothing to flag
#   1  job alerts raised (so callers can react)
#   2  usage / configuration error (no squeue on PATH, bad env override)
#   3  DEGRADED -- the watchdog could not see part of the picture (squeue or
#      sacct failed or timed out). This code takes precedence over 1: an
#      incomplete pass must never be reported as a clean one. The script
#      FAILS CLOSED -- it never prints the "OK" line or exits 0 when blind.
#
# NEAR_WALLCLOCK: fires only when a RUNNING job has used > 0 seconds, has
# consumed more than half its limit, and has <= WATCHDOG_WALL_WARN minutes
# left. The half-of-limit guard stops jobs whose *total* limit is shorter than
# the warn window (common here, since time limits are kept realistic) from
# alerting the instant they start.
#
# DEDUP: an alert is appended to the status log only if that exact
# jobid|state pair is not already present, so the log stays idempotent under a
# repeating cron. Repeats are still printed on stdout, marked
# "(previously reported)", and still produce exit 1 -- the condition is
# unresolved until the operator acts on it. Watchdog-health records use a
# source-qualified pseudo job id ("WATCHDOG:squeue", "WATCHDOG:sacct",
# "WATCHDOG:config") so distinct blind spots dedup independently. The log is
# rolled over to <log>.1 once it exceeds WATCHDOG_LOG_MAX_BYTES.
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
#   Monitor cadence. Branch on squeue's OWN exit status, never on whether its
#   output is empty: a failed or timed-out squeue also produces no rows, and a
#   loop that cannot tell those apart disarms itself during an outage.
#
#       while :; do
#         if ! rows="$(timeout 30 squeue -u irowerojas -h -o '%i')"; then
#           echo "squeue unavailable -- staying armed"   # do NOT tear down
#           sleep 60; continue
#         fi
#         [ -n "$rows" ] || break                        # queue genuinely empty
#         bash scripts/common/slurm_watchdog.sh          # exit 1 = job alert,
#         sleep 600                                      # exit 3 = watchdog blind
#       done
#
#   Tear the loop down only on that verified-empty branch -- the script itself
#   is stateless, so there is nothing else to clean up.
#
# ENV OVERRIDES (all the numeric ones are validated; a bad value is a hard
# error, never a silently skipped check):
#   WATCHDOG_USER            SLURM user to watch      (default: $USER, else irowerojas)
#   WATCHDOG_PENDING_MAX     minutes                  (default: 30)
#   WATCHDOG_STALL_MAX       minutes                  (default: 20)
#   WATCHDOG_WALL_WARN       minutes                  (default: 15)
#   WATCHDOG_LOOKBACK_HOURS  hours of sacct history   (default: 2)
#   WATCHDOG_STATUS_LOG      status-log path          (default: see above)
#   WATCHDOG_CMD_TIMEOUT     seconds per slurm call   (default: 30)
#   WATCHDOG_LOG_MAX_BYTES   rotate log above this    (default: 5242880)

set -uo pipefail

WATCHDOG_USER="${WATCHDOG_USER:-${USER:-irowerojas}}"
WATCHDOG_PENDING_MAX="${WATCHDOG_PENDING_MAX:-30}"
WATCHDOG_STALL_MAX="${WATCHDOG_STALL_MAX:-20}"
WATCHDOG_WALL_WARN="${WATCHDOG_WALL_WARN:-15}"
WATCHDOG_LOOKBACK_HOURS="${WATCHDOG_LOOKBACK_HOURS:-2}"
WATCHDOG_CMD_TIMEOUT="${WATCHDOG_CMD_TIMEOUT:-30}"
WATCHDOG_LOG_MAX_BYTES="${WATCHDOG_LOG_MAX_BYTES:-5242880}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATUS_LOG="${WATCHDOG_STATUS_LOG:-${REPO_ROOT}/runs/active/slurm_watchdog_status.log}"

ALERT_COUNT=0
DEGRADED=0
NOW_EPOCH="$(date +%s)"

# Roll the status log over once it gets large; keep a single previous copy.
rotate_status_log() {
    [ -f "$STATUS_LOG" ] || return 0
    # May run before validation (config errors log an alert too), so re-check.
    [[ "$WATCHDOG_LOG_MAX_BYTES" =~ ^[0-9]+$ ]] || return 0
    local size
    size="$(stat -c %s "$STATUS_LOG" 2>/dev/null || echo 0)"
    if [ "$size" -gt "$WATCHDOG_LOG_MAX_BYTES" ]; then
        mv -f "$STATUS_LOG" "${STATUS_LOG}.1" 2>/dev/null || true
    fi
}

# Append one structured record unless this jobid|state pair is already logged.
# Always prints a human-readable line and counts towards ALERT_COUNT.
log_alert() {  # jobid, state, reason
    local jobid="$1" state="$2" reason="$3" repeat=""
    mkdir -p "$(dirname "$STATUS_LOG")" 2>/dev/null || true
    rotate_status_log
    if [ -f "$STATUS_LOG" ] && grep -qF "|${jobid}|${state}|" "$STATUS_LOG" 2>/dev/null; then
        repeat=" (previously reported)"
    else
        printf '%s|%s|%s|%s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$jobid" "$state" "$reason" >> "$STATUS_LOG" 2>/dev/null || \
            echo "slurm_watchdog: WARNING could not write $STATUS_LOG" >&2
    fi
    echo "ALERT [$state] job $jobid: ${reason}${repeat}"
    ALERT_COUNT=$((ALERT_COUNT + 1))
}

# The watchdog itself is impaired: say so loudly and remember it. The pseudo
# job id carries the source ("WATCHDOG:squeue", "WATCHDOG:sacct", ...) so that
# two different blind spots in one pass are not deduped into one record.
log_blind() {  # source, reason
    DEGRADED=1
    echo "slurm_watchdog: WARNING $2" >&2
    log_alert "WATCHDOG:$1" "WATCHDOG_BLIND" "$2"
}

die_config() {  # message
    echo "slurm_watchdog: ERROR $1" >&2
    log_alert "WATCHDOG:config" "WATCHDOG_BLIND" "configuration error: $1"
    exit 2
}

require_uint() {  # var name, value
    case "$2" in
        ''|*[!0-9]*) die_config "$1 must be a non-negative integer, got '$2'" ;;
    esac
}

require_uint WATCHDOG_PENDING_MAX "$WATCHDOG_PENDING_MAX"
require_uint WATCHDOG_STALL_MAX "$WATCHDOG_STALL_MAX"
require_uint WATCHDOG_WALL_WARN "$WATCHDOG_WALL_WARN"
require_uint WATCHDOG_LOOKBACK_HOURS "$WATCHDOG_LOOKBACK_HOURS"
require_uint WATCHDOG_CMD_TIMEOUT "$WATCHDOG_CMD_TIMEOUT"
require_uint WATCHDOG_LOG_MAX_BYTES "$WATCHDOG_LOG_MAX_BYTES"

command -v squeue >/dev/null 2>&1 || die_config "squeue not found (not on a SLURM host?)"

# Convert a SLURM duration ([DD-]HH:MM:SS | MM:SS | UNLIMITED) to seconds.
# Prints -1 for anything that carries no finite duration, without leaking
# arithmetic errors on unexpected input.
slurm_dur_to_sec() {
    local raw="${1:-}" days=0 rest secs=0
    local parts IFS
    case "$raw" in
        ""|UNLIMITED|INVALID|NOT_SET|N/A|Unknown) echo -1; return 0 ;;
    esac
    if [[ "$raw" == *-* ]]; then
        days="${raw%%-*}"
        rest="${raw#*-}"
    else
        rest="$raw"
    fi
    # Reject anything that is not purely digits and colons before doing math.
    if [[ ! "$days" =~ ^[0-9]+$ ]] || [[ ! "$rest" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
        echo -1; return 0
    fi
    IFS=':'
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
QUEUE_OK=1
if ! QUEUE_ROWS="$(timeout "$WATCHDOG_CMD_TIMEOUT" squeue -u "$WATCHDOG_USER" -h -o '%i|%T|%j|%M|%l|%V|%r' 2>/dev/null)"; then
    QUEUE_OK=0
    QUEUE_ROWS=""
    log_blind "squeue" "squeue failed or timed out after ${WATCHDOG_CMD_TIMEOUT}s -- queue state unknown"
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

        # Near wallclock limit? Require real elapsed time and more than half
        # the limit consumed, so short-limit jobs do not alert at t=0.
        if [ "$used_s" -gt 0 ] && [ "$limit_s" -gt 0 ] && [ $((used_s * 2)) -ge "$limit_s" ]; then
            remaining_s=$((limit_s - used_s))
            if [ "$remaining_s" -le $((WATCHDOG_WALL_WARN * 60)) ]; then
                rem_min=$((remaining_s / 60))
                [ "$rem_min" -lt 0 ] && rem_min=0
                log_alert "$jobid" "NEAR_WALLCLOCK" \
                    "$name has ${rem_min}m left of ${limit} wallclock (used ${used}, warn <= ${WATCHDOG_WALL_WARN}m)"
            fi
        fi

        # Stalled log? Parse StdOut line-anchored so paths with spaces survive.
        stdout_path=""
        if job_detail="$(timeout "$WATCHDOG_CMD_TIMEOUT" scontrol show job "$jobid" 2>/dev/null)"; then
            stdout_path="$(printf '%s\n' "$job_detail" | sed -n 's/^[[:space:]]*StdOut=//p' | head -n1)"
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
if ! command -v sacct >/dev/null 2>&1; then
    log_blind "sacct" "sacct not found -- recently finished jobs not checked"
else
    SINCE="$(date -d "${WATCHDOG_LOOKBACK_HOURS} hours ago" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || echo "")"
    if [ -z "$SINCE" ]; then
        log_blind "sacct" "could not compute sacct start time from WATCHDOG_LOOKBACK_HOURS='${WATCHDOG_LOOKBACK_HOURS}'"
    else
        SACCT_ROWS=""
        if ! SACCT_ROWS="$(timeout "$WATCHDOG_CMD_TIMEOUT" sacct -u "$WATCHDOG_USER" -S "$SINCE" -X -n -P \
                -o JobID,JobName,State,ExitCode,End 2>/dev/null)"; then
            log_blind "sacct" "sacct failed or timed out after ${WATCHDOG_CMD_TIMEOUT}s -- recent failures unknown"
            SACCT_ROWS=""
        fi
        while IFS='|' read -r jobid name state exitcode endtime; do
            [ -n "${jobid:-}" ] || continue
            # sacct states can carry a suffix, e.g. "CANCELLED by 12345".
            base_state="${state%% *}"
            case "$base_state" in
                FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE)
                    log_alert "$jobid" "$base_state" \
                        "$name finished ${endtime:-?} with exit ${exitcode:-?} (last ${WATCHDOG_LOOKBACK_HOURS}h)"
                    ;;
            esac
        done <<< "$SACCT_ROWS"
    fi
fi

# ---------------------------------------------------------------------- result
if [ "$DEGRADED" -ne 0 ]; then
    echo "DEGRADED: watchdog could not see the full picture -- treat this pass as INCONCLUSIVE, not healthy"
    echo "slurm_watchdog: ${ALERT_COUNT} alert(s) recorded in $STATUS_LOG"
    exit 3
fi

if [ "$ALERT_COUNT" -eq 0 ]; then
    if [ "$QUEUE_JOBS" -eq 0 ] && [ "$QUEUE_OK" -eq 1 ]; then
        echo "OK: queue empty, no recent failures (user=$WATCHDOG_USER, last ${WATCHDOG_LOOKBACK_HOURS}h)"
    else
        echo "OK: ${QUEUE_JOBS} job(s) in queue healthy, no recent failures (user=$WATCHDOG_USER)"
    fi
    exit 0
fi

echo "slurm_watchdog: ${ALERT_COUNT} alert(s) recorded in $STATUS_LOG"
exit 1
