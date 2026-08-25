#!/bin/bash
# gpu_budget_guard.sh -- read an allocation balance and decide whether to run.
#
# WHY: issue #84 chains six GPU chunks with --dependency=afterany, so every
# chunk starts regardless of what happened to the one before it. The budget, not
# the dependency graph, is what stops the chain: each chunk asks `accounts` for
# the live balance and exits 0 without running once it is below the floor. Exit 0
# matters -- a nonzero exit would still be "afterany", but it would light up the
# watchdog and sacct with FAILED states for jobs that did exactly the right
# thing.
#
# USAGE
#   source scripts/common/gpu_budget_guard.sh
#   budget_balance beto-delta-gpu          # echoes hours, e.g. 39
#   budget_floor_reached beto-delta-gpu 5  # exit status 0 = at/below floor
#
#   # the whole guard, as a chunk preamble:
#   if budget_floor_reached "$ACCOUNT" "$FLOOR"; then
#       echo "BUDGET FLOOR REACHED"
#       exit 0
#   fi
#
# PARSING: `accounts` prints a table whose rows are
#   <account>  <balance>  <deposited>  <project...>
# so the balance is field 2 of the row whose field 1 is the account. Anything
# else -- no such row, a non-numeric field, `accounts` missing or failing -- is
# reported as a hard error by budget_balance (empty output, nonzero status).
#
# FAIL-CLOSED: budget_floor_reached treats an unreadable balance as "floor
# reached". A chunk that cannot see the budget must not spend it.

budget_balance() {
    local account="${1:?usage: budget_balance ACCOUNT}"
    local out
    if ! out="$(timeout 60 accounts 2>/dev/null)"; then
        echo "budget_balance: 'accounts' failed or timed out" >&2
        return 1
    fi
    local value
    value="$(awk -v acct="${account}" '$1 == acct { print $2; found=1 } END { exit !found }' \
        <<< "${out}")" || {
        echo "budget_balance: no row for account '${account}' in accounts output" >&2
        return 1
    }
    if ! [[ "${value}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        echo "budget_balance: non-numeric balance '${value}' for '${account}'" >&2
        return 1
    fi
    echo "${value}"
}

# 0 = at or below the floor (do not run), 1 = above it (safe to run).
budget_floor_reached() {
    local account="${1:?usage: budget_floor_reached ACCOUNT FLOOR}"
    local floor="${2:?usage: budget_floor_reached ACCOUNT FLOOR}"
    local balance
    if ! balance="$(budget_balance "${account}")"; then
        echo "budget_floor_reached: balance unreadable -- failing closed" >&2
        return 0
    fi
    awk -v b="${balance}" -v f="${floor}" 'BEGIN { exit !(b < f) }'
}

budget_log() {
    local account="${1:?usage: budget_log ACCOUNT LABEL}"
    local label="${2:-balance}"
    local balance
    balance="$(budget_balance "${account}")" || balance="UNREADABLE"
    printf '[budget] %s %s: %s h  (%s)\n' "${account}" "${label}" "${balance}" "$(date -Is)"
}
