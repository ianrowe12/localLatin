"""The two guards that decide whether a bulk chunk keeps spending GPU time.

Both come from the PR #90 review, and both are about a chunk that fails in a way
SLURM would still record as success:

* the wallclock budget must be measured from *process* start, or a capped chunk
  runs setup + cap + registry past its limit, TIMEOUTs, and loses the registry;
* a systematic post-model-load fault (CUDA OOM, a call-time captum failure)
  fails every pair identically, so without a circuit breaker the chunk burns its
  whole reservation and exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

torch = pytest.importorskip("torch", reason="run_bulk_attribution imports torch")

from run_bulk_attribution import error_abort_reason, parse_args  # noqa: E402

CONSEC = 20
FRAC = 0.10
MIN_ATTEMPTS = 20


def reason(consecutive: int, errors: int, attempted: int) -> str | None:
    return error_abort_reason(consecutive, errors, attempted, CONSEC, FRAC, MIN_ATTEMPTS)


# --- the breaker holds for a healthy chunk ----------------------------------


def test_a_healthy_chunk_never_aborts() -> None:
    # 5,000 pairs, a handful of genuinely bad ones, never in a row.
    assert reason(1, 12, 5000) is None


def test_a_couple_of_early_errors_do_not_abort() -> None:
    """Two failures out of three attempts is 67%, and must still not fire."""
    assert reason(2, 2, 3) is None


def test_the_rate_rule_waits_for_min_attempts() -> None:
    assert reason(1, 19, MIN_ATTEMPTS - 1) is None
    assert reason(1, 19, MIN_ATTEMPTS) is not None


# --- ...and fires on the failures it exists for ------------------------------


def test_a_run_of_failures_aborts() -> None:
    """A late fault: thousands of good pairs, then everything fails.

    Attempts are kept high enough that the rate rule is nowhere near firing
    (20/2000 is 1%), so this isolates the consecutive-run rule -- which is the
    one that has to catch a late fault before it costs an hour.
    """
    assert reason(CONSEC - 1, CONSEC - 1, 2000) is None
    fired = reason(CONSEC, CONSEC, 2000)
    assert fired is not None
    assert "consecutive" in fired


def test_a_high_rate_without_a_run_aborts() -> None:
    """A fault failing most but not all pairs shows no consecutive run."""
    fired = reason(1, 300, 1000)
    assert fired is not None
    assert "--max_error_frac" in fired


def test_total_failure_from_the_first_pair_aborts_quickly() -> None:
    """CUDA OOM on every pair: caught in 20 pairs, not 7,000."""
    assert reason(CONSEC, CONSEC, CONSEC) is not None


def test_either_rule_can_be_disabled() -> None:
    assert error_abort_reason(999, 999, 999, 0, FRAC, MIN_ATTEMPTS) is not None
    assert error_abort_reason(999, 999, 999, 0, 0.0, MIN_ATTEMPTS) is None


# --- the defaults the chunks actually run with -------------------------------


def test_cli_defaults_match_the_reviewed_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_bulk_attribution.py", "--model", "bowphs/LaTa"])
    args = parse_args()

    assert args.max_consecutive_errors == CONSEC
    assert args.max_error_frac == pytest.approx(FRAC)
    assert args.error_frac_min_attempts == MIN_ATTEMPTS
    # 0 means "no wallclock budget"; the sbatch always passes a real one.
    assert args.max_seconds == 0.0


def test_max_seconds_is_measured_from_process_start() -> None:
    """Pin the fix: the budget clock must not restart after setup.

    The sbatch derives --max_seconds as (SLURM limit - 8 min) and those 8
    minutes are the registry's headroom. If the loop compared against its own
    start, a chunk would run setup + the full cap + the registry, which on a
    large vocabulary overruns the reservation.
    """
    source = (REPO_ROOT / "scripts" / "ig" / "run_bulk_attribution.py").read_text()

    assert "if args.max_seconds and (time.time() - started) > args.max_seconds:" in source
    assert "(time.time() - loop_started) > args.max_seconds" not in source
    # loop_started still exists -- it is what the reported s/pair rate uses, and
    # that rate must keep excluding setup or the chunk sizing drifts.
    assert "loop_started = time.time()" in source
    assert "rate = (time.time() - loop_started) / counts[\"built\"]" in source


def test_registry_is_written_before_the_abort_exit() -> None:
    """An aborting chunk must still hand the next run what it did build."""
    source = (REPO_ROOT / "scripts" / "ig" / "run_bulk_attribution.py").read_text()

    registry_write = source.index("tmp.replace(registry_path)")
    abort_exit = source.index('raise SystemExit(f"aborted: {abort_reason}")')
    assert registry_write < abort_exit


def test_stale_tmp_npz_is_swept_before_the_skip_scan() -> None:
    """A leftover .tmp.npz parses to a real example_id and can shadow it."""
    source = (REPO_ROOT / "scripts" / "ig" / "run_bulk_attribution.py").read_text()

    sweep = source.index('artifacts_dir.glob("*.tmp.npz")')
    todo_scan = source.index("todo = [p for p in pairs if not p.artifact_path(")
    assert sweep < todo_scan
