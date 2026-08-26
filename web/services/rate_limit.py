"""Tiny in-process sliding-window rate limiter.

Deliberately not backed by Redis or the database: the reviewer pilot runs a
single uvicorn worker (``--workers 1`` in ``deploy/locallatin.service``), which
is what makes per-process counters correct. A service restart wipes every open
window, so the limiter blunts sustained guessing but is not an audit trail.

Callers drive it in three steps rather than one, because for password endpoints
only a *failed* attempt may count against the window: ``allows()`` to consult
the window, ``record()`` after a verification failure, and ``reset()`` after a
success. Recording on success would let a user lock themselves out of the one
route that can clear a forced-change gate.
"""

from __future__ import annotations

import math
import time
from collections import deque


class RateLimiter:
    """Counts recorded failures per key inside a rolling time window."""

    def __init__(self, monotonic=time.monotonic) -> None:
        # A plain dict, not a defaultdict: reading an unknown key must not create
        # an entry, or every probe would grow the key space for good.
        self._hits: dict[str, deque[float]] = {}
        self._monotonic = monotonic

    def allows(self, key: str, max_attempts: int, window_seconds: float) -> bool:
        """True when ``key`` still has room in its window. Records nothing."""
        if max_attempts <= 0:
            return True
        return len(self._live_hits(key, window_seconds)) < max_attempts

    def record(self, key: str) -> None:
        """Count one failed attempt against ``key``."""
        self._hits.setdefault(key, deque()).append(self._monotonic())

    def reset(self, key: str) -> None:
        """Forget every recorded failure for ``key`` (call after a success)."""
        self._hits.pop(key, None)

    def retry_after(self, key: str, window_seconds: float) -> int:
        """Whole seconds until the oldest recorded failure leaves the window."""
        hits = self._live_hits(key, window_seconds)
        if not hits:
            return 0
        remaining = window_seconds - (self._monotonic() - hits[0])
        return max(1, int(math.ceil(remaining)))

    def tracked_keys(self) -> int:
        """How many keys currently hold at least one hit (for tests/diagnostics)."""
        return len(self._hits)

    def _live_hits(self, key: str, window_seconds: float) -> deque[float]:
        hits = self._hits.get(key)
        if hits is None:
            return deque()
        cutoff = self._monotonic() - window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            # Evict rather than keep an empty deque around: without this the key
            # space would only ever grow, one entry per address ever seen.
            del self._hits[key]
            return deque()
        return hits
