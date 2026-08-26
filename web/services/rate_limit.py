"""Tiny in-process sliding-window rate limiter.

Deliberately not backed by Redis or the database: the reviewer pilot runs a
single uvicorn worker, and the only endpoints that need throttling are the
password ones, where a modest per-actor cap is enough to blunt brute-force
guessing without adding infrastructure. State lives on ``app.state`` so each
app instance (and therefore each test) starts clean.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """Counts hits per key inside a rolling time window."""

    def __init__(self, monotonic=time.monotonic) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._monotonic = monotonic

    def check(self, key: str, max_attempts: int, window_seconds: float) -> bool:
        """Record a hit for ``key``; return False once the window is full.

        The hit is recorded only when it is allowed, so a caller that is being
        refused does not push its own window forward forever.
        """
        if max_attempts <= 0:
            return True
        now = self._monotonic()
        hits = self._hits[key]
        cutoff = now - window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= max_attempts:
            return False
        hits.append(now)
        return True

    def reset(self, key: str) -> None:
        self._hits.pop(key, None)
