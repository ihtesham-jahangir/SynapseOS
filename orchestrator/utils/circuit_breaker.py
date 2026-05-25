"""
Reusable three-state circuit breaker.

States:
  CLOSED    — normal operation; all calls go through
  OPEN      — fail fast; no calls sent to backend (recovery_timeout countdown)
  HALF_OPEN — one trial call allowed; success → CLOSED, failure → OPEN

Usage:
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0)
    if cb.is_open():
        raise SomeError("circuit open")
    try:
        result = do_thing()
        cb.record_success()
    except Exception:
        cb.record_failure()
        raise
"""
from __future__ import annotations

import time
from typing import Optional

from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class CircuitBreaker:
    """
    Three-state circuit breaker: CLOSED → OPEN (after N failures) → HALF_OPEN.

    Thread-safe via monotonic clock comparisons; no locking needed for reads.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        name: str = "unnamed",
    ) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_timeout_s
        self._name = name
        self._failures = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self._recovery:
            return "half_open"
        return "open"

    def is_open(self) -> bool:
        return self.state == "open"

    def record_success(self) -> None:
        if self._opened_at is not None:
            log.info("Circuit breaker CLOSED — backend recovered", circuit=self._name)
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning(
                "Circuit breaker OPEN — failing fast",
                circuit=self._name,
                failures=self._failures,
                recovery_s=self._recovery,
            )

    def status(self) -> dict:
        return {
            "state": self.state,
            "failures": self._failures,
            "threshold": self._threshold,
            "recovery_timeout_s": self._recovery,
        }
