# admission_controller.py

import asyncio
import threading
import time
from typing import Optional, Tuple

from .env import (
    read_evaluation_rate_limit_burst,
    read_evaluation_rate_limit_enable,
    read_evaluation_rate_limit_rps,
)


class _TokenBucketLimiter:
    """
    Thread-safe token bucket limiter.
    Pure rate limiting primitive, no business semantics.
    """

    def __init__(self, requests_per_sec: int, burst: int):
        self._requests_per_sec = int(requests_per_sec)
        self._capacity = float(max(1, burst))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        if self._requests_per_sec <= 0:
            # Disabled
            return True

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._requests_per_sec,
            )
            if self._tokens >= cost:
                self._tokens -= cost
                self._last = now
                return True
            return False


class EvaluationAdmissionController:
    """
    Evaluation-specific admission control.

    - Currently request-based (1 invocation = 1 cost)
    - Future extensions:
        * token-based cost
        * per-metric cost
        * batch-level admission
    """

    ERROR_CODE_RATE_LIMITED = "client_evaluation_rate_limited"

    def __init__(self):
        enabled = read_evaluation_rate_limit_enable()
        rps = read_evaluation_rate_limit_rps()
        burst = read_evaluation_rate_limit_burst()

        self._limiter: Optional[_TokenBucketLimiter] = None
        if enabled and rps > 0:
            self._limiter = _TokenBucketLimiter(
                requests_per_sec=rps,
                burst=burst,
            )

    def allow(self) -> Tuple[bool, Optional[str]]:
        """Return (True, None) if allowed; (False, error_code) if dropped."""
        if self._limiter is None:
            return True, None

        if self._limiter.allow():
            return True, None

        return False, self.ERROR_CODE_RATE_LIMITED

    async def allow_async(self) -> Tuple[bool, Optional[str]]:
        """Async version of allow()."""
        if self._limiter is None:
            return True, None
        allowed = await asyncio.to_thread(self._limiter.allow)
        if allowed:
            return True, None
        return False, self.ERROR_CODE_RATE_LIMITED
