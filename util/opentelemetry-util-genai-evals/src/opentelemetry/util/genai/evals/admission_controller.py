# admission_controller.py

import asyncio
import threading
import time
from typing import Optional, Tuple

from .env import (
    read_evaluation_rate_limit_burst,
    read_evaluation_rate_limit_rps,
)


class _TokenBucketLimiter:
    """
    Thread-safe token bucket limiter.
    Pure rate limiting primitive, no business semantics.
    """

    def __init__(self, rate_per_sec: float, burst: int):
        self._rate = float(rate_per_sec)
        self._capacity = float(max(1, burst))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        if self._rate <= 0:
            # Disabled
            return True

        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last
            self._last = now
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate,
            )
            if self._tokens >= cost:
                self._tokens -= cost
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

    def __init__(self):
        rps = read_evaluation_rate_limit_rps()
        burst = read_evaluation_rate_limit_burst()

        self._limiter: Optional[_TokenBucketLimiter] = None
        if rps > 0:
            self._limiter = _TokenBucketLimiter(
                rate_per_sec=rps,
                burst=burst,
            )

    def allow(self, invocation) -> Tuple[bool, Optional[str]]:
        """
        Returns:
          (True, None) if allowed
          (False, error_code) if dropped
        """
        _ = invocation

        if self._limiter is None:
            return True, None

        if self._limiter.allow():
            return True, None

        return False, "client_evaluation_rate_limited"

    async def allow_async(self, invocation) -> Tuple[bool, Optional[str]]:
        """Check if invocation should be allowed (async version).

        Returns:
          (True, None) if allowed
          (False, error_code) if dropped
        """
        _ = invocation
        if self._limiter is None:
            return True, None
        allowed = await asyncio.to_thread(self._limiter.allow)
        if allowed:
            return True, None
        return False, "client_evaluation_rate_limited"
