# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Error event structures for evaluation error reporting."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """Structured error event for evaluation failures.

    Provides comprehensive context for debugging and operational monitoring.
    Supports both sequential and concurrent evaluation modes.
    """

    timestamp: float
    error_type: str
    severity: str  # "fatal", "error", "warning", "info"
    component: str  # e.g., "manager", "registry", "worker", "evaluator"
    message: str
    evaluator_name: str | None = None
    metric_name: str | None = None
    invocation_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    exception: str | None = None
    recovery_action: str | None = None
    operational_impact: str | None = None
    # Concurrent mode support
    worker_name: str | None = None  # Thread name for multi-worker tracking
    async_context: bool = False  # Whether error occurred in async context

    def to_log_extra(self) -> dict[str, Any]:
        """Convert to dict for structured logging extra parameter."""
        return {
            "error_type": self.error_type,
            "severity": self.severity,
            "component": self.component,
            "evaluator_name": self.evaluator_name,
            "metric_name": self.metric_name,
            "invocation_id": self.invocation_id,
            "recovery_action": self.recovery_action,
            "operational_impact": self.operational_impact,
            "worker_name": self.worker_name,
            "async_context": self.async_context,
            **self.details,
        }


class ErrorTracker:
    """Tracks and reports evaluator errors with rate limiting.

    Thread-safe implementation supporting concurrent evaluation mode
    with multiple worker threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.error_counts: dict[str, int] = {}
        self.last_errors: dict[str, ErrorEvent] = {}
        self.total_errors = 0
        # Per-worker error tracking for concurrent mode
        self.worker_error_counts: dict[str, int] = {}
        self._async_error_count = 0
        self._sync_error_count = 0

    def record_error(
        self,
        error_type: str,
        component: str,
        message: str,
        *,
        evaluator_name: str | None = None,
        metric_name: str | None = None,
        invocation_id: str | None = None,
        exception: Exception | None = None,
        recovery_action: str | None = None,
        operational_impact: str | None = None,
        severity: str = "error",
        details: dict[str, Any] | None = None,
        worker_name: str | None = None,
        async_context: bool = False,
    ) -> ErrorEvent:
        """Record an error event with context.

        Thread-safe method for recording errors from multiple worker threads.

        Args:
            error_type: Type of error (e.g., "evaluator_error", "handler_error")
            component: Component where error occurred
            message: Human-readable error description
            evaluator_name: Name of evaluator if applicable
            metric_name: Metric being evaluated if applicable
            invocation_id: Trace/span ID if available
            exception: Exception object if available
            recovery_action: What the system did in response
            operational_impact: Impact on evaluation results
            severity: Error severity level
            details: Additional structured context
            worker_name: Name of the worker thread (for concurrent mode)
            async_context: Whether error occurred in async context

        Returns:
            ErrorEvent instance
        """
        # Auto-detect worker name if not provided
        if worker_name is None:
            worker_name = threading.current_thread().name

        key = f"{error_type}:{component}:{evaluator_name or 'none'}"

        exception_str = None
        if exception is not None:
            exception_str = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )

        event = ErrorEvent(
            timestamp=time.time(),
            error_type=error_type,
            severity=severity,
            component=component,
            evaluator_name=evaluator_name,
            metric_name=metric_name,
            invocation_id=invocation_id,
            message=message,
            details=details or {},
            exception=exception_str,
            recovery_action=recovery_action,
            operational_impact=operational_impact,
            worker_name=worker_name,
            async_context=async_context,
        )

        # Thread-safe updates
        with self._lock:
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            self.total_errors += 1
            self.last_errors[key] = event

            # Track per-worker errors
            if worker_name:
                self.worker_error_counts[worker_name] = (
                    self.worker_error_counts.get(worker_name, 0) + 1
                )

            # Track async vs sync errors
            if async_context:
                self._async_error_count += 1
            else:
                self._sync_error_count += 1

        return event

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of tracked errors for diagnostics.

        This method provides operational visibility into evaluation system health.
        Use cases include:
        - Monitoring dashboards to track error rates and patterns
        - Debugging evaluation failures by examining error types and counts
        - Identifying problematic evaluators or workers in concurrent mode
        - Post-mortem analysis of evaluation system issues

        Returns:
            Dictionary containing:
            - total_errors: Total number of errors recorded
            - error_counts: Breakdown by error type/component/evaluator
            - unique_error_types: Number of distinct error categories
            - errors_by_worker: Per-worker error counts (concurrent mode)
            - async_errors: Errors from async evaluation context
            - sync_errors: Errors from sync evaluation context
        """
        with self._lock:
            return {
                "total_errors": self.total_errors,
                "error_counts": dict(self.error_counts),
                "unique_error_types": len(self.error_counts),
                "errors_by_worker": dict(self.worker_error_counts),
                "async_errors": self._async_error_count,
                "sync_errors": self._sync_error_count,
            }

    def clear(self) -> None:
        """Clear tracked errors."""
        with self._lock:
            self.error_counts.clear()
            self.last_errors.clear()
            self.worker_error_counts.clear()
            self.total_errors = 0
            self._async_error_count = 0
            self._sync_error_count = 0


__all__ = [
    "ErrorEvent",
    "ErrorTracker",
]
