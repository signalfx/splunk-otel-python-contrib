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
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """Structured error event for evaluation failures.

    Provides comprehensive context for debugging and operational monitoring.
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
            **self.details,
        }


class ErrorTracker:
    """Tracks and reports evaluator errors with rate limiting."""

    def __init__(self) -> None:
        self.error_counts: dict[str, int] = {}
        self.last_errors: dict[str, ErrorEvent] = {}
        self.total_errors = 0

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
    ) -> ErrorEvent:
        """Record an error event with context.

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

        Returns:
            ErrorEvent instance
        """
        key = f"{error_type}:{component}:{evaluator_name or 'none'}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.total_errors += 1

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
        )

        self.last_errors[key] = event
        return event

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of tracked errors."""
        return {
            "total_errors": self.total_errors,
            "error_counts": dict(self.error_counts),
            "unique_error_types": len(self.error_counts),
        }

    def clear(self) -> None:
        """Clear tracked errors."""
        self.error_counts.clear()
        self.last_errors.clear()
        self.total_errors = 0


__all__ = [
    "ErrorEvent",
    "ErrorTracker",
]
