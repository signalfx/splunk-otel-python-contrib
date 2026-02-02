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

"""Test emitter for capturing all GenAI telemetry in memory.

This emitter is designed for testing, validation, and performance benchmarking
of the GenAI evaluation system. It captures all telemetry events and provides
APIs for querying and exporting the captured data.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.util.genai.interfaces import EmitterMeta
from opentelemetry.util.genai.types import (
    Error,
    EvaluationResult,
    GenAI,
)

_LOGGER = logging.getLogger(__name__)

# Singleton instance
_test_emitter_instance: Optional["TestEmitter"] = None
_test_emitter_lock = threading.Lock()


@dataclass
class TelemetryEvent:
    """Represents a single telemetry event captured by the test emitter."""

    timestamp: float
    event_type: str  # "start", "end", "error", "evaluation_results"
    invocation_type: str
    run_id: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    evaluation_results: Optional[List[Dict[str, Any]]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestEmitterStats:
    """Statistics collected by the test emitter."""

    total_starts: int = 0
    total_ends: int = 0
    total_errors: int = 0
    total_evaluation_results: int = 0
    evaluation_results_by_metric: Dict[str, int] = field(default_factory=dict)
    invocations_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    start_time: Optional[float] = None
    last_event_time: Optional[float] = None


class TestEmitter(EmitterMeta):
    """Test emitter that captures all GenAI telemetry for testing and validation.

    Features:
    - Thread-safe capture of all telemetry events
    - Statistics tracking for invocations, evaluations, and errors
    - JSON export capability
    - Correlation between invocations and their evaluation results

    Usage:
        emitter = get_test_emitter()
        # ... run GenAI operations ...
        stats = emitter.get_stats()
        emitter.export_to_json("results.json")
        emitter.reset()
    """

    role = "evaluation"  # Receives evaluation results
    name = "test"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[TelemetryEvent] = []
        self._stats = TestEmitterStats()
        self._invocations: Dict[str, GenAI] = {}  # run_id -> invocation
        self._evaluation_results: Dict[
            str, List[EvaluationResult]
        ] = {}  # run_id -> results
        self._pending_invocations: set[str] = (
            set()
        )  # run_ids awaiting evaluation

    def handles(self, obj: Any) -> bool:
        """Accept all GenAI objects."""
        return isinstance(obj, GenAI)

    def on_start(self, obj: Any) -> None:
        """Capture invocation start events."""
        if not isinstance(obj, GenAI):
            return

        with self._lock:
            now = time.time()
            if self._stats.start_time is None:
                self._stats.start_time = now
            self._stats.last_event_time = now

            run_id = str(obj.run_id)
            invocation_type = type(obj).__name__

            event = TelemetryEvent(
                timestamp=now,
                event_type="start",
                invocation_type=invocation_type,
                run_id=run_id,
                trace_id=hex(obj.trace_id) if obj.trace_id else None,
                span_id=hex(obj.span_id) if obj.span_id else None,
                attributes=dict(obj.attributes) if obj.attributes else {},
            )
            self._events.append(event)
            self._invocations[run_id] = obj
            self._pending_invocations.add(run_id)

            self._stats.total_starts += 1
            self._stats.invocations_by_type[invocation_type] = (
                self._stats.invocations_by_type.get(invocation_type, 0) + 1
            )

    def on_end(self, obj: Any) -> None:
        """Capture invocation end events."""
        if not isinstance(obj, GenAI):
            return

        with self._lock:
            now = time.time()
            self._stats.last_event_time = now

            run_id = str(obj.run_id)
            invocation_type = type(obj).__name__

            # Build attributes including any evaluation info
            attrs = dict(obj.attributes) if obj.attributes else {}

            event = TelemetryEvent(
                timestamp=now,
                event_type="end",
                invocation_type=invocation_type,
                run_id=run_id,
                trace_id=hex(obj.trace_id) if obj.trace_id else None,
                span_id=hex(obj.span_id) if obj.span_id else None,
                attributes=attrs,
            )
            self._events.append(event)
            self._stats.total_ends += 1

    def on_error(self, error: Error, obj: Any) -> None:
        """Capture invocation error events."""
        if not isinstance(obj, GenAI):
            return

        with self._lock:
            now = time.time()
            self._stats.last_event_time = now

            run_id = str(obj.run_id)
            invocation_type = type(obj).__name__
            # error.type is Type[BaseException], convert to string
            err_type = getattr(error, "type", None)
            error_type = err_type.__name__ if err_type else "unknown"

            event = TelemetryEvent(
                timestamp=now,
                event_type="error",
                invocation_type=invocation_type,
                run_id=run_id,
                trace_id=hex(obj.trace_id) if obj.trace_id else None,
                span_id=hex(obj.span_id) if obj.span_id else None,
                error={
                    "type": error_type,
                    "message": getattr(error, "message", None),
                },
                attributes=dict(obj.attributes) if obj.attributes else {},
            )
            self._events.append(event)

            self._stats.total_errors += 1
            self._stats.errors_by_type[error_type] = (
                self._stats.errors_by_type.get(error_type, 0) + 1
            )

            # Remove from pending if was waiting for evaluation
            self._pending_invocations.discard(run_id)

    def on_evaluation_results(
        self,
        results: Sequence[EvaluationResult],
        obj: Any | None = None,
    ) -> None:
        """Capture evaluation results."""
        if not results:
            return

        with self._lock:
            now = time.time()
            self._stats.last_event_time = now

            run_id = (
                str(obj.run_id)
                if obj and hasattr(obj, "run_id")
                else "unknown"
            )
            invocation_type = type(obj).__name__ if obj else "unknown"

            # Convert results to serializable format
            results_data = []
            for result in results:
                result_dict = {
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "label": result.label,
                    "explanation": result.explanation,
                }
                results_data.append(result_dict)

                # Track by metric name
                metric_name = (
                    result.metric_name.lower()
                    if result.metric_name
                    else "unknown"
                )
                self._stats.evaluation_results_by_metric[metric_name] = (
                    self._stats.evaluation_results_by_metric.get(
                        metric_name, 0
                    )
                    + 1
                )

            event = TelemetryEvent(
                timestamp=now,
                event_type="evaluation_results",
                invocation_type=invocation_type,
                run_id=run_id,
                trace_id=hex(obj.trace_id) if obj and obj.trace_id else None,
                span_id=hex(obj.span_id) if obj and obj.span_id else None,
                evaluation_results=results_data,
            )
            self._events.append(event)

            # Store results for correlation
            if run_id not in self._evaluation_results:
                self._evaluation_results[run_id] = []
            self._evaluation_results[run_id].extend(results)

            self._stats.total_evaluation_results += len(results)

            # Mark as evaluated
            self._pending_invocations.discard(run_id)

    # Public API for testing --------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics as a dictionary."""
        with self._lock:
            return {
                "total_starts": self._stats.total_starts,
                "total_ends": self._stats.total_ends,
                "total_errors": self._stats.total_errors,
                "total_evaluation_results": self._stats.total_evaluation_results,
                "evaluation_results_by_metric": dict(
                    self._stats.evaluation_results_by_metric
                ),
                "invocations_by_type": dict(self._stats.invocations_by_type),
                "errors_by_type": dict(self._stats.errors_by_type),
                "pending_evaluations": len(self._pending_invocations),
                "start_time": self._stats.start_time,
                "last_event_time": self._stats.last_event_time,
                "duration": (
                    self._stats.last_event_time - self._stats.start_time
                    if self._stats.start_time and self._stats.last_event_time
                    else None
                ),
            }

    def get_events(self) -> List[TelemetryEvent]:
        """Get all captured events."""
        with self._lock:
            return list(self._events)

    def get_evaluation_results(
        self, run_id: Optional[str] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """Get evaluation results, optionally filtered by run_id."""
        with self._lock:
            if run_id:
                return {run_id: list(self._evaluation_results.get(run_id, []))}
            return {k: list(v) for k, v in self._evaluation_results.items()}

    def get_pending_count(self) -> int:
        """Get count of invocations still waiting for evaluation results."""
        with self._lock:
            return len(self._pending_invocations)

    def get_invocations_with_results(self) -> int:
        """Get count of invocations that received evaluation results."""
        with self._lock:
            return len(self._evaluation_results)

    def wait_for_evaluations(
        self,
        expected_count: int,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait for expected number of evaluation results.

        Args:
            expected_count: Number of evaluation result batches to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between polls in seconds

        Returns:
            True if expected count reached, False if timeout
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._evaluation_results) >= expected_count:
                    return True
            time.sleep(poll_interval)
        return False

    def export_to_json(self, filepath: str) -> None:
        """Export all captured telemetry to a JSON file."""
        with self._lock:
            stats_data = {
                "total_starts": self._stats.total_starts,
                "total_ends": self._stats.total_ends,
                "total_errors": self._stats.total_errors,
                "total_evaluation_results": self._stats.total_evaluation_results,
                "evaluation_results_by_metric": dict(
                    self._stats.evaluation_results_by_metric
                ),
                "invocations_by_type": dict(self._stats.invocations_by_type),
                "errors_by_type": dict(self._stats.errors_by_type),
                "pending_evaluations": len(self._pending_invocations),
                "start_time": self._stats.start_time,
                "last_event_time": self._stats.last_event_time,
                "duration": (
                    self._stats.last_event_time - self._stats.start_time
                    if self._stats.start_time and self._stats.last_event_time
                    else None
                ),
            }
            data = {
                "export_time": datetime.now().isoformat(),
                "stats": stats_data,
                "events": [asdict(e) for e in self._events],
            }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        _LOGGER.info("Exported telemetry to %s", filepath)

    def reset(self) -> None:
        """Reset all captured data and statistics."""
        with self._lock:
            self._events.clear()
            self._stats = TestEmitterStats()
            self._invocations.clear()
            self._evaluation_results.clear()
            self._pending_invocations.clear()
        _LOGGER.debug("Test emitter reset")

    def print_summary(self) -> None:
        """Print a human-readable summary of captured telemetry."""
        stats = self.get_stats()
        duration = stats.get("duration")
        duration_str = f"{duration:.2f}s" if duration else "N/A"

        print("\n" + "=" * 60)
        print("TEST EMITTER SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration_str}")
        print(f"Total Starts: {stats['total_starts']}")
        print(f"Total Ends: {stats['total_ends']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Total Evaluation Results: {stats['total_evaluation_results']}")
        print(f"Pending Evaluations: {stats['pending_evaluations']}")
        print()
        print("Invocations by Type:")
        for inv_type, count in stats["invocations_by_type"].items():
            print(f"  {inv_type}: {count}")
        print()
        print("Evaluation Results by Metric:")
        for metric, count in stats["evaluation_results_by_metric"].items():
            print(f"  {metric}: {count}")
        if stats["errors_by_type"]:
            print()
            print("Errors by Type:")
            for err_type, count in stats["errors_by_type"].items():
                print(f"  {err_type}: {count}")
        print("=" * 60 + "\n")


def get_test_emitter() -> TestEmitter:
    """Get the singleton TestEmitter instance."""
    global _test_emitter_instance
    if _test_emitter_instance is None:
        with _test_emitter_lock:
            if _test_emitter_instance is None:
                _test_emitter_instance = TestEmitter()
    return _test_emitter_instance


def _emitter_factory(context: Any) -> TestEmitter:
    """Factory function for creating TestEmitter via plugin system."""
    return get_test_emitter()


def load_emitters() -> List[Dict[str, Any]]:
    """Entry point for emitter plugin discovery.

    Returns:
        List of emitter spec dicts compatible with the plugin system
    """
    # Register the test emitter in multiple categories to capture all events
    # - span: captures on_start, on_end, on_error
    # - evaluation: captures on_evaluation_results
    return [
        {
            "name": "test",
            "category": "span",
            "factory": _emitter_factory,
        },
        {
            "name": "test",
            "category": "evaluation",
            "factory": _emitter_factory,
        },
    ]


__all__ = [
    "TestEmitter",
    "TelemetryEvent",
    "TestEmitterStats",
    "get_test_emitter",
    "load_emitters",
]
