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

"""Evaluation monitoring metrics for LLM-as-a-judge operations.

This module provides the ``EvaluationMonitor`` class which emits metrics for:
- Duration of evaluation LLM calls
- Token usage for evaluation LLM calls
- Evaluation queue size
- Enqueue errors

Enable via ``OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

from opentelemetry.metrics import Counter, Histogram, Meter, get_meter

from ..version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from queue import Queue

_LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Context for a single evaluation operation.

    Captures timing and metadata for metrics emission.
    """

    evaluation_name: str
    evaluator_name: str = "deepeval"
    request_model: Optional[str] = None
    provider: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_type: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def elapsed_seconds(self) -> float:
        """Return elapsed time since start in seconds."""
        return time.perf_counter() - self.start_time


class EvaluationMonitor:
    """Monitors evaluation operations and emits metrics.

    Metrics emitted:
    - gen_ai.evaluation.client.operation.duration: Histogram of evaluation durations
    - gen_ai.evaluation.client.token.usage: Histogram of token usage
    - gen_ai.evaluation.client.queue.size: Observable gauge of queue size
    - gen_ai.evaluation.client.enqueue.errors: Counter of enqueue failures
    """

    def __init__(
        self,
        meter: Optional[Meter] = None,
        evaluation_queue: Optional["Queue[Any]"] = None,
    ) -> None:
        """Initialize the evaluation monitor.

        Args:
            meter: Optional OpenTelemetry Meter. If not provided, uses global.
            evaluation_queue: Optional queue to observe for size metrics.
        """
        self._meter = meter or get_meter(__name__, __version__)
        self._queue = evaluation_queue
        self._queue_size = 0  # Track queue size for observable gauge
        self._lock = threading.Lock()

        # Create metric instruments
        self._duration_histogram: Histogram = self._meter.create_histogram(
            name="gen_ai.evaluation.client.operation.duration",
            unit="s",
            description="Duration of GenAI evaluation LLM-as-a-judge operations",
        )

        self._token_histogram: Histogram = self._meter.create_histogram(
            name="gen_ai.evaluation.client.token.usage",
            unit="{token}",
            description="Number of tokens used by evaluation LLM-as-a-judge operations",
        )

        self._enqueue_error_counter: Counter = self._meter.create_counter(
            name="gen_ai.evaluation.client.enqueue.errors",
            unit="{error}",
            description="Count of evaluation enqueue failures",
        )

        # Observable gauge for queue size using callback
        self._meter.create_observable_gauge(
            name="gen_ai.evaluation.client.queue.size",
            callbacks=[self._observe_queue_size],
            unit="{invocation}",
            description="Current size of the evaluation queue",
        )

        _LOGGER.debug("EvaluationMonitor initialized")

    def _observe_queue_size(self, options: Any) -> Any:
        """Callback for observable gauge to report queue size."""
        from opentelemetry.metrics import Observation

        with self._lock:
            size = self._queue_size
        return [Observation(size)]

    def on_enqueue(self) -> None:
        """Record that an invocation was enqueued for evaluation."""
        with self._lock:
            self._queue_size += 1

    def on_dequeue(self) -> None:
        """Record that an invocation was dequeued for processing."""
        with self._lock:
            self._queue_size = max(0, self._queue_size - 1)

    def on_enqueue_error(
        self,
        error_type: str = "queue_full",
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record an enqueue error.

        Args:
            error_type: Type of error (e.g., "queue_full", "queue_error")
            attributes: Additional attributes to include in the metric
        """
        attrs: dict[str, Any] = {"error.type": error_type}
        if attributes:
            attrs.update(attributes)
        self._enqueue_error_counter.add(1, attrs)
        _LOGGER.debug("Recorded enqueue error: %s", error_type)

    def record_evaluation(
        self,
        ctx: EvaluationContext,
    ) -> None:
        """Record metrics for a completed evaluation.

        Args:
            ctx: Evaluation context with timing and metadata
        """
        duration_seconds = ctx.elapsed_seconds()

        # Build common attributes
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "evaluate",
            "gen_ai.evaluation.name": ctx.evaluation_name,
        }

        if ctx.evaluator_name:
            attrs["gen_ai.evaluation.evaluator"] = ctx.evaluator_name

        if ctx.request_model:
            attrs["gen_ai.request.model"] = ctx.request_model

        if ctx.provider:
            attrs["gen_ai.system"] = ctx.provider

        if ctx.error_type:
            attrs["error.type"] = ctx.error_type

        # Add any extra attributes from context
        attrs.update(ctx.attributes)

        # Record duration
        self._duration_histogram.record(duration_seconds, attrs)

        # Record token usage if available
        if ctx.input_tokens is not None or ctx.output_tokens is not None:
            input_tokens = ctx.input_tokens or 0
            output_tokens = ctx.output_tokens or 0

            if input_tokens > 0:
                token_attrs = {**attrs, "gen_ai.token.type": "input"}
                self._token_histogram.record(input_tokens, token_attrs)

            if output_tokens > 0:
                token_attrs = {**attrs, "gen_ai.token.type": "output"}
                self._token_histogram.record(output_tokens, token_attrs)

        _LOGGER.debug(
            "Recorded evaluation: %s duration=%.3fs input_tokens=%s output_tokens=%s",
            ctx.evaluation_name,
            duration_seconds,
            ctx.input_tokens,
            ctx.output_tokens,
        )

    def start_evaluation(
        self,
        evaluation_name: str,
        evaluator_name: str = "deepeval",
        request_model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> EvaluationContext:
        """Start timing an evaluation operation.

        Args:
            evaluation_name: Name of the evaluation metric (e.g., "bias", "toxicity")
            evaluator_name: Name of the evaluator (e.g., "deepeval")
            request_model: Model used for evaluation (e.g., "gpt-4o-mini")
            provider: Provider for the evaluation model (e.g., "openai")

        Returns:
            EvaluationContext to be passed to record_evaluation()
        """
        return EvaluationContext(
            evaluation_name=evaluation_name,
            evaluator_name=evaluator_name,
            request_model=request_model,
            provider=provider,
            start_time=time.perf_counter(),
        )


# Global monitor instance (lazy initialization)
_monitor: Optional[EvaluationMonitor] = None
_monitor_lock = threading.Lock()


def get_evaluation_monitor(
    meter: Optional[Meter] = None,
) -> EvaluationMonitor:
    """Get or create the global EvaluationMonitor instance.

    Args:
        meter: Optional meter to use. Only used on first call.

    Returns:
        The global EvaluationMonitor instance.
    """
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = EvaluationMonitor(meter=meter)
        return _monitor


def reset_evaluation_monitor() -> None:
    """Reset the global monitor instance (for testing)."""
    global _monitor
    with _monitor_lock:
        _monitor = None


__all__ = [
    "EvaluationMonitor",
    "EvaluationContext",
    "get_evaluation_monitor",
    "reset_evaluation_monitor",
]
