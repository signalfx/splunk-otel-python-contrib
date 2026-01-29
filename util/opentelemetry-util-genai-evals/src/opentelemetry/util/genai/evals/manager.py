from __future__ import annotations  # noqa: I001

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from ..callbacks import CompletionCallback

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..handler import TelemetryHandler

from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)

from ..environment_variables import OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS
from ..types import (
    AgentCreation,
    AgentInvocation,
    EmbeddingInvocation,
    EvaluationResult,
    GenAI,
    LLMInvocation,
    Step,
    ToolCall,
    Workflow,
)
from .admission_controller import EvaluationAdmissionController
from .base import Evaluator
from .env import (
    read_aggregation_flag,
    read_concurrent_flag,
    read_interval,
    read_queue_size,
    read_raw_evaluators,
    read_worker_count,
)
from .errors import ErrorTracker
from .normalize import is_tool_only_llm
from .registry import get_default_metrics, get_evaluator, list_evaluators

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..handler import TelemetryHandler

_LOGGER = logging.getLogger(__name__)


class Sampler(Protocol):
    def should_sample(self, invocation: GenAI) -> bool: ...


class _AllSampler:
    def should_sample(
        self, invocation: GenAI
    ) -> bool:  # pragma: no cover - trivial
        return True


@dataclass(frozen=True)
class MetricConfig:
    name: str
    options: Mapping[str, str]


@dataclass(frozen=True)
class EvaluatorPlan:
    name: str
    per_type: Mapping[str, Sequence[MetricConfig]]


_GENAI_TYPE_LOOKUP: Mapping[str, type[GenAI]] = {
    "LLMInvocation": LLMInvocation,
    "AgentInvocation": AgentInvocation,
    "AgentCreation": AgentCreation,
    "EmbeddingInvocation": EmbeddingInvocation,
    "ToolCall": ToolCall,
    "Workflow": Workflow,
    "Step": Step,
}


class Manager(CompletionCallback):
    """Asynchronous evaluation manager implementing the completion callback.

    Supports two processing modes controlled by OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT:

    1. Sequential mode (default): Single worker thread processes evaluations one at a time.
       Suitable for low-volume workloads or when LLM API rate limits are strict.

    2. Concurrent mode: Multiple worker threads process evaluations in parallel with
       async LLM calls. Significantly improves throughput for LLM-as-a-judge evaluations.
       Configure worker count with OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS.
    """

    def __init__(
        self,
        handler: "TelemetryHandler",
        *,
        interval: float | None = None,
        aggregate_results: bool | None = None,
        queue_size: int | None = None,
        concurrent_mode: bool | None = None,
        worker_count: int | None = None,
    ) -> None:
        self._handler = handler
        self._interval = interval if interval is not None else read_interval()
        self._aggregate_results = (
            aggregate_results
            if aggregate_results is not None
            else read_aggregation_flag()
        )
        self._error_tracker = ErrorTracker()
        self._admission = EvaluationAdmissionController()
        self._plans = self._load_plans()
        self._evaluators = self._instantiate_evaluators(self._plans)

        # Queue configuration with bounded size for backpressure
        self._queue_size = (
            queue_size if queue_size is not None else read_queue_size()
        )
        self._queue: queue.Queue[GenAI] = queue.Queue(maxsize=self._queue_size)
        _LOGGER.debug(
            "Evaluation queue configured with size: %d", self._queue_size
        )

        # Concurrent mode configuration
        self._concurrent_mode = (
            concurrent_mode
            if concurrent_mode is not None
            else read_concurrent_flag()
        )
        self._worker_count = (
            worker_count if worker_count is not None else read_worker_count()
        )

        self._shutdown = threading.Event()
        self._workers: list[threading.Thread] = []

        if self.has_evaluators:
            if self._concurrent_mode:
                # Concurrent mode: multiple workers with async support
                _LOGGER.info(
                    "Starting concurrent evaluation mode with %d workers",
                    self._worker_count,
                )
                for i in range(self._worker_count):
                    worker = threading.Thread(
                        target=self._concurrent_worker_loop,
                        name=f"genai-evaluator-{i}",
                        daemon=True,
                    )
                    self._workers.append(worker)
                    worker.start()
            else:
                # Sequential mode (legacy): single worker
                _LOGGER.debug(
                    "Starting sequential evaluation mode (single worker)"
                )
                worker = threading.Thread(
                    target=self._worker_loop,
                    name="opentelemetry-genai-evaluator",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()

    @property
    def concurrent_mode(self) -> bool:
        """Whether concurrent evaluation mode is enabled."""
        return self._concurrent_mode

    # CompletionCallback -------------------------------------------------
    def on_completion(self, invocation: GenAI) -> None:
        # Early exit if no evaluators configured
        if not self.has_evaluators:
            return

        # Only evaluate LLMInvocation or AgentInvocation or Workflow
        if (
            not isinstance(invocation, LLMInvocation)
            and not isinstance(invocation, AgentInvocation)
            and not isinstance(invocation, Workflow)
        ):
            invocation.evaluation_error = (
                "client_evaluation_skipped_as_invocation_type_not_supported"
            )
            _LOGGER.debug(
                "Skipping evaluation for invocation type: %s. Only support LLM, Agent and Workflow invocation types.",
                type(invocation).__name__,
            )
            return

        offer: bool = True
        if invocation.sample_for_evaluation:
            # Do not evaluate if llm invocation is for tool invocation because it will not have output message for evaluations tests case.
            if isinstance(invocation, LLMInvocation):
                msgs = getattr(invocation, "output_messages", [])
                if msgs:
                    first = msgs[0]
                    if (
                        first.parts
                        and first.parts[0] == "ToolCall"
                        and first.finish_reason == "tool_calls"
                    ):
                        invocation.evaluation_error = "client_evaluation_skipped_as_tool_llm_invocation_type_not_supported"
                        _LOGGER.debug(
                            "Skipping evaluation for type tool llm invocation: %s. No output to evaluate.",
                            type(invocation).__name__,
                        )
                        offer = False

            # Do not evaluate if error
            error = invocation.attributes.get(ErrorAttributes.ERROR_TYPE)
            if error:
                invocation.evaluation_error = (
                    "client_evaluation_skipped_as_error_on_invocation"
                )
                _LOGGER.debug(
                    "Skipping evaluation for invocation type: %s as error on span, error: %s.",
                    type(invocation).__name__,
                    error,
                )
                offer = False

            if offer:
                self.offer(invocation)

    # Public API ---------------------------------------------------------
    def offer(self, invocation: GenAI) -> None:
        """Enqueue an invocation for asynchronous evaluation.

        If the queue is bounded and full, the invocation is dropped with a warning.
        This implements backpressure to prevent memory exhaustion under heavy load.
        """
        if not self.has_evaluators:
            return
        try:
            self._queue.put_nowait(invocation)
        except queue.Full:
            # Bounded queue is full - apply backpressure by dropping
            invocation.evaluation_error = "client_evaluation_queue_full"
            invocation_id = getattr(invocation, "span_id", None) or getattr(
                invocation, "trace_id", None
            )
            _LOGGER.warning(
                "Evaluation queue full. Dropping invocation.",
                extra={
                    "error_type": "queue_full",
                    "component": "manager",
                    "invocation_type": type(invocation).__name__,
                    "invocation_id": invocation_id,
                    "queue_size": self._queue_size
                    if self._queue_size > 0
                    else self._queue.maxsize,
                    "queue_depth": self._queue.qsize(),
                },
            )
            self._error_tracker.record_error(
                error_type="queue_full",
                component="manager",
                message="Evaluation queue full, dropping invocation",
                invocation_id=invocation_id,
                recovery_action="invocation_dropped",
                operational_impact="Evaluation skipped due to backpressure",
                details={
                    "invocation_type": type(invocation).__name__,
                    "queue_size": self._queue_size
                    if self._queue_size > 0
                    else self._queue.maxsize,
                    "queue_depth": self._queue.qsize(),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            invocation_id = getattr(invocation, "span_id", None) or getattr(
                invocation, "trace_id", None
            )
            _LOGGER.error(
                "Failed to enqueue invocation for evaluation",
                extra={
                    "error_type": "queue_error",
                    "component": "manager",
                    "invocation_type": type(invocation).__name__,
                    "invocation_id": invocation_id,
                    "exception_type": type(exc).__name__,
                },
                exc_info=True,
            )
            self._error_tracker.record_error(
                error_type="queue_error",
                component="manager",
                message="Failed to enqueue invocation",
                invocation_id=invocation_id,
                exception=exc,
                recovery_action="invocation_dropped",
                operational_impact="Evaluation skipped for this invocation",
            )

    def wait_for_all(self, timeout: float | None = None) -> None:
        if not self.has_evaluators:
            return
        if timeout is None:
            self._queue.join()
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return
            time.sleep(0.05)

    def shutdown(self) -> None:
        """Gracefully shutdown evaluation workers."""
        if not self._workers:
            return
        self._shutdown.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers.clear()
        _LOGGER.debug("Evaluation manager shutdown complete")

    def evaluate_now(self, invocation: GenAI) -> list[EvaluationResult]:
        """Synchronously evaluate an invocation."""
        allowed, reason = self._admission.allow()
        if not allowed:
            _LOGGER.error(
                "Evaluation rate limited (%s), dropping invocation.",
                reason,
            )
            return []
        buckets = self._evaluate_invocation(invocation)
        flattened = self._publish_results(invocation, buckets)
        self._flag_invocation(invocation)
        return flattened

    @property
    def has_evaluators(self) -> bool:
        return any(self._evaluators.values())

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of tracked errors for diagnostics.

        Returns:
            Dictionary with error counts and statistics
        """
        return self._error_tracker.get_error_summary()

    # Internal helpers ---------------------------------------------------
    def _worker_loop(self) -> None:
        """Sequential worker loop (legacy mode)."""
        worker_name = threading.current_thread().name
        while not self._shutdown.is_set():
            try:
                invocation = self._queue.get(timeout=self._interval)
            except queue.Empty:
                continue
            try:
                # Apply rate limiting on processing side
                allowed, reason = self._admission.allow()
                if not allowed:
                    _LOGGER.error(
                        "Evaluation rate limited (%s), dropping invocation.",
                        reason,
                    )
                else:
                    try:
                        self._process_invocation(invocation)
                    except Exception as exc:  # pragma: no cover - defensive
                        invocation_id = getattr(
                            invocation, "span_id", None
                        ) or getattr(invocation, "trace_id", None)
                        _LOGGER.error(
                            "Evaluator processing failed",
                            extra={
                                "error_type": "processing_error",
                                "component": "worker",
                                "worker_name": worker_name,
                                "invocation_type": type(invocation).__name__,
                                "invocation_id": invocation_id,
                                "exception_type": type(exc).__name__,
                            },
                            exc_info=True,
                        )
                        self._error_tracker.record_error(
                            error_type="processing_error",
                            component="worker",
                            message="Failed to process invocation",
                            invocation_id=invocation_id,
                            exception=exc,
                            recovery_action="invocation_skipped",
                            operational_impact="No evaluation results for this invocation",
                            worker_name=worker_name,
                        )
            finally:
                self._queue.task_done()

    def _concurrent_worker_loop(self) -> None:
        """Concurrent worker loop with async support.

        Each worker creates its own event loop and processes invocations
        using async evaluation when available.
        """
        worker_name = threading.current_thread().name
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.error(
                "Failed to create asyncio event loop",
                extra={
                    "error_type": "eventloop_error",
                    "component": "concurrent_worker",
                    "worker_name": worker_name,
                    "exception_type": type(exc).__name__,
                },
                exc_info=True,
            )
            self._error_tracker.record_error(
                error_type="eventloop_error",
                component="concurrent_worker",
                message="Failed to create asyncio event loop",
                exception=exc,
                recovery_action="worker_disabled",
                operational_impact="Worker thread cannot process evaluations",
                worker_name=worker_name,
                async_context=False,
            )
            return  # Exit worker gracefully

        try:
            while not self._shutdown.is_set():
                try:
                    invocation = self._queue.get(timeout=self._interval)
                except queue.Empty:
                    continue
                try:
                    # Apply rate limiting on processing side
                    allowed, reason = loop.run_until_complete(
                        self._admission.allow_async()
                    )
                    if not allowed:
                        _LOGGER.error(
                            "Evaluation rate limited (%s), dropping invocation.",
                            reason,
                        )
                    else:
                        loop.run_until_complete(
                            self._process_invocation_async(invocation)
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    invocation_id = getattr(
                        invocation, "span_id", None
                    ) or getattr(invocation, "trace_id", None)
                    _LOGGER.error(
                        "Concurrent evaluator processing failed",
                        extra={
                            "error_type": "concurrent_processing_error",
                            "component": "concurrent_worker",
                            "worker_name": worker_name,
                            "invocation_type": type(invocation).__name__,
                            "invocation_id": invocation_id,
                            "exception_type": type(exc).__name__,
                        },
                        exc_info=True,
                    )
                    self._error_tracker.record_error(
                        error_type="concurrent_processing_error",
                        component="concurrent_worker",
                        message="Failed to process invocation in concurrent mode",
                        invocation_id=invocation_id,
                        exception=exc,
                        recovery_action="invocation_skipped",
                        operational_impact="No evaluation results for this invocation",
                        worker_name=worker_name,
                        async_context=True,
                        details={"invocation_type": type(invocation).__name__},
                    )
                finally:
                    self._queue.task_done()
        finally:
            try:
                loop.close()
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Error closing asyncio event loop",
                    extra={
                        "error_type": "eventloop_close_error",
                        "component": "concurrent_worker",
                        "worker_name": worker_name,
                        "exception_type": type(exc).__name__,
                    },
                    exc_info=True,
                )

    async def _process_invocation_async(self, invocation: GenAI) -> None:
        """Process an invocation asynchronously with concurrent evaluator calls."""
        if not self.has_evaluators:
            return

        buckets = await self._evaluate_invocation_async(invocation)
        self._publish_results(invocation, buckets)
        self._flag_invocation(invocation)

    async def _evaluate_invocation_async(
        self, invocation: GenAI
    ) -> Sequence[Sequence[EvaluationResult]]:
        """Evaluate an invocation using concurrent async evaluator calls."""
        if not self.has_evaluators:
            return ()
        if self._should_skip(invocation):
            return ()

        type_name = type(invocation).__name__
        evaluators = self._evaluators.get(type_name, ())
        if not evaluators:
            return ()

        invocation_id = getattr(invocation, "span_id", None) or getattr(
            invocation, "trace_id", None
        )
        worker_name = threading.current_thread().name

        # Run all evaluators concurrently
        async def run_evaluator(
            evaluator: Evaluator,
        ) -> Sequence[EvaluationResult] | None:
            evaluator_name = getattr(
                evaluator, "__class__", type(evaluator)
            ).__name__
            used_async = hasattr(evaluator, "evaluate_async")
            try:
                # Use async evaluation if available, otherwise run sync in thread pool
                if used_async:
                    return await evaluator.evaluate_async(invocation)
                else:
                    # Run sync evaluate in thread pool to not block
                    return await asyncio.to_thread(
                        evaluator.evaluate, invocation
                    )
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Async evaluator failed",
                    extra={
                        "error_type": "async_evaluator_error",
                        "component": "async_evaluator",
                        "evaluator_name": evaluator_name,
                        "invocation_type": type_name,
                        "invocation_id": invocation_id,
                        "worker_name": worker_name,
                        "async_mode": used_async,
                        "exception_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                self._error_tracker.record_error(
                    error_type="async_evaluator_error",
                    component="async_evaluator",
                    message=f"Async evaluator {evaluator_name} failed",
                    evaluator_name=evaluator_name,
                    invocation_id=invocation_id,
                    exception=exc,
                    recovery_action="skip_evaluator",
                    operational_impact="Partial evaluation results",
                    worker_name=worker_name,
                    async_context=True,
                    details={
                        "invocation_type": type_name,
                        "async_mode": used_async,
                    },
                )
                return None

        # Execute all evaluators concurrently
        tasks = [run_evaluator(evaluator) for evaluator in evaluators]
        results_list = await asyncio.gather(*tasks, return_exceptions=False)

        # Collect non-empty results
        buckets: list[Sequence[EvaluationResult]] = []
        for results in results_list:
            if results:
                buckets.append(list(results))

        return buckets

    def _process_invocation(self, invocation: GenAI) -> None:
        if not self.has_evaluators:
            return
        buckets = self._evaluate_invocation(invocation)
        self._publish_results(invocation, buckets)
        self._flag_invocation(invocation)

    def _evaluate_invocation(
        self, invocation: GenAI
    ) -> Sequence[Sequence[EvaluationResult]]:
        if not self.has_evaluators:
            return ()
        if self._should_skip(invocation):
            return ()
        type_name = type(invocation).__name__
        evaluators = self._evaluators.get(type_name, ())
        if not evaluators:
            return ()
        buckets: list[Sequence[EvaluationResult]] = []
        invocation_id = getattr(invocation, "span_id", None) or getattr(
            invocation, "trace_id", None
        )
        for descriptor in evaluators:
            evaluator_name = getattr(
                descriptor, "__class__", type(descriptor)
            ).__name__
            try:
                results = descriptor.evaluate(invocation)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Evaluator failed",
                    extra={
                        "error_type": "evaluator_error",
                        "component": "evaluator",
                        "evaluator_name": evaluator_name,
                        "invocation_type": type_name,
                        "invocation_id": invocation_id,
                        "exception_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                self._error_tracker.record_error(
                    error_type="evaluator_error",
                    component="evaluator",
                    message=f"Evaluator {evaluator_name} failed",
                    evaluator_name=evaluator_name,
                    invocation_id=invocation_id,
                    exception=exc,
                    recovery_action="skip_evaluator",
                    operational_impact="Partial evaluation results",
                    details={"invocation_type": type_name},
                )
                continue
            if results:
                buckets.append(list(results))
        return buckets

    def _publish_results(
        self,
        invocation: GenAI,
        buckets: Sequence[Sequence[EvaluationResult]],
    ) -> list[EvaluationResult]:
        if not buckets:
            return []
        # Central aggregation: if enabled we collapse all evaluator buckets into
        # a single list and emit exactly once. This shifts any downstream
        # aggregation burden (e.g., Splunk single-event formatting) out of the
        # emitters and into this manager loop.
        aggregate = self._aggregate_results
        if aggregate is None:
            env_flag = read_aggregation_flag()
            aggregate = bool(env_flag)
        flattened: list[EvaluationResult] = []
        for bucket in buckets:
            flattened.extend(bucket)

        invocation_id = getattr(invocation, "span_id", None) or getattr(
            invocation, "trace_id", None
        )

        if aggregate:
            if flattened:
                attrs = getattr(invocation, "attributes", None)
                if isinstance(attrs, dict):
                    attrs.setdefault("gen_ai.evaluation.aggregated", True)
                try:
                    self._handler.evaluation_results(invocation, flattened)
                except Exception as exc:
                    _LOGGER.error(
                        "Handler evaluation_results callback failed",
                        extra={
                            "error_type": "handler_error",
                            "component": "manager",
                            "invocation_id": invocation_id,
                            "result_count": len(flattened),
                            "exception_type": type(exc).__name__,
                        },
                        exc_info=True,
                    )
                    self._error_tracker.record_error(
                        error_type="handler_error",
                        component="manager",
                        message="Failed to publish aggregated evaluation results",
                        invocation_id=invocation_id,
                        exception=exc,
                        recovery_action="results_dropped",
                        operational_impact="Evaluation results lost",
                        severity="error",
                        details={
                            "result_count": len(flattened),
                            "aggregated": True,
                        },
                    )
            return flattened
        # Non-aggregated path: emit each bucket individually (legacy behavior)
        for bucket in buckets:
            if bucket:
                try:
                    self._handler.evaluation_results(invocation, list(bucket))
                except Exception as exc:
                    _LOGGER.error(
                        "Handler evaluation_results callback failed",
                        extra={
                            "error_type": "handler_error",
                            "component": "manager",
                            "invocation_id": invocation_id,
                            "result_count": len(bucket),
                            "exception_type": type(exc).__name__,
                        },
                        exc_info=True,
                    )
                    self._error_tracker.record_error(
                        error_type="handler_error",
                        component="manager",
                        message="Failed to publish evaluation results bucket",
                        invocation_id=invocation_id,
                        exception=exc,
                        recovery_action="bucket_dropped",
                        operational_impact="Partial evaluation results lost",
                        severity="error",
                        details={
                            "result_count": len(bucket),
                            "aggregated": False,
                        },
                    )
        return flattened

    def _should_skip(self, invocation: GenAI) -> bool:
        """Centralised evaluation skip policy."""
        try:
            if isinstance(invocation, LLMInvocation):
                if is_tool_only_llm(invocation):
                    _LOGGER.debug(
                        "Skipping evaluation for tool-only LLM output",
                        extra={
                            "skip_reason": "tool_only_llm",
                            "invocation_type": type(invocation).__name__,
                        },
                    )
                    return True
            elif isinstance(invocation, AgentCreation):
                _LOGGER.debug(
                    "Skipping evaluation for agent creation",
                    extra={
                        "skip_reason": "agent_creation",
                        "invocation_type": type(invocation).__name__,
                    },
                )
                return True
            elif isinstance(invocation, AgentInvocation):
                operation = getattr(invocation, "operation", "invoke_agent")
                if operation != "invoke_agent":
                    _LOGGER.debug(
                        "Skipping evaluation for non-invoke agent operation",
                        extra={
                            "skip_reason": "non_invoke_operation",
                            "operation": operation,
                            "invocation_type": type(invocation).__name__,
                        },
                    )
                    return True
        except Exception as exc:  # pragma: no cover - defensive
            invocation_id = getattr(invocation, "span_id", None) or getattr(
                invocation, "trace_id", None
            )
            _LOGGER.warning(
                "Skip policy evaluation failed",
                extra={
                    "error_type": "skip_policy_error",
                    "component": "manager",
                    "invocation_type": type(invocation).__name__,
                    "invocation_id": invocation_id,
                    "exception_type": type(exc).__name__,
                },
                exc_info=True,
            )
            self._error_tracker.record_error(
                error_type="skip_policy_error",
                component="manager",
                message="Skip policy evaluation failed",
                invocation_id=invocation_id,
                exception=exc,
                recovery_action="assume_not_skipped",
                operational_impact="Invocation may be evaluated incorrectly",
                severity="warning",
            )
        return False

    def _flag_invocation(self, invocation: GenAI) -> None:
        if not self.has_evaluators:
            return
        attributes = getattr(invocation, "attributes", None)
        if isinstance(attributes, dict):
            attributes.setdefault("gen_ai.evaluation.executed", True)

    # Configuration ------------------------------------------------------
    def _load_plans(self) -> Sequence[EvaluatorPlan]:
        raw_value = read_raw_evaluators()
        raw = (raw_value or "").strip()
        normalized = raw.lower()
        if normalized in {"none", "off", "false"}:
            _LOGGER.info(
                "GenAI evaluations disabled via %s",
                OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
            )
            return []
        if not raw:
            # Auto-discover defaults when no explicit config provided.
            plans = self._generate_default_plans()
            if not plans:
                _LOGGER.info(
                    "GenAI evaluations disabled (no defaults registered); set %s to enable specific evaluators",
                    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
                )
            else:
                _LOGGER.debug(
                    "Auto-discovered evaluator default metrics: %s",
                    [p.name for p in plans],
                )
            return plans
        try:
            requested = _parse_evaluator_config(raw)
        except ValueError as exc:
            available = list_evaluators()
            _LOGGER.error(
                "Failed to parse evaluator configuration",
                extra={
                    "error_type": "config_parse_error",
                    "component": "manager",
                    "config_value": raw[:200]
                    if len(raw) <= 200
                    else raw[:200] + "...",
                    "exception_type": type(exc).__name__,
                    "available_evaluators": available,
                },
                exc_info=True,
            )
            self._error_tracker.record_error(
                error_type="config_parse_error",
                component="manager",
                message=f"Failed to parse evaluator configuration: {exc}",
                exception=exc,
                recovery_action="evaluations_disabled",
                operational_impact="All evaluations disabled",
                severity="error",
                details={
                    "config_snippet": raw[:200]
                    if len(raw) <= 200
                    else raw[:200] + "...",
                    "available_evaluators": available,
                },
            )
            return []
        available = list_evaluators()
        available_lower = {name.lower() for name in available}
        plans: list[EvaluatorPlan] = []
        for spec in requested:
            if spec.name.lower() not in available_lower:
                _LOGGER.error(
                    "Unknown evaluator requested",
                    extra={
                        "error_type": "unknown_evaluator",
                        "component": "manager",
                        "evaluator_name": spec.name,
                        "available_evaluators": sorted(available),
                    },
                )
                self._error_tracker.record_error(
                    error_type="unknown_evaluator",
                    component="manager",
                    message=f"Evaluator '{spec.name}' is not registered",
                    evaluator_name=spec.name,
                    recovery_action="evaluator_skipped",
                    operational_impact="This evaluator will not run",
                    severity="error",
                    details={"available_evaluators": sorted(available)},
                )
                continue
            try:
                defaults = get_default_metrics(spec.name)
            except ValueError:
                defaults = {}
            per_type: dict[str, Sequence[MetricConfig]] = {}
            if spec.per_type:
                for type_name, metrics in spec.per_type.items():
                    per_type[type_name] = metrics
            else:
                per_type = {
                    key: [MetricConfig(name=m, options={}) for m in value]
                    for key, value in defaults.items()
                }
            if not per_type:
                _LOGGER.debug(
                    "Evaluator '%s' does not declare any metrics", spec.name
                )
                continue
            plans.append(
                EvaluatorPlan(
                    name=spec.name,
                    per_type=per_type,
                )
            )
        return plans

    def _instantiate_evaluators(
        self, plans: Sequence[EvaluatorPlan]
    ) -> Mapping[str, Sequence[Evaluator]]:
        evaluators_by_type: dict[str, list[Evaluator]] = {}
        for plan in plans:
            for type_name, metrics in plan.per_type.items():
                if type_name not in _GENAI_TYPE_LOOKUP:
                    _LOGGER.error(
                        "Unsupported GenAI invocation type",
                        extra={
                            "error_type": "unsupported_type",
                            "component": "manager",
                            "evaluator_name": plan.name,
                            "requested_type": type_name,
                            "supported_types": list(_GENAI_TYPE_LOOKUP.keys()),
                        },
                    )
                    self._error_tracker.record_error(
                        error_type="unsupported_type",
                        component="manager",
                        message=f"Unsupported invocation type '{type_name}' for evaluator '{plan.name}'",
                        evaluator_name=plan.name,
                        recovery_action="type_skipped",
                        operational_impact="Evaluator won't run for this invocation type",
                        severity="error",
                        details={
                            "requested_type": type_name,
                            "supported_types": list(_GENAI_TYPE_LOOKUP.keys()),
                        },
                    )
                    continue
                metric_names = [metric.name for metric in metrics]
                options: Mapping[str, Mapping[str, str]] = {
                    metric.name: metric.options
                    for metric in metrics
                    if metric.options
                }
                try:
                    evaluator = get_evaluator(
                        plan.name,
                        metric_names,
                        invocation_type=type_name,
                        options=options,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    _LOGGER.error(
                        "Evaluator instantiation failed",
                        extra={
                            "error_type": "instantiation_error",
                            "component": "manager",
                            "evaluator_name": plan.name,
                            "invocation_type": type_name,
                            "metrics": metric_names,
                            "exception_type": type(exc).__name__,
                        },
                        exc_info=True,
                    )
                    self._error_tracker.record_error(
                        error_type="instantiation_error",
                        component="manager",
                        message=f"Failed to instantiate evaluator '{plan.name}' for type '{type_name}'",
                        evaluator_name=plan.name,
                        exception=exc,
                        recovery_action="evaluator_disabled",
                        operational_impact="This evaluator won't run for this invocation type",
                        severity="error",
                        details={
                            "invocation_type": type_name,
                            "metrics": metric_names,
                        },
                    )
                    continue
                evaluators_by_type.setdefault(type_name, []).append(evaluator)
        return evaluators_by_type

    def _generate_default_plans(self) -> Sequence[EvaluatorPlan]:
        plans: list[EvaluatorPlan] = []
        available = list_evaluators()
        if not available:
            _LOGGER.info(
                "No evaluator entry points registered; skipping evaluations"
            )
            return plans
        for name in available:
            try:
                defaults = get_default_metrics(name)
            except ValueError:
                continue
            if not defaults:
                continue
            per_type: dict[str, Sequence[MetricConfig]] = {}
            for type_name, metrics in defaults.items():
                filtered_metrics: list[str] = []
                for metric in metrics:
                    if metric is None:
                        continue
                    text = str(metric).strip()
                    if not text:
                        continue
                    if text.lower() == "length":
                        continue
                    filtered_metrics.append(text)
                entries = [
                    MetricConfig(name=metric, options={})
                    for metric in filtered_metrics
                ]
                if entries:
                    per_type[type_name] = entries
            if not per_type:
                continue
            plans.append(EvaluatorPlan(name=name, per_type=per_type))
        if not plans:
            _LOGGER.warning(
                "No evaluators declared default metrics; set %s to an explicit list to enable evaluations",
                OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
            )
        return plans


# ---------------------------------------------------------------------------
# Evaluator configuration parser


@dataclass
class _EvaluatorSpec:
    name: str
    per_type: Mapping[str, Sequence[MetricConfig]]


class _ConfigParser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._length = len(text)
        self._pos = 0

    def parse(self) -> Sequence[_EvaluatorSpec]:
        specs: list[_EvaluatorSpec] = []
        while True:
            self._skip_ws()
            if self._pos >= self._length:
                break
            specs.append(self._parse_evaluator())
            self._skip_ws()
            if self._pos >= self._length:
                break
            self._expect(",")
        return specs

    def _parse_evaluator(self) -> _EvaluatorSpec:
        name = self._parse_identifier()
        per_type: dict[str, Sequence[MetricConfig]] = {}
        self._skip_ws()
        if self._peek() == "(":
            self._advance()
            while True:
                self._skip_ws()
                type_name = self._parse_identifier()
                metrics: list[MetricConfig] = []
                self._skip_ws()
                if self._peek() == "(":
                    self._advance()
                    while True:
                        self._skip_ws()
                        metrics.append(self._parse_metric())
                        self._skip_ws()
                        char = self._peek()
                        if char == ",":
                            self._advance()
                            continue
                        if char == ")":
                            self._advance()
                            break
                        raise ValueError(
                            f"Unexpected character '{char}' while parsing metrics"
                        )
                per_type[type_name] = metrics
                self._skip_ws()
                char = self._peek()
                if char == ",":
                    self._advance()
                    continue
                if char == ")":
                    self._advance()
                    break
                raise ValueError(
                    f"Unexpected character '{char}' while parsing type configuration"
                )
        return _EvaluatorSpec(name=name, per_type=per_type)

    def _parse_metric(self) -> MetricConfig:
        name = self._parse_identifier()
        options: dict[str, str] = {}
        self._skip_ws()
        if self._peek() == "(":
            self._advance()
            while True:
                self._skip_ws()
                key = self._parse_identifier()
                self._skip_ws()
                self._expect("=")
                self._skip_ws()
                value = self._parse_value()
                options[key] = value
                self._skip_ws()
                char = self._peek()
                if char == ",":
                    self._advance()
                    continue
                if char == ")":
                    self._advance()
                    break
                raise ValueError(
                    f"Unexpected character '{char}' while parsing metric options"
                )
        return MetricConfig(name=name, options=options)

    def _parse_value(self) -> str:
        start = self._pos
        while self._pos < self._length and self._text[self._pos] not in {
            ",",
            ")",
        }:
            self._pos += 1
        value = self._text[start : self._pos].strip()
        if not value:
            raise ValueError("Metric option value cannot be empty")
        return value

    def _parse_identifier(self) -> str:
        self._skip_ws()
        start = self._pos
        while self._pos < self._length and (
            self._text[self._pos].isalnum() or self._text[self._pos] in {"_"}
        ):
            self._pos += 1
        if start == self._pos:
            raise ValueError("Expected identifier")
        return self._text[start : self._pos]

    def _skip_ws(self) -> None:
        while self._pos < self._length and self._text[self._pos].isspace():
            self._pos += 1

    def _expect(self, char: str) -> None:
        self._skip_ws()
        if self._peek() != char:
            raise ValueError(f"Expected '{char}'")
        self._advance()

    def _peek(self) -> str:
        if self._pos >= self._length:
            return ""
        return self._text[self._pos]

    def _advance(self) -> None:
        self._pos += 1


def _parse_evaluator_config(text: str) -> Sequence[EvaluatorPlan]:
    parser = _ConfigParser(text)
    specs = parser.parse()
    plans: list[EvaluatorPlan] = []
    for spec in specs:
        plans.append(
            EvaluatorPlan(
                name=spec.name,
                per_type=spec.per_type,
            )
        )
    return plans


__all__ = [
    "Manager",
    "Sampler",
    "MetricConfig",
]
