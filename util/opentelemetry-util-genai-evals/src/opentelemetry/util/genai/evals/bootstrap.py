"""Bootstrap utilities for wiring evaluation callbacks into the handler."""

from __future__ import annotations

import logging
from typing import Any

# Import debug module to ensure parent logger is configured when debug is enabled
# This must happen before any logging calls in this module
from opentelemetry.util.genai import debug as _debug  # noqa: F401
from opentelemetry.util.genai.callbacks import CompletionCallback

from .manager import Manager
from .proxy import EvalManagerProxy, is_separate_process_enabled

_LOGGER = logging.getLogger(__name__)


def create_evaluation_manager(
    handler: Any,
    *,
    interval: float | None = None,
    aggregate_results: bool | None = None,
) -> Manager | EvalManagerProxy:
    """Instantiate an evaluation manager bound to the provided handler.

    This function returns either:
    - EvalManagerProxy (separate process mode) when enabled via environment
    - Manager (in-process mode) otherwise

    The separate process mode is preferred when running LLM-as-a-judge
    evaluations (e.g., DeepEval) because it prevents the evaluator's
    OpenAI calls from being instrumented alongside the application's
    telemetry.

    Configuration via environment variables:
    - OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS: "true" or "false" (default: "false")
    - OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: Evaluator configuration

    Args:
        handler: The TelemetryHandler instance.
        interval: Polling interval for the evaluation queue.
        aggregate_results: Whether to aggregate evaluation results.

    Returns:
        Either a Manager or EvalManagerProxy implementing CompletionCallback.
    """
    if is_separate_process_enabled():
        _LOGGER.info("Using separate process evaluation mode")
        try:
            return EvalManagerProxy(
                handler,
                interval=interval,
                aggregate_results=aggregate_results,
            )
        except Exception as exc:
            _LOGGER.warning(
                "Failed to create EvalManagerProxy, falling back to in-process mode: %s",
                exc,
            )
            # Fall through to in-process mode

    return Manager(
        handler,
        interval=interval,
        aggregate_results=aggregate_results,
    )


class EvaluatorCompletionCallback(CompletionCallback):
    """Completion callback façade that lazily instantiates the manager."""

    def __init__(self) -> None:
        self._handler: Any | None = None
        self._manager: Manager | None = None

    # Manager lifecycle -------------------------------------------------
    def bind_handler(self, handler: Any) -> bool:
        """Bind the owning handler and create the manager if applicable."""

        self._handler = handler
        if self._manager is not None:
            return self._manager.has_evaluators
        manager = create_evaluation_manager(handler)
        if not getattr(manager, "has_evaluators", False):
            manager.shutdown()
            self._manager = None
            return False
        self._manager = manager
        return True

    def shutdown(self) -> None:
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def wait_for_all(self, timeout: float | None = None) -> None:
        if self._manager is not None:
            self._manager.wait_for_all(timeout)

    # CompletionCallback -------------------------------------------------
    @property
    def manager(self) -> Manager | None:
        return self._manager

    def on_completion(self, invocation: Any) -> None:
        if self._manager is None:
            if self._handler is None:
                return
            if not self.bind_handler(self._handler):
                return
        self._manager.on_completion(invocation)


def create_completion_callback() -> EvaluatorCompletionCallback:
    """Entry-point exposed factory returning the manager façade."""

    return EvaluatorCompletionCallback()


__all__ = [
    "EvaluatorCompletionCallback",
    "create_completion_callback",
    "create_evaluation_manager",
]
