from __future__ import annotations

import logging
from typing import Any, Optional

from opentelemetry._logs import Logger, get_logger

from ..interfaces import EmitterMeta
from ..types import (
    AgentCreation,
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    LLMInvocation,
    Step,
    Workflow,
)
from .utils import (
    _agent_to_log_record,
    _embedding_to_log_record,
    _llm_invocation_to_log_record,
    _step_to_log_record,
    _workflow_to_log_record,
)


class ContentEventsEmitter(EmitterMeta):
    """Emits input/output content as events (log records) instead of span attributes.

    Supported: LLMInvocation only.

    Exclusions:
      * EmbeddingInvocation – embeddings are vector lookups; content events intentionally omitted to reduce noise & cost.
      * ToolCall – tool calls typically reference external functions/APIs; their arguments are already span attributes and
        are not duplicated as content events (future structured tool audit events may be added separately).

    This explicit exclusion avoids surprising cardinality growth and keeps event volume proportional to user/chat messages.
    """

    role = "content_event"
    name = "semconv_content_events"

    def __init__(
        self, logger: Optional[Logger] = None, capture_content: bool = False
    ):
        self._logger: Logger = logger or get_logger(__name__)
        self._capture_content = capture_content
        self._py_logger = logging.getLogger(f"{__name__}.ContentEventsEmitter")
        if self._py_logger.isEnabledFor(logging.DEBUG):
            self._py_logger.debug(
                "Initialized ContentEventsEmitter capture_content=%s logger=%s",
                capture_content,
                type(self._logger).__name__,
            )

    def on_start(self, obj: Any) -> None:
        # LLM events are emitted in finish() when we have both input and output
        return None

    def on_end(self, obj: Any) -> None:
        if not self._capture_content:
            if self._py_logger.isEnabledFor(logging.DEBUG):
                self._py_logger.debug(
                    "Skipping content emission (capture_content disabled) obj_type=%s",
                    type(obj).__name__,
                )
            return
        # Emit workflow event (includes initial input + final output messages)
        if isinstance(obj, Workflow):
            self._emit_workflow_event(obj)
            return
        # Emit agent creation/invocation event (input/output messages where available)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            self._emit_agent_event(obj)
            return
        # Optional: step and embedding events (currently excluded from request scope)
        # Uncomment if needed later:
        # if isinstance(obj, Step):
        #     self._emit_step_event(obj); return
        # if isinstance(obj, EmbeddingInvocation):
        #     self._emit_embedding_event(obj); return
        if isinstance(obj, LLMInvocation):
            # Emit a single event for the entire LLM invocation
            try:
                record = _llm_invocation_to_log_record(
                    obj,
                    self._capture_content,
                )
                if record and self._logger:
                    if self._py_logger.isEnabledFor(logging.DEBUG):
                        self._py_logger.debug(
                            "Emitting LLM content event trace_id=%s span_id=%s",
                            getattr(obj, "trace_id", None),
                            getattr(obj, "span_id", None),
                        )
                    self._logger.emit(record)
                elif self._py_logger.isEnabledFor(logging.DEBUG):
                    self._py_logger.debug(
                        "No log record generated for LLM invocation (capture_content=%s)",
                        self._capture_content,
                    )
            except (TypeError, ValueError, AttributeError) as e:
                logging.getLogger(__name__).warning(
                    "Failed to emit LLM invocation event: %s", e, exc_info=True
                )

    def on_error(self, error: Error, obj: Any) -> None:
        return None

    def handles(self, obj: Any) -> bool:
        return isinstance(
            obj,
            (LLMInvocation, Workflow, AgentCreation, AgentInvocation, Step),
        )

    # Helper methods for new agentic types
    def _emit_workflow_event(self, workflow: Workflow) -> None:
        """Emit an event for a workflow."""
        try:
            record = _workflow_to_log_record(workflow, self._capture_content)
            if record and self._logger:
                self._logger.emit(record)
        except (TypeError, ValueError, AttributeError):
            return None

    def _emit_agent_event(
        self, agent: AgentCreation | AgentInvocation
    ) -> None:
        """Emit an event for an agent operation."""
        try:
            record = _agent_to_log_record(agent, self._capture_content)
            if record and self._logger:
                self._logger.emit(record)
        except (TypeError, ValueError, AttributeError):
            return None

    def _emit_step_event(self, step: Step) -> None:
        """Emit an event for a step."""
        try:
            record = _step_to_log_record(step, self._capture_content)
            if record and self._logger:
                self._logger.emit(record)
        except (TypeError, ValueError, AttributeError):
            return None

    def _emit_embedding_event(self, embedding: EmbeddingInvocation) -> None:
        """Emit an event for an embedding operation."""
        try:
            record = _embedding_to_log_record(embedding, self._capture_content)
            if record and self._logger:
                self._logger.emit(record)
        except (TypeError, ValueError, AttributeError):
            return None
