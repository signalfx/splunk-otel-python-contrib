# Span emitter (moved from generators/span_emitter.py)
from __future__ import annotations

import json  # noqa: F401 (kept for backward compatibility if external code relies on this module re-exporting json)
from dataclasses import fields as dataclass_fields
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from ..attributes import (
    FINISH_REASON_CANCELLED,
    FINISH_REASON_FAILED,
    FINISH_REASON_INTERRUPTED,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_AGENT_TOOLS,
    GEN_AI_AGENT_TYPE,
    GEN_AI_COMMAND,
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_EMBEDDINGS_INPUT_TEXTS,
    GEN_AI_FINISH_REASON,
    GEN_AI_FINISH_REASON_DESCRIPTION,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_ENCODING_FORMATS,
    GEN_AI_RETRIEVAL_DOCUMENTS_RETRIEVED,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_STEP_ASSIGNED_AGENT,
    GEN_AI_STEP_NAME,
    GEN_AI_STEP_OBJECTIVE,
    GEN_AI_STEP_SOURCE,
    GEN_AI_STEP_STATUS,
    GEN_AI_STEP_TYPE,
    GEN_AI_WORKFLOW_DESCRIPTION,
    GEN_AI_WORKFLOW_NAME,
    GEN_AI_WORKFLOW_TYPE,
    SERVER_ADDRESS,
    SERVER_PORT,
)
from ..interfaces import EmitterMeta
from ..span_context import extract_span_context, store_span_context
from ..types import (
    AgentCreation,
    AgentInvocation,
    ContentCapturingMode,
    EmbeddingInvocation,
    Error,
    ErrorClassification,
    LLMInvocation,
    RetrievalInvocation,
    Step,
    ToolCall,
    Workflow,
)
from ..types import (
    GenAI as GenAIType,
)
from .utils import (
    _apply_function_definitions,
    _apply_llm_finish_semconv,
    _extract_system_instructions,
    _serialize_messages,
    filter_semconv_gen_ai_attributes,
)

_SPAN_ALLOWED_SUPPLEMENTAL_KEYS: tuple[str, ...] = (
    "gen_ai.framework",
    "gen_ai.request.id",
    GEN_AI_COMMAND,
    GEN_AI_FINISH_REASON,
    GEN_AI_WORKFLOW_NAME,
)
_SPAN_BLOCKED_SUPPLEMENTAL_KEYS: set[str] = {"request_top_p", "ls_temperature"}


def _sanitize_span_attribute_value(value: Any) -> Optional[Any]:
    """Cast arbitrary invocation attribute values to OTEL-compatible types."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        sanitized_items: list[Any] = []
        for item in value:
            sanitized = _sanitize_span_attribute_value(item)
            if sanitized is None:
                continue
            if isinstance(sanitized, list):
                sanitized_items.append(str(sanitized))
            else:
                sanitized_items.append(sanitized)
        return sanitized_items
    if isinstance(value, dict):
        try:
            return json.dumps(value, default=str)
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _apply_gen_ai_semconv_attributes(
    span: Span,
    attributes: Optional[dict[str, Any]],
) -> None:
    if not attributes:
        return
    for key, value in attributes.items():
        sanitized = _sanitize_span_attribute_value(value)
        if sanitized is None:
            continue
        try:
            span.set_attribute(key, sanitized)
        except Exception:  # pragma: no cover - defensive
            pass


def _apply_tool_semconv_attributes(
    span: Span,
    tool: "ToolCall",
    capture_content: bool = False,
) -> None:
    """Apply semantic convention attributes from ToolCall/MCPToolCall fields.

    Iterates over dataclass fields and applies attributes based on metadata:
    - "semconv": Always applied if value is not None
    - "semconv_content": Only applied if capture_content is True

    This handles both base ToolCall attributes (gen_ai.tool.*) and
    MCPToolCall attributes (mcp.*, network.*) automatically.
    """

    for data_field in dataclass_fields(tool):
        # Check for regular semconv attribute
        semconv_key = data_field.metadata.get("semconv")
        if semconv_key:
            value = getattr(tool, data_field.name)
            if value is not None:
                sanitized = _sanitize_span_attribute_value(value)
                if sanitized is not None:
                    span.set_attribute(semconv_key, sanitized)
            continue

        # Check for content-gated semconv attribute
        semconv_content_key = data_field.metadata.get("semconv_content")
        if semconv_content_key and capture_content:
            value = getattr(tool, data_field.name)
            if value is not None:
                try:
                    if isinstance(value, str):
                        span.set_attribute(semconv_content_key, value)
                    else:
                        span.set_attribute(
                            semconv_content_key,
                            json.dumps(value, default=str),
                        )
                except Exception:
                    pass


def _apply_custom_attributes(
    span: Span,
    attributes: Optional[dict[str, Any]],
) -> None:
    """Apply custom attributes dictionary to a span.

    This helper function applies any custom attributes from the attributes
    dictionary to the span. Used for supplemental attributes that aren't
    defined as dataclass fields with semconv metadata.
    """
    if not attributes:
        return
    for key, value in attributes.items():
        if value is not None:
            sanitized = _sanitize_span_attribute_value(value)
            if sanitized is not None:
                span.set_attribute(key, sanitized)


def _apply_evaluation_attributes(
    span: Span,
    invocation: GenAIType,
) -> None:
    # Check if span is recording before setting attribute
    # This handles ReadableSpan which has already ended, gracefully
    if (
        span is not None
        and hasattr(span, "is_recording")
        and span.is_recording()
    ):
        span.set_attribute(
            "gen_ai.evaluation.sampled", invocation.sample_for_evaluation
        )
        span.set_attribute(
            "gen_ai.evaluation.error",
            str(invocation.evaluation_error),
        )
    elif span is not None and hasattr(span, "_attributes"):
        # Fallback for ReadableSpan: directly mutate _attributes
        try:
            span._attributes["gen_ai.evaluation.sampled"] = str(
                invocation.sample_for_evaluation
            ).lower()
            span._attributes["gen_ai.evaluation.error"] = str(
                invocation.evaluation_error
            )

        except Exception:
            pass


class SpanEmitter(EmitterMeta):
    """Span-focused emitter supporting optional content capture.

    Original implementation migrated from generators/span_emitter.py. Additional telemetry
    (metrics, content events) are handled by separate emitters composed via CompositeEmitter.
    """

    role = "span"
    name = "semconv_span"

    def __init__(
        self, tracer: Optional[Tracer] = None, capture_content: bool = False
    ):
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)
        self._capture_content = capture_content
        self._content_mode = ContentCapturingMode.NO_CONTENT

    def set_capture_content(
        self, value: bool
    ):  # pragma: no cover - trivial mutator
        self._capture_content = value

    def set_content_mode(
        self, mode: ContentCapturingMode
    ) -> None:  # pragma: no cover - trivial mutator
        self._content_mode = mode

    def handles(self, obj: object) -> bool:
        return True

    # ---- helpers ---------------------------------------------------------
    def _apply_start_attrs(self, invocation: GenAIType):
        span = getattr(invocation, "span", None)
        if span is None:
            return
        semconv_attrs = dict(invocation.semantic_convention_attributes())
        if isinstance(invocation, ToolCall):
            enum_val = getattr(
                GenAI.GenAiOperationNameValues, "EXECUTE_TOOL", None
            )
            semconv_attrs[GenAI.GEN_AI_OPERATION_NAME] = (
                enum_val.value if enum_val else "execute_tool"
            )
        elif isinstance(invocation, EmbeddingInvocation):
            semconv_attrs.setdefault(
                GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
            )
        elif isinstance(invocation, LLMInvocation):
            semconv_attrs.setdefault(
                GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
            )
        _apply_gen_ai_semconv_attributes(span, semconv_attrs)
        supplemental = getattr(invocation, "attributes", None)
        if supplemental:
            semconv_subset = filter_semconv_gen_ai_attributes(
                supplemental, extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS
            )
            if semconv_subset:
                _apply_gen_ai_semconv_attributes(span, semconv_subset)
            for key, value in supplemental.items():
                if key in (semconv_subset or {}):
                    continue
                if key in _SPAN_BLOCKED_SUPPLEMENTAL_KEYS:
                    continue
                if (
                    not key.startswith("custom_")
                    and key not in _SPAN_ALLOWED_SUPPLEMENTAL_KEYS
                ):
                    continue
                if key in span.attributes:  # type: ignore[attr-defined]
                    continue
                sanitized = _sanitize_span_attribute_value(value)
                if sanitized is None:
                    continue
                try:
                    span.set_attribute(key, sanitized)
                except Exception:  # pragma: no cover - defensive
                    pass
        provider = getattr(invocation, "provider", None)
        if provider:
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
        framework = getattr(invocation, "framework", None)
        if framework:
            span.set_attribute("gen_ai.framework", framework)
        server_address = getattr(invocation, "server_address", None)
        if server_address:
            span.set_attribute(SERVER_ADDRESS, server_address)
        server_port = getattr(invocation, "server_port", None)
        if server_port:
            span.set_attribute(SERVER_PORT, server_port)
        # framework (named field)
        if isinstance(invocation, LLMInvocation) and invocation.framework:
            span.set_attribute("gen_ai.framework", invocation.framework)
        # function definitions (semantic conv derived from structured list)
        if isinstance(invocation, LLMInvocation):
            _apply_function_definitions(span, invocation.request_functions)
        # Agent context (already covered by semconv metadata on base fields)

    def _apply_finish_attrs(
        self, invocation: LLMInvocation | EmbeddingInvocation
    ):
        span = getattr(invocation, "span", None)
        if span is None:
            return

        # Capture input messages and system instructions if enabled
        if (
            self._capture_content
            and isinstance(invocation, LLMInvocation)
            and invocation.input_messages
        ):
            # Extract and set system instructions separately
            system_instructions = _extract_system_instructions(
                invocation.input_messages
            )
            if system_instructions is not None:
                span.set_attribute(
                    GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions
                )

            # Serialize input messages (excluding system messages)
            serialized_in = _serialize_messages(
                invocation.input_messages, exclude_system=True
            )
            if serialized_in is not None:
                span.set_attribute(GEN_AI_INPUT_MESSAGES, serialized_in)

        # Finish-time semconv attributes (response + usage tokens + functions)
        if isinstance(invocation, LLMInvocation):
            _apply_llm_finish_semconv(span, invocation)
        _apply_gen_ai_semconv_attributes(
            span, invocation.semantic_convention_attributes()
        )
        extra_attrs = filter_semconv_gen_ai_attributes(
            getattr(invocation, "attributes", None),
            extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS,
        )
        if extra_attrs:
            _apply_gen_ai_semconv_attributes(span, extra_attrs)

        # Capture output messages if enabled
        if (
            self._capture_content
            and isinstance(invocation, LLMInvocation)
            and invocation.output_messages
        ):
            serialized = _serialize_messages(invocation.output_messages)
            if serialized is not None:
                span.set_attribute(GEN_AI_OUTPUT_MESSAGES, serialized)

    def _add_span_to_invocation(
        self,
        invocation: GenAIType,
        span: Span,
    ) -> None:
        invocation.span = span  # type: ignore[assignment]
        store_span_context(invocation, extract_span_context(span))

    # ---- lifecycle -------------------------------------------------------
    def on_start(
        self, invocation: LLMInvocation | EmbeddingInvocation
    ) -> None:  # type: ignore[override]
        # Handle new agentic types
        if isinstance(invocation, Workflow):
            self._start_workflow(invocation)
        elif isinstance(invocation, (AgentCreation, AgentInvocation)):
            self._start_agent(invocation)
        elif isinstance(invocation, Step):
            self._start_step(invocation)
        # Handle existing types
        elif isinstance(invocation, ToolCall):
            self._start_tool_call(invocation)
        elif isinstance(invocation, EmbeddingInvocation):
            self._start_embedding(invocation)
        elif isinstance(invocation, RetrievalInvocation):
            self._start_retrieval(invocation)
        else:
            # Use operation field for span name (defaults to "chat")
            operation = getattr(invocation, "operation", "chat")
            model_name = invocation.request_model
            span_name = f"{operation} {model_name}"
            parent_span = getattr(invocation, "parent_span", None)
            parent_ctx = (
                trace.set_span_in_context(parent_span)
                if parent_span is not None
                else None
            )
            span = self._tracer.start_span(
                span_name,
                kind=SpanKind.CLIENT,
                context=parent_ctx,
            )
            self._add_span_to_invocation(invocation, span)
            self._apply_start_attrs(invocation)

    def on_end(self, invocation: LLMInvocation | EmbeddingInvocation) -> None:
        _apply_evaluation_attributes(invocation.span, invocation)  # type: ignore[override]
        if isinstance(invocation, Workflow):
            self._finish_workflow(invocation)
        elif isinstance(invocation, (AgentCreation, AgentInvocation)):
            self._finish_agent(invocation)
        elif isinstance(invocation, Step):
            self._finish_step(invocation)
        elif isinstance(invocation, ToolCall):
            self._finish_tool_call(invocation)
        elif isinstance(invocation, EmbeddingInvocation):
            self._finish_embedding(invocation)
        elif isinstance(invocation, RetrievalInvocation):
            self._finish_retrieval(invocation)
        else:
            span = getattr(invocation, "span", None)
            if span is None:
                return
            # Check if span is still recording (not already ended)
            # This allows reusing on_end with ReadableSpan from translators
            is_recording = (
                hasattr(span, "is_recording") and span.is_recording()
            )
            if is_recording:
                self._apply_finish_attrs(invocation)
                span.end()

    def _apply_error_status(self, span: Span, error: Error) -> None:
        """Apply span status based on error classification.

        For INTERRUPT and CANCELLATION, we intentionally do NOT call
        set_status() — leaving the span in its default UNSET state.
        UNSET means "no error occurred" without the stronger assertion
        of OK ("validated as successful"). This matches the OTel spec:
        interrupts are not errors, but they are not successful completions.
        """
        classification = error.classification
        if classification == ErrorClassification.INTERRUPT:
            # Leave status UNSET (default) — not an error, not a success
            if span.is_recording():
                span.set_attribute(
                    GEN_AI_FINISH_REASON, FINISH_REASON_INTERRUPTED
                )
                if error.message:
                    span.set_attribute(
                        GEN_AI_FINISH_REASON_DESCRIPTION, error.message
                    )
        elif classification == ErrorClassification.CANCELLATION:
            # Leave status UNSET (default) — not an error
            if span.is_recording():
                span.set_attribute(
                    GEN_AI_FINISH_REASON, FINISH_REASON_CANCELLED
                )
                if error.message:
                    span.set_attribute(
                        GEN_AI_FINISH_REASON_DESCRIPTION, error.message
                    )
        else:
            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )
                span.set_attribute(GEN_AI_FINISH_REASON, FINISH_REASON_FAILED)
                if error.message:
                    span.set_attribute(
                        GEN_AI_FINISH_REASON_DESCRIPTION, error.message
                    )

    def on_error(
        self, error: Error, invocation: LLMInvocation | EmbeddingInvocation
    ) -> None:  # type: ignore[override]
        if isinstance(invocation, Workflow):
            self._error_workflow(error, invocation)
        elif isinstance(invocation, (AgentCreation, AgentInvocation)):
            self._error_agent(error, invocation)
        elif isinstance(invocation, Step):
            self._error_step(error, invocation)
        elif isinstance(invocation, ToolCall):
            self._error_tool_call(error, invocation)
        elif isinstance(invocation, EmbeddingInvocation):
            self._error_embedding(error, invocation)
        elif isinstance(invocation, RetrievalInvocation):
            self._error_retrieval(error, invocation)
        else:
            span = getattr(invocation, "span", None)
            if span is None:
                return
            self._apply_error_status(span, error)
            self._apply_finish_attrs(invocation)
            span.end()

    # ---- Workflow lifecycle ----------------------------------------------
    def _start_workflow(self, workflow: Workflow) -> None:
        """Start a workflow span."""
        span_name = f"workflow {workflow.name}"
        parent_span = getattr(workflow, "parent_span", None)
        parent_ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        self._add_span_to_invocation(workflow, span)

        # Set workflow attributes
        # TODO: Align to enum when semconvs is updated.
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "invoke_workflow")
        span.set_attribute(GEN_AI_WORKFLOW_NAME, workflow.name)
        if workflow.workflow_type:
            span.set_attribute(GEN_AI_WORKFLOW_TYPE, workflow.workflow_type)
        if workflow.description:
            span.set_attribute(
                GEN_AI_WORKFLOW_DESCRIPTION, workflow.description
            )
        if workflow.framework:
            span.set_attribute("gen_ai.framework", workflow.framework)
        if self._capture_content and workflow.input_messages:
            serialized = _serialize_messages(workflow.input_messages)
            if serialized is not None:
                span.set_attribute("gen_ai.input.messages", serialized)
        _apply_gen_ai_semconv_attributes(
            span, workflow.semantic_convention_attributes()
        )
        # Apply supplemental attributes (e.g., gen_ai.command)
        supplemental = getattr(workflow, "attributes", None)
        if supplemental:
            semconv_subset = filter_semconv_gen_ai_attributes(
                supplemental, extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS
            )
            if semconv_subset:
                _apply_gen_ai_semconv_attributes(span, semconv_subset)

    def _finish_workflow(self, workflow: Workflow) -> None:
        """Finish a workflow span."""
        span = workflow.span
        if span is None:
            return
        # Set output if capture_content enabled
        if self._capture_content and workflow.output_messages:
            serialized = _serialize_messages(workflow.output_messages)
            if serialized is not None:
                span.set_attribute("gen_ai.output.messages", serialized)
        _apply_gen_ai_semconv_attributes(
            span, workflow.semantic_convention_attributes()
        )
        # Re-apply supplemental attributes so values set after start
        # (e.g. interrupt bubbling to root) are captured on the span.
        supplemental = getattr(workflow, "attributes", None)
        if supplemental:
            semconv_subset = filter_semconv_gen_ai_attributes(
                supplemental, extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS
            )
            if semconv_subset:
                _apply_gen_ai_semconv_attributes(span, semconv_subset)
        span.end()

    def _error_workflow(self, error: Error, workflow: Workflow) -> None:
        """Fail a workflow span with error status."""
        span = workflow.span
        if span is None:
            return
        self._apply_error_status(span, error)
        _apply_gen_ai_semconv_attributes(
            span, workflow.semantic_convention_attributes()
        )
        span.end()

    # ---- Agent lifecycle -------------------------------------------------
    def _start_agent(self, agent: AgentCreation | AgentInvocation) -> None:
        """Start an agent span (create or invoke)."""
        # Span name per semantic conventions
        if agent.operation == "create_agent":
            span_name = f"create_agent {agent.name}"
        else:
            span_name = f"invoke_agent {agent.name}"

        parent_span = getattr(agent, "parent_span", None)
        parent_ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        self._add_span_to_invocation(agent, span)

        # Required attributes per semantic conventions
        # Set operation name based on agent operation (create or invoke)
        semconv_attrs = dict(agent.semantic_convention_attributes())
        semconv_attrs.setdefault(GEN_AI_AGENT_NAME, agent.name)
        semconv_attrs.setdefault(
            GEN_AI_AGENT_ID,
            f"{agent.span_id:016x}"
            if agent.span_id is not None
            else str(id(agent)),
        )
        _apply_gen_ai_semconv_attributes(span, semconv_attrs)

        # Optional attributes
        if agent.agent_type:
            span.set_attribute(GEN_AI_AGENT_TYPE, agent.agent_type)
        if agent.framework:
            span.set_attribute("gen_ai.framework", agent.framework)
        if agent.tools:
            span.set_attribute(GEN_AI_AGENT_TOOLS, agent.tools)
        if agent.system_instructions and self._capture_content:
            system_parts = [
                {"type": "text", "content": agent.system_instructions}
            ]
            span.set_attribute(
                GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, json.dumps(system_parts)
            )
        if self._capture_content:
            input_messages = getattr(agent, "input_messages", None)
            if input_messages:
                serialized = _serialize_messages(input_messages)
                if serialized is not None:
                    span.set_attribute("gen_ai.input.messages", serialized)
        _apply_gen_ai_semconv_attributes(
            span, agent.semantic_convention_attributes()
        )

        # Apply supplemental attributes (e.g., workflow_name)
        supplemental = getattr(agent, "attributes", None)
        if supplemental:
            semconv_subset = filter_semconv_gen_ai_attributes(
                supplemental, extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS
            )
            if semconv_subset:
                _apply_gen_ai_semconv_attributes(span, semconv_subset)

    def _finish_agent(self, agent: AgentCreation | AgentInvocation) -> None:
        """Finish an agent span."""
        span = agent.span
        if span is None:
            return
        # Set output if capture_content enabled
        if self._capture_content and isinstance(agent, AgentInvocation):
            if agent.output_messages:
                serialized = _serialize_messages(agent.output_messages)
                if serialized is not None:
                    span.set_attribute("gen_ai.output.messages", serialized)
        _apply_gen_ai_semconv_attributes(
            span, agent.semantic_convention_attributes()
        )
        # Re-apply supplemental attributes so values set after start
        # (e.g. interrupt bubbling to root) are captured on the span.
        supplemental = getattr(agent, "attributes", None)
        if supplemental:
            semconv_subset = filter_semconv_gen_ai_attributes(
                supplemental, extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS
            )
            if semconv_subset:
                _apply_gen_ai_semconv_attributes(span, semconv_subset)
        span.end()

    def _error_agent(
        self, error: Error, agent: AgentCreation | AgentInvocation
    ) -> None:
        """Fail an agent span with error status."""
        span = agent.span
        if span is None:
            return
        self._apply_error_status(span, error)
        _apply_gen_ai_semconv_attributes(
            span, agent.semantic_convention_attributes()
        )
        span.end()

    # ---- Step lifecycle --------------------------------------------------
    def _start_step(self, step: Step) -> None:
        """Start a step span."""
        span_name = f"step {step.name}"
        parent_span = getattr(step, "parent_span", None)
        parent_ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        self._add_span_to_invocation(step, span)

        # Set step attributes
        span.set_attribute(GEN_AI_STEP_NAME, step.name)
        if step.step_type:
            span.set_attribute(GEN_AI_STEP_TYPE, step.step_type)
        if step.objective:
            span.set_attribute(GEN_AI_STEP_OBJECTIVE, step.objective)
        if step.source:
            span.set_attribute(GEN_AI_STEP_SOURCE, step.source)
        if step.assigned_agent:
            span.set_attribute(GEN_AI_STEP_ASSIGNED_AGENT, step.assigned_agent)
        if step.status:
            span.set_attribute(GEN_AI_STEP_STATUS, step.status)
        _apply_gen_ai_semconv_attributes(
            span, step.semantic_convention_attributes()
        )

    def _finish_step(self, step: Step) -> None:
        """Finish a step span."""
        span = step.span
        if span is None:
            return
        # Update status if changed
        if step.status:
            span.set_attribute(GEN_AI_STEP_STATUS, step.status)
        _apply_gen_ai_semconv_attributes(
            span, step.semantic_convention_attributes()
        )
        span.end()

    def _error_step(self, error: Error, step: Step) -> None:
        """Fail a step span with error status."""
        span = step.span
        if span is None:
            return
        self._apply_error_status(span, error)
        # Update step status based on error classification
        classification = error.classification
        if classification == ErrorClassification.INTERRUPT:
            span.set_attribute(GEN_AI_STEP_STATUS, FINISH_REASON_INTERRUPTED)
        elif classification == ErrorClassification.CANCELLATION:
            span.set_attribute(GEN_AI_STEP_STATUS, FINISH_REASON_CANCELLED)
        else:
            span.set_attribute(GEN_AI_STEP_STATUS, FINISH_REASON_FAILED)
        _apply_gen_ai_semconv_attributes(
            span, step.semantic_convention_attributes()
        )
        span.end()

    # ---- Tool Call lifecycle ---------------------------------------------
    def _start_tool_call(self, tool: ToolCall) -> None:
        """Start a tool call span per execute_tool semantic conventions.

        Span name: execute_tool {gen_ai.tool.name}
        Span kind: INTERNAL
        See: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md#execute-tool-span
        """
        # Span name per semconv: "execute_tool {gen_ai.tool.name}"
        span_name = f"execute_tool {tool.name}"
        parent_span = getattr(tool, "parent_span", None)
        parent_ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )
        self._add_span_to_invocation(tool, span)

        # Required: gen_ai.operation.name = "execute_tool"
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME,
            GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
        )

        # Apply all semconv attributes from dataclass fields
        # This handles gen_ai.tool.*, mcp.*, network.*, error.type
        _apply_tool_semconv_attributes(span, tool, self._capture_content)

        # Apply any supplemental custom attributes
        _apply_custom_attributes(span, getattr(tool, "attributes", None))

    def _finish_tool_call(self, tool: ToolCall) -> None:
        """Finish a tool call span."""
        span = tool.span
        if span is None:
            return
        # Check if span is still recording
        is_recording = hasattr(span, "is_recording") and span.is_recording()
        if is_recording:
            _apply_tool_semconv_attributes(span, tool, self._capture_content)
            _apply_custom_attributes(span, getattr(tool, "attributes", None))
            span.end()

    def _error_tool_call(self, error: Error, tool: ToolCall) -> None:
        """Fail a tool call span with error status."""
        span = tool.span
        if span is None:
            return
        self._apply_error_status(span, error)
        if span.is_recording():
            _apply_tool_semconv_attributes(span, tool, self._capture_content)
            _apply_custom_attributes(span, getattr(tool, "attributes", None))
        span.end()

    # ---- Embedding lifecycle ---------------------------------------------
    def _start_embedding(self, embedding: EmbeddingInvocation) -> None:
        """Start an embedding span."""
        span_name = f"{embedding.operation_name} {embedding.request_model}"
        parent_span = getattr(embedding, "parent_span", None)
        parent_ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        self._add_span_to_invocation(embedding, span)
        self._apply_start_attrs(embedding)

        # Set embedding-specific start attributes
        if embedding.encoding_formats:
            span.set_attribute(
                GEN_AI_REQUEST_ENCODING_FORMATS, embedding.encoding_formats
            )
        if self._capture_content and embedding.input_texts:
            # Capture input texts as array attribute
            span.set_attribute(
                GEN_AI_EMBEDDINGS_INPUT_TEXTS, embedding.input_texts
            )

    def _finish_embedding(self, embedding: EmbeddingInvocation) -> None:
        """Finish an embedding span."""
        span = embedding.span
        if span is None:
            return
        # Apply finish-time semantic conventions
        if embedding.dimension_count:
            span.set_attribute(
                GEN_AI_EMBEDDINGS_DIMENSION_COUNT, embedding.dimension_count
            )
        if embedding.input_tokens is not None:
            span.set_attribute(
                GenAI.GEN_AI_USAGE_INPUT_TOKENS, embedding.input_tokens
            )
        span.end()

    def _error_embedding(
        self, error: Error, embedding: EmbeddingInvocation
    ) -> None:
        """Fail an embedding span with error status."""
        span = embedding.span
        if span is None:
            return
        self._apply_error_status(span, error)
        # Set error type from invocation if available
        if embedding.error_type:
            span.set_attribute(
                ErrorAttributes.ERROR_TYPE, embedding.error_type
            )
        span.end()

    def _start_retrieval(self, retrieval: RetrievalInvocation) -> None:
        """Start a retrieval span."""
        span_name = f"{retrieval.operation_name}"
        if retrieval.provider:
            span_name = f"{retrieval.operation_name} {retrieval.provider}"
        parent_span = getattr(retrieval, "parent_span", None)
        parent_ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        self._add_span_to_invocation(retrieval, span)
        self._apply_start_attrs(retrieval)

        # Set retrieval-specific start attributes
        if retrieval.server_address:
            span.set_attribute(SERVER_ADDRESS, retrieval.server_address)
        if retrieval.server_port:
            span.set_attribute(SERVER_PORT, retrieval.server_port)
        if retrieval.top_k is not None:
            span.set_attribute(GEN_AI_RETRIEVAL_TOP_K, retrieval.top_k)
        if self._capture_content and retrieval.query:
            span.set_attribute(GEN_AI_RETRIEVAL_QUERY_TEXT, retrieval.query)

    def _finish_retrieval(self, retrieval: RetrievalInvocation) -> None:
        """Finish a retrieval span."""
        span = retrieval.span
        if span is None:
            return
        # Apply finish-time semantic conventions
        if retrieval.documents_retrieved is not None:
            span.set_attribute(
                GEN_AI_RETRIEVAL_DOCUMENTS_RETRIEVED,
                retrieval.documents_retrieved,
            )
        span.end()

    def _error_retrieval(
        self, error: Error, retrieval: RetrievalInvocation
    ) -> None:
        """Fail a retrieval span with error status."""
        span = retrieval.span
        if span is None:
            return
        self._apply_error_status(span, error)
        # Set error type from invocation if available
        if retrieval.error_type:
            span.set_attribute(
                ErrorAttributes.ERROR_TYPE, retrieval.error_type
            )
        span.end()
