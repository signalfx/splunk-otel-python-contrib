"""
GenAI Semantic Convention Trace Processor

This module implements a custom trace processor that enriches spans with
OpenTelemetry GenAI semantic conventions attributes following the
OpenInference processor pattern. It adds standardized attributes for
generative AI operations using iterator-based attribute extraction.

References:
- OpenTelemetry GenAI Semantic Conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- OpenInference Pattern: https://github.com/Arize-ai/openinference
"""

# pylint: disable=too-many-lines,invalid-name,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,too-many-nested-blocks,too-many-arguments,too-many-instance-attributes,broad-exception-caught,no-self-use,consider-iterating-dictionary,unused-variable,unnecessary-pass
# ruff: noqa: I001

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from opentelemetry.trace import Status, StatusCode
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.util.genai.attributes import GEN_AI_WORKFLOW_NAME
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
)
from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    Workflow,
)
from opentelemetry.util.genai.utils import gen_ai_json_dumps
from opentelemetry.util.types import AttributeValue
from opentelemetry.metrics import get_meter
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.util.genai.instruments import Instruments


# Invocation State Management


@dataclass
class _InvocationState:
    """Tracks invocation state and parent-child relationships."""

    invocation: Optional[
        Union[
            AgentCreation, AgentInvocation, LLMInvocation, ToolCall, Workflow
        ]
    ] = None
    parent_span_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    # Accumulated content during span lifetime
    input_messages: List[Any] = field(default_factory=list)
    output_messages: List[Any] = field(default_factory=list)
    system_instructions: List[Any] = field(default_factory=list)
    request_model: Optional[str] = None


try:
    from agents.tracing import Span, Trace, TracingProcessor
    from agents.tracing.span_data import (
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        GuardrailSpanData,
        HandoffSpanData,
        ResponseSpanData,
        SpeechSpanData,
        TranscriptionSpanData,
    )
except ModuleNotFoundError:  # pragma: no cover - test stubs
    tracing_module = importlib.import_module("agents.tracing")
    Span = getattr(tracing_module, "Span")
    Trace = getattr(tracing_module, "Trace")
    TracingProcessor = getattr(tracing_module, "TracingProcessor")
    AgentSpanData = getattr(tracing_module, "AgentSpanData", Any)  # type: ignore[assignment]
    FunctionSpanData = getattr(tracing_module, "FunctionSpanData", Any)  # type: ignore[assignment]
    GenerationSpanData = getattr(tracing_module, "GenerationSpanData", Any)  # type: ignore[assignment]
    GuardrailSpanData = getattr(tracing_module, "GuardrailSpanData", Any)  # type: ignore[assignment]
    HandoffSpanData = getattr(tracing_module, "HandoffSpanData", Any)  # type: ignore[assignment]
    ResponseSpanData = getattr(tracing_module, "ResponseSpanData", Any)  # type: ignore[assignment]
    SpeechSpanData = getattr(tracing_module, "SpeechSpanData", Any)  # type: ignore[assignment]
    TranscriptionSpanData = getattr(
        tracing_module, "TranscriptionSpanData", Any
    )  # type: ignore[assignment]

# Import all semantic convention constants
# ---- GenAI semantic convention helpers (embedded from constants.py) ----


def _enum_values(enum_cls) -> dict[str, str]:
    """Return mapping of enum member name to value."""
    return {member.name: member.value for member in enum_cls}


_PROVIDER_VALUES = _enum_values(GenAIAttributes.GenAiProviderNameValues)


class GenAIProvider:
    OPENAI = _PROVIDER_VALUES["OPENAI"]
    GCP_GEN_AI = _PROVIDER_VALUES["GCP_GEN_AI"]
    GCP_VERTEX_AI = _PROVIDER_VALUES["GCP_VERTEX_AI"]
    GCP_GEMINI = _PROVIDER_VALUES["GCP_GEMINI"]
    ANTHROPIC = _PROVIDER_VALUES["ANTHROPIC"]
    COHERE = _PROVIDER_VALUES["COHERE"]
    AZURE_AI_INFERENCE = _PROVIDER_VALUES["AZURE_AI_INFERENCE"]
    AZURE_AI_OPENAI = _PROVIDER_VALUES["AZURE_AI_OPENAI"]
    IBM_WATSONX_AI = _PROVIDER_VALUES["IBM_WATSONX_AI"]
    AWS_BEDROCK = _PROVIDER_VALUES["AWS_BEDROCK"]
    PERPLEXITY = _PROVIDER_VALUES["PERPLEXITY"]
    X_AI = _PROVIDER_VALUES["X_AI"]
    DEEPSEEK = _PROVIDER_VALUES["DEEPSEEK"]
    GROQ = _PROVIDER_VALUES["GROQ"]
    MISTRAL_AI = _PROVIDER_VALUES["MISTRAL_AI"]

    ALL = set(_PROVIDER_VALUES.values())


_OPERATION_VALUES = _enum_values(GenAIAttributes.GenAiOperationNameValues)


class GenAIOperationName:
    CHAT = _OPERATION_VALUES["CHAT"]
    GENERATE_CONTENT = _OPERATION_VALUES["GENERATE_CONTENT"]
    TEXT_COMPLETION = _OPERATION_VALUES["TEXT_COMPLETION"]
    EMBEDDINGS = _OPERATION_VALUES["EMBEDDINGS"]
    CREATE_AGENT = _OPERATION_VALUES["CREATE_AGENT"]
    INVOKE_AGENT = _OPERATION_VALUES["INVOKE_AGENT"]
    EXECUTE_TOOL = _OPERATION_VALUES["EXECUTE_TOOL"]
    # Operations below are not yet covered by the spec but remain for backwards compatibility
    TRANSCRIPTION = "transcription"
    SPEECH = "speech_generation"
    GUARDRAIL = "guardrail_check"
    HANDOFF = "agent_handoff"
    RESPONSE = "response"  # internal aggregator in current processor

    CLASS_FALLBACK = {
        "generationspan": CHAT,
        "responsespan": RESPONSE,
        "functionspan": EXECUTE_TOOL,
        "agentspan": INVOKE_AGENT,
    }


_OUTPUT_VALUES = _enum_values(GenAIAttributes.GenAiOutputTypeValues)


class GenAIOutputType:
    TEXT = _OUTPUT_VALUES["TEXT"]
    JSON = _OUTPUT_VALUES["JSON"]
    IMAGE = _OUTPUT_VALUES["IMAGE"]
    SPEECH = _OUTPUT_VALUES["SPEECH"]


class GenAIToolType:
    FUNCTION = "function"
    EXTENSION = "extension"
    DATASTORE = "datastore"

    ALL = {FUNCTION, EXTENSION, DATASTORE}


class GenAIEvaluationAttributes:
    NAME = "gen_ai.evaluation.name"
    SCORE_VALUE = "gen_ai.evaluation.score.value"
    SCORE_LABEL = "gen_ai.evaluation.score.label"
    EXPLANATION = "gen_ai.evaluation.explanation"


def _attr(name: str, fallback: str) -> str:
    return getattr(GenAIAttributes, name, fallback)


GEN_AI_PROVIDER_NAME = _attr("GEN_AI_PROVIDER_NAME", "gen_ai.provider.name")
GEN_AI_OPERATION_NAME = _attr("GEN_AI_OPERATION_NAME", "gen_ai.operation.name")
GEN_AI_REQUEST_MODEL = _attr("GEN_AI_REQUEST_MODEL", "gen_ai.request.model")
GEN_AI_REQUEST_MAX_TOKENS = _attr(
    "GEN_AI_REQUEST_MAX_TOKENS", "gen_ai.request.max_tokens"
)
GEN_AI_REQUEST_TEMPERATURE = _attr(
    "GEN_AI_REQUEST_TEMPERATURE", "gen_ai.request.temperature"
)
GEN_AI_REQUEST_TOP_P = _attr("GEN_AI_REQUEST_TOP_P", "gen_ai.request.top_p")
GEN_AI_REQUEST_TOP_K = _attr("GEN_AI_REQUEST_TOP_K", "gen_ai.request.top_k")
GEN_AI_REQUEST_FREQUENCY_PENALTY = _attr(
    "GEN_AI_REQUEST_FREQUENCY_PENALTY", "gen_ai.request.frequency_penalty"
)
GEN_AI_REQUEST_PRESENCE_PENALTY = _attr(
    "GEN_AI_REQUEST_PRESENCE_PENALTY", "gen_ai.request.presence_penalty"
)
GEN_AI_REQUEST_CHOICE_COUNT = _attr(
    "GEN_AI_REQUEST_CHOICE_COUNT", "gen_ai.request.choice.count"
)
GEN_AI_REQUEST_STOP_SEQUENCES = _attr(
    "GEN_AI_REQUEST_STOP_SEQUENCES", "gen_ai.request.stop_sequences"
)
GEN_AI_REQUEST_ENCODING_FORMATS = _attr(
    "GEN_AI_REQUEST_ENCODING_FORMATS", "gen_ai.request.encoding_formats"
)
GEN_AI_REQUEST_SEED = _attr("GEN_AI_REQUEST_SEED", "gen_ai.request.seed")
GEN_AI_RESPONSE_ID = _attr("GEN_AI_RESPONSE_ID", "gen_ai.response.id")
GEN_AI_RESPONSE_MODEL = _attr("GEN_AI_RESPONSE_MODEL", "gen_ai.response.model")
GEN_AI_RESPONSE_FINISH_REASONS = _attr(
    "GEN_AI_RESPONSE_FINISH_REASONS", "gen_ai.response.finish_reasons"
)
GEN_AI_USAGE_INPUT_TOKENS = _attr(
    "GEN_AI_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"
)
GEN_AI_USAGE_OUTPUT_TOKENS = _attr(
    "GEN_AI_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"
)
GEN_AI_CONVERSATION_ID = _attr(
    "GEN_AI_CONVERSATION_ID", "gen_ai.conversation.id"
)
GEN_AI_AGENT_ID = _attr("GEN_AI_AGENT_ID", "gen_ai.agent.id")
GEN_AI_AGENT_NAME = _attr("GEN_AI_AGENT_NAME", "gen_ai.agent.name")
GEN_AI_AGENT_DESCRIPTION = _attr(
    "GEN_AI_AGENT_DESCRIPTION", "gen_ai.agent.description"
)
GEN_AI_TOOL_NAME = _attr("GEN_AI_TOOL_NAME", "gen_ai.tool.name")
GEN_AI_TOOL_TYPE = _attr("GEN_AI_TOOL_TYPE", "gen_ai.tool.type")
GEN_AI_TOOL_CALL_ID = _attr("GEN_AI_TOOL_CALL_ID", "gen_ai.tool.call.id")
GEN_AI_TOOL_DESCRIPTION = _attr(
    "GEN_AI_TOOL_DESCRIPTION", "gen_ai.tool.description"
)
GEN_AI_OUTPUT_TYPE = _attr("GEN_AI_OUTPUT_TYPE", "gen_ai.output.type")
GEN_AI_SYSTEM_INSTRUCTIONS = _attr(
    "GEN_AI_SYSTEM_INSTRUCTIONS", "gen_ai.system_instructions"
)
GEN_AI_INPUT_MESSAGES = _attr("GEN_AI_INPUT_MESSAGES", "gen_ai.input.messages")
GEN_AI_OUTPUT_MESSAGES = _attr(
    "GEN_AI_OUTPUT_MESSAGES", "gen_ai.output.messages"
)
GEN_AI_DATA_SOURCE_ID = _attr("GEN_AI_DATA_SOURCE_ID", "gen_ai.data_source.id")

# Token usage aliases for backwards compatibility
GEN_AI_USAGE_PROMPT_TOKENS = _attr(
    "GEN_AI_USAGE_PROMPT_TOKENS", "gen_ai.usage.prompt_tokens"
)
GEN_AI_USAGE_COMPLETION_TOKENS = _attr(
    "GEN_AI_USAGE_COMPLETION_TOKENS", "gen_ai.usage.completion_tokens"
)

# Non-spec attributes
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
GEN_AI_ORCHESTRATOR_AGENT_DEFINITIONS = "gen_ai.orchestrator.agent.definitions"
GEN_AI_GUARDRAIL_NAME = "gen_ai.guardrail.name"
GEN_AI_GUARDRAIL_TRIGGERED = "gen_ai.guardrail.triggered"
GEN_AI_HANDOFF_FROM_AGENT = "gen_ai.handoff.from_agent"
GEN_AI_HANDOFF_TO_AGENT = "gen_ai.handoff.to_agent"
GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"
GEN_AI_TOKEN_TYPE = _attr("GEN_AI_TOKEN_TYPE", "gen_ai.token.type")

# Normalization utilities


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    """Normalize provider name to spec-compliant value."""
    if not provider:
        return None
    normalized = provider.strip().lower()
    if normalized in GenAIProvider.ALL:
        return normalized
    return provider  # passthrough if unknown (forward compat)


def validate_tool_type(tool_type: Optional[str]) -> str:
    """Validate and normalize tool type."""
    if not tool_type:
        return GenAIToolType.FUNCTION  # default
    normalized = tool_type.strip().lower()
    return (
        normalized
        if normalized in GenAIToolType.ALL
        else GenAIToolType.FUNCTION
    )


def normalize_output_type(output_type: Optional[str]) -> str:
    """Normalize output type to spec-compliant value."""
    if not output_type:
        return GenAIOutputType.TEXT  # default
    normalized = output_type.strip().lower()
    base_map = {
        "json_object": GenAIOutputType.JSON,
        "jsonschema": GenAIOutputType.JSON,
        "speech_audio": GenAIOutputType.SPEECH,
        "audio_speech": GenAIOutputType.SPEECH,
        "image_png": GenAIOutputType.IMAGE,
        "function_arguments_json": GenAIOutputType.JSON,
        "tool_call": GenAIOutputType.JSON,
        "transcription_json": GenAIOutputType.JSON,
    }
    if normalized in base_map:
        return base_map[normalized]
    if normalized in {
        GenAIOutputType.TEXT,
        GenAIOutputType.JSON,
        GenAIOutputType.IMAGE,
        GenAIOutputType.SPEECH,
    }:
        return normalized
    return GenAIOutputType.TEXT  # default for unknown


logger = logging.getLogger(__name__)

GEN_AI_SYSTEM_KEY = getattr(GenAIAttributes, "GEN_AI_SYSTEM", "gen_ai.system")


class ContentCaptureMode(Enum):
    """Controls whether sensitive content is recorded on spans, events, or both."""

    NO_CONTENT = "no_content"
    SPAN_ONLY = "span_only"
    EVENT_ONLY = "event_only"
    SPAN_AND_EVENT = "span_and_event"

    @property
    def capture_in_span(self) -> bool:
        return self in (
            ContentCaptureMode.SPAN_ONLY,
            ContentCaptureMode.SPAN_AND_EVENT,
        )

    @property
    def capture_in_event(self) -> bool:
        return self in (
            ContentCaptureMode.EVENT_ONLY,
            ContentCaptureMode.SPAN_AND_EVENT,
        )


@dataclass
class ContentPayload:
    """Container for normalized content associated with a span."""

    input_messages: Optional[list[dict[str, Any]]] = None
    output_messages: Optional[list[dict[str, Any]]] = None
    system_instructions: Optional[list[dict[str, str]]] = None
    tool_arguments: Any = None
    tool_result: Any = None


def _is_instance_of(value: Any, classes: Any) -> bool:
    """Safe isinstance that tolerates typing.Any placeholders."""
    if not isinstance(classes, tuple):
        classes = (classes,)
    for cls in classes:
        try:
            if isinstance(value, cls):
                return True
        except TypeError:
            continue
    return False


def _infer_server_attributes(base_url: Optional[str]) -> dict[str, Any]:
    """Return server.address / server.port attributes if base_url provided."""
    out: dict[str, Any] = {}
    if not base_url:
        return out
    try:
        parsed = urlparse(base_url)
        if parsed.hostname:
            out[ServerAttributes.SERVER_ADDRESS] = parsed.hostname
        if parsed.port:
            out[ServerAttributes.SERVER_PORT] = parsed.port
    except Exception:
        return out
    return out


def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string (fallback to str)."""
    try:
        return gen_ai_json_dumps(obj)
    except (TypeError, ValueError):
        return str(obj)


def _serialize_tool_value(value: Any) -> Optional[str]:
    """Serialize tool input/output value to string."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return safe_json_dumps(value)
    return str(value)


def _as_utc_nano(dt: datetime) -> int:
    """Convert datetime to UTC nanoseconds timestamp."""
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def _get_span_status(span: Any) -> Status:
    """Get OpenTelemetry span status from agent span."""
    if error := getattr(span, "error", None):
        return Status(
            status_code=StatusCode.ERROR,
            description=f"{error.get('message', '')}: {error.get('data', '')}",
        )
    return Status(StatusCode.OK)


def get_span_name(
    operation_name: str,
    model: Optional[str] = None,
    agent_name: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> str:
    """Generate spec-compliant span name based on operation type."""
    base_name = operation_name

    if operation_name in {
        GenAIOperationName.CHAT,
        GenAIOperationName.TEXT_COMPLETION,
        GenAIOperationName.EMBEDDINGS,
        GenAIOperationName.TRANSCRIPTION,
        GenAIOperationName.SPEECH,
    }:
        return f"{base_name} {model}" if model else base_name

    if operation_name == GenAIOperationName.CREATE_AGENT:
        return f"{base_name} {agent_name}" if agent_name else base_name

    if operation_name == GenAIOperationName.INVOKE_AGENT:
        return f"{base_name} {agent_name}" if agent_name else base_name

    if operation_name == GenAIOperationName.EXECUTE_TOOL:
        return f"{base_name} {tool_name}" if tool_name else base_name

    if operation_name == GenAIOperationName.HANDOFF:
        return f"{base_name} {agent_name}" if agent_name else base_name

    return base_name


class GenAISemanticProcessor(TracingProcessor):
    """Trace processor adding GenAI semantic convention attributes with metrics."""

    def __init__(
        self,
        handler: TelemetryHandler,
        system_name: str = "openai",
        include_sensitive_data: bool = True,
        content_mode: ContentCaptureMode = ContentCaptureMode.SPAN_AND_EVENT,
        base_url: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
        agent_name_default: Optional[str] = None,
        agent_id_default: Optional[str] = None,
        agent_description_default: Optional[str] = None,
        base_url_default: Optional[str] = None,
        server_address_default: Optional[str] = None,
        server_port_default: Optional[int] = None,
    ):
        """Initialize processor.

        Args:
            handler: TelemetryHandler for creating spans via utils
            system_name: Provider name (openai/azure.ai.inference/etc.)
            include_sensitive_data: Include model/tool IO when True
            base_url: API endpoint for server.address/port
            server_address: Server address (can be overridden by env var or base_url)
            server_port: Server port (can be overridden by env var or base_url)
        """
        self.system_name = normalize_provider(system_name) or system_name
        self._content_mode = content_mode
        self.include_sensitive_data = include_sensitive_data and (
            content_mode.capture_in_span or content_mode.capture_in_event
        )
        effective_base_url = base_url or base_url_default
        self.base_url = effective_base_url

        # Agent defaults
        self._agent_name_default = agent_name_default
        self._agent_id_default = agent_id_default
        self._agent_description_default = agent_description_default

        # Server info
        self.server_address = server_address or server_address_default
        resolved_port = (
            server_port if server_port is not None else server_port_default
        )
        self.server_port = resolved_port

        # Infer from base_url if missing
        if (
            not self.server_address or not self.server_port
        ) and effective_base_url:
            server_attrs = _infer_server_attributes(effective_base_url)
            if not self.server_address:
                self.server_address = server_attrs.get(
                    ServerAttributes.SERVER_ADDRESS
                )
            if not self.server_port:
                self.server_port = server_attrs.get(
                    ServerAttributes.SERVER_PORT
                )

        # Content capture
        self._capture_messages = (
            content_mode.capture_in_span or content_mode.capture_in_event
        )
        self._capture_system_instructions = True
        self._capture_tool_definitions = True

        # Tracking
        self._invocations: Dict[str, _InvocationState] = {}
        self._handler = handler
        self._workflow: Workflow | None = None
        self._workflow_first_input: Optional[str] = None
        self._workflow_last_output: Optional[str] = None

    def _get_server_attributes(self) -> dict[str, Any]:
        """Get server attributes from configured values."""
        attrs = {}
        if self.server_address:
            attrs[ServerAttributes.SERVER_ADDRESS] = self.server_address
        if self.server_port:
            attrs[ServerAttributes.SERVER_PORT] = self.server_port
        return attrs

    def _init_metrics(self):
        """Initialize metric instruments."""
        self._meter = get_meter(
            "opentelemetry.instrumentation.openai_agents", "0.1.0"
        )

        # Use the shared Instruments class to ensure consistent bucket boundaries
        # per OpenTelemetry semantic conventions:
        # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md
        instruments = Instruments(self._meter)
        self._duration_histogram = instruments.operation_duration_histogram
        self._token_usage_histogram = instruments.token_usage_histogram

    def _record_metrics(
        self, span: Span[Any], attributes: dict[str, AttributeValue]
    ) -> None:
        """Record metrics for the span."""
        if not self._metrics_enabled or (
            self._duration_histogram is None
            and self._token_usage_histogram is None
        ):
            return

        try:
            # Calculate duration
            duration = None
            if hasattr(span, "started_at") and hasattr(span, "ended_at"):
                try:
                    start = datetime.fromisoformat(span.started_at)
                    end = datetime.fromisoformat(span.ended_at)
                    duration = (end - start).total_seconds()
                except Exception:
                    pass

            # Build metric attributes
            metric_attrs = {
                GEN_AI_PROVIDER_NAME: attributes.get(GEN_AI_PROVIDER_NAME),
                GEN_AI_OPERATION_NAME: attributes.get(GEN_AI_OPERATION_NAME),
                GEN_AI_REQUEST_MODEL: (
                    attributes.get(GEN_AI_REQUEST_MODEL)
                    or attributes.get(GEN_AI_RESPONSE_MODEL)
                ),
                ServerAttributes.SERVER_ADDRESS: attributes.get(
                    ServerAttributes.SERVER_ADDRESS
                ),
                ServerAttributes.SERVER_PORT: attributes.get(
                    ServerAttributes.SERVER_PORT
                ),
            }

            # Add error type if present
            if error := getattr(span, "error", None):
                error_type = error.get("type") or error.get("name")
                if error_type:
                    metric_attrs["error.type"] = error_type

            # Remove None values
            metric_attrs = {
                k: v for k, v in metric_attrs.items() if v is not None
            }

            # Record duration
            if duration is not None and self._duration_histogram is not None:
                self._duration_histogram.record(duration, metric_attrs)

            # Record token usage
            if self._token_usage_histogram:
                input_tokens = attributes.get(GEN_AI_USAGE_INPUT_TOKENS)
                if isinstance(input_tokens, (int, float)):
                    token_attrs = dict(metric_attrs)
                    token_attrs[GEN_AI_TOKEN_TYPE] = "input"
                    self._token_usage_histogram.record(
                        input_tokens, token_attrs
                    )

                output_tokens = attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS)
                if isinstance(output_tokens, (int, float)):
                    token_attrs = dict(metric_attrs)
                    token_attrs[GEN_AI_TOKEN_TYPE] = "output"
                    self._token_usage_histogram.record(
                        output_tokens, token_attrs
                    )

        except Exception as e:
            logger.debug("Failed to record metrics: %s", e)

    def _emit_content_events(
        self,
        span: Span[Any],
        otel_span: OtelSpan,
        payload: ContentPayload,
        agent_content: Optional[Dict[str, list[Any]]] = None,
    ) -> None:
        """Intentionally skip emitting gen_ai.* events to avoid payload duplication."""
        if (
            not self.include_sensitive_data
            or not self._content_mode.capture_in_event
            or not otel_span.is_recording()
        ):
            return

        logger.debug(
            "Event capture requested for span %s but is currently disabled",
            getattr(span, "span_id", "<unknown>"),
        )
        return

    def _collect_system_instructions(
        self, messages: Sequence[Any] | None
    ) -> list[dict[str, str]]:
        """Extract system/ai instructions as [{"type": "text", "content": "..."}]."""
        if not messages:
            return []
        out: list[dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role in {"system", "ai"}:
                content = m.get("content")
                out.extend(self._normalize_to_text_parts(content))
        return out

    def _normalize_to_text_parts(self, content: Any) -> list[dict[str, str]]:
        """Convert content to [{"type": "text", "content": ...}] format."""
        parts: list[dict[str, str]] = []
        if content is None:
            return parts
        if isinstance(content, str):
            parts.append({"type": "text", "content": content})
            return parts
        if isinstance(content, (list, tuple)):
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": "text", "content": item})
                elif isinstance(item, dict):
                    txt = item.get("text") or item.get("content")
                    if isinstance(txt, str) and txt:
                        parts.append({"type": "text", "content": txt})
                    else:
                        parts.append({"type": "text", "content": str(item)})
                else:
                    parts.append({"type": "text", "content": str(item)})
            return parts
        if isinstance(content, dict):
            txt = content.get("text") or content.get("content")
            if isinstance(txt, str) and txt:
                parts.append({"type": "text", "content": txt})
            else:
                parts.append({"type": "text", "content": str(content)})
            return parts
        parts.append({"type": "text", "content": str(content)})
        return parts

    def _redacted_text_parts(self) -> list[dict[str, str]]:
        """Return redacted text part."""
        return [{"type": "text", "content": "readacted"}]

    def _normalize_messages_to_role_parts(
        self, messages: Sequence[Any] | None
    ) -> list[dict[str, Any]]:
        """Normalize messages to {"role": ..., "parts": [...]} format."""
        if not messages:
            return []
        normalized: list[dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                # Fallback: treat as user text
                normalized.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "content": (
                                    "readacted"
                                    if not self.include_sensitive_data
                                    else str(m)
                                ),
                            }
                        ],
                    }
                )
                continue

            role = m.get("role") or "user"
            parts: list[dict[str, Any]] = []

            # Existing parts array
            if isinstance(m.get("parts"), (list, tuple)):
                for p in m["parts"]:
                    if isinstance(p, dict):
                        ptype = p.get("type") or "text"
                        newp: dict[str, Any] = {"type": ptype}
                        if ptype == "text":
                            txt = p.get("content") or p.get("text")
                            newp["content"] = (
                                "readacted"
                                if not self.include_sensitive_data
                                else (txt if isinstance(txt, str) else str(p))
                            )
                        elif ptype == "tool_call":
                            newp["id"] = p.get("id")
                            newp["name"] = p.get("name")
                            args = p.get("arguments")
                            newp["arguments"] = (
                                "readacted"
                                if not self.include_sensitive_data
                                else args
                            )
                        elif ptype == "tool_call_response":
                            newp["id"] = p.get("id") or m.get("tool_call_id")
                            result = p.get("result") or p.get("content")
                            newp["result"] = (
                                "readacted"
                                if not self.include_sensitive_data
                                else result
                            )
                        else:
                            newp["content"] = (
                                "readacted"
                                if not self.include_sensitive_data
                                else str(p)
                            )
                        parts.append(newp)
                    else:
                        parts.append(
                            {
                                "type": "text",
                                "content": (
                                    "readacted"
                                    if not self.include_sensitive_data
                                    else str(p)
                                ),
                            }
                        )

            # OpenAI content
            content = m.get("content")
            if isinstance(content, str):
                parts.append(
                    {
                        "type": "text",
                        "content": (
                            "readacted"
                            if not self.include_sensitive_data
                            else content
                        ),
                    }
                )
            elif isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict):
                        itype = item.get("type") or "text"
                        if itype == "text":
                            txt = item.get("text") or item.get("content")
                            parts.append(
                                {
                                    "type": "text",
                                    "content": (
                                        "readacted"
                                        if not self.include_sensitive_data
                                        else (
                                            txt
                                            if isinstance(txt, str)
                                            else str(item)
                                        )
                                    ),
                                }
                            )
                        else:
                            # Fallback for other part types
                            parts.append(
                                {
                                    "type": "text",
                                    "content": (
                                        "readacted"
                                        if not self.include_sensitive_data
                                        else str(item)
                                    ),
                                }
                            )
                    else:
                        parts.append(
                            {
                                "type": "text",
                                "content": (
                                    "readacted"
                                    if not self.include_sensitive_data
                                    else str(item)
                                ),
                            }
                        )

            # Assistant tool_calls
            if role == "assistant" and isinstance(
                m.get("tool_calls"), (list, tuple)
            ):
                for tc in m["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    p = {"type": "tool_call"}
                    p["id"] = tc.get("id")
                    fn = tc.get("function") or {}
                    if isinstance(fn, dict):
                        p["name"] = fn.get("name")
                        args = fn.get("arguments")
                        p["arguments"] = (
                            "readacted"
                            if not self.include_sensitive_data
                            else args
                        )
                    parts.append(p)

            # Tool call response
            if role in {"tool", "function"}:
                p = {"type": "tool_call_response"}
                p["id"] = m.get("tool_call_id") or m.get("id")
                result = m.get("result") or m.get("content")
                p["result"] = (
                    "readacted" if not self.include_sensitive_data else result
                )
                parts.append(p)

            if parts:
                normalized.append({"role": role, "parts": parts})
            elif not self.include_sensitive_data:
                normalized.append(
                    {"role": role, "parts": self._redacted_text_parts()}
                )

        return normalized

    def _format_output_item(self, item: Any) -> Optional[dict[str, Any]]:
        """Format a single output item into a proper part structure.

        Handles ResponseFunctionToolCall, ResponseReasoningItem, and other output types.
        """
        if not self.include_sensitive_data:
            return {"type": "text", "content": "redacted"}

        # Check for type attribute to identify special response items
        item_type = getattr(item, "type", None)

        # Handle function_call (ResponseFunctionToolCall)
        if item_type == "function_call":
            tool_name = getattr(item, "name", "unknown_tool")
            arguments = getattr(item, "arguments", "{}")
            call_id = getattr(item, "call_id", "")
            return {
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_call_id": call_id,
                "arguments": arguments,
            }

        # Handle reasoning (ResponseReasoningItem) - skip or summarize
        if item_type == "reasoning":
            # Reasoning items typically don't have user-visible content
            # Return None to skip, or include a marker if needed
            return None

        # Handle text content
        txt = getattr(item, "content", None)
        if isinstance(txt, str) and txt:
            return {"type": "text", "content": txt}

        # Handle message type (ResponseOutputMessage)
        if item_type == "message":
            msg_content = getattr(item, "content", None)
            if isinstance(msg_content, list):
                # Extract text from content parts
                texts = []
                for part in msg_content:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", "")
                        if text:
                            texts.append(text)
                if texts:
                    return {"type": "text", "content": "\n".join(texts)}
            elif isinstance(msg_content, str) and msg_content:
                return {"type": "text", "content": msg_content}

        # Fallback: try to extract meaningful data or skip
        # Don't stringify complex objects - return None to skip
        if hasattr(item, "model_dump"):
            # Pydantic model - dump to dict for cleaner output
            try:
                data = item.model_dump(exclude_none=True)
                if "content" in data and data["content"]:
                    return {"type": "text", "content": str(data["content"])}
                # For tool calls without proper type detection
                if "arguments" in data and "name" in data:
                    return {
                        "type": "tool_call",
                        "tool_name": data.get("name", "unknown"),
                        "tool_call_id": data.get("call_id", ""),
                        "arguments": data.get("arguments", "{}"),
                    }
            except Exception:
                pass

        # Last resort: stringify if it's a simple type
        if isinstance(item, str):
            return {"type": "text", "content": item}

        return None

    def _normalize_output_messages_to_role_parts(
        self, span_data: Any
    ) -> list[dict[str, Any]]:
        """Normalize output messages to enforced role+parts schema.

        Produces: [{"role": "assistant", "parts": [{"type": "text", "content": "..."}],
                    optional "finish_reason": "..." }]
        """
        messages: list[dict[str, Any]] = []
        parts: list[dict[str, Any]] = []
        finish_reason: Optional[str] = None

        # Response span: prefer consolidated output_text
        response = getattr(span_data, "response", None)
        if response is not None:
            # Collect text content
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                parts.append(
                    {
                        "type": "text",
                        "content": (
                            "readacted"
                            if not self.include_sensitive_data
                            else output_text
                        ),
                    }
                )
            else:
                output = getattr(response, "output", None)
                if isinstance(output, Sequence):
                    for item in output:
                        part = self._format_output_item(item)
                        if part:
                            parts.append(part)
                        # Capture finish_reason from parts when present
                        fr = getattr(item, "finish_reason", None)
                        if isinstance(fr, str) and not finish_reason:
                            finish_reason = fr

        # Generation span: use span_data.output
        if not parts:
            output = getattr(span_data, "output", None)
            if isinstance(output, Sequence):
                for item in output:
                    if isinstance(item, dict):
                        # Handle dict items (text, tool_call, etc.)
                        if item.get("type") == "text":
                            txt = item.get("content") or item.get("text")
                            if isinstance(txt, str) and txt:
                                parts.append(
                                    {
                                        "type": "text",
                                        "content": (
                                            "redacted"
                                            if not self.include_sensitive_data
                                            else txt
                                        ),
                                    }
                                )
                        elif item.get("type") == "function_call":
                            # Tool call in dict format
                            parts.append(
                                {
                                    "type": "tool_call",
                                    "tool_name": item.get("name", "unknown"),
                                    "tool_call_id": item.get("call_id", ""),
                                    "arguments": item.get("arguments", "{}"),
                                }
                            )
                        elif "content" in item and isinstance(
                            item["content"], str
                        ):
                            parts.append(
                                {
                                    "type": "text",
                                    "content": (
                                        "redacted"
                                        if not self.include_sensitive_data
                                        else item["content"]
                                    ),
                                }
                            )
                        if not finish_reason and isinstance(
                            item.get("finish_reason"), str
                        ):
                            finish_reason = item.get("finish_reason")
                    else:
                        # Use helper for non-dict items (Pydantic models, etc.)
                        part = self._format_output_item(item)
                        if part:
                            parts.append(part)
                        # Extract finish_reason if present
                        fr = getattr(item, "finish_reason", None)
                        if isinstance(fr, str) and not finish_reason:
                            finish_reason = fr

        # Build assistant message
        msg: dict[str, Any] = {"role": "assistant", "parts": parts}
        if finish_reason:
            msg["finish_reason"] = finish_reason
        # Only include if there is content
        if parts:
            messages.append(msg)
        return messages

    def _build_content_payload(self, span: Span[Any]) -> ContentPayload:
        """Normalize content from span data for attribute/event capture."""
        payload = ContentPayload()
        span_data = getattr(span, "span_data", None)
        if span_data is None or not self.include_sensitive_data:
            return payload

        capture_messages = self._capture_messages and (
            self._content_mode.capture_in_span
            or self._content_mode.capture_in_event
        )
        capture_system = self._capture_system_instructions and (
            self._content_mode.capture_in_span
            or self._content_mode.capture_in_event
        )
        capture_tools = self._content_mode.capture_in_span or (
            self._content_mode.capture_in_event
            and _is_instance_of(span_data, FunctionSpanData)
        )

        if _is_instance_of(span_data, GenerationSpanData):
            span_input = getattr(span_data, "input", None)
            if capture_messages and span_input:
                payload.input_messages = (
                    self._normalize_messages_to_role_parts(span_input)
                )
            if capture_system and span_input:
                sys_instr = self._collect_system_instructions(span_input)
                if sys_instr:
                    payload.system_instructions = sys_instr
            if capture_messages and (
                getattr(span_data, "output", None)
                or getattr(span_data, "response", None)
            ):
                normalized_out = self._normalize_output_messages_to_role_parts(
                    span_data
                )
                if normalized_out:
                    payload.output_messages = normalized_out

        elif _is_instance_of(span_data, ResponseSpanData):
            span_input = getattr(span_data, "input", None)
            if capture_messages and span_input:
                payload.input_messages = (
                    self._normalize_messages_to_role_parts(span_input)
                )
            if capture_system and span_input:
                sys_instr = self._collect_system_instructions(span_input)
                if sys_instr:
                    payload.system_instructions = sys_instr
            if capture_messages:
                normalized_out = self._normalize_output_messages_to_role_parts(
                    span_data
                )
                if normalized_out:
                    payload.output_messages = normalized_out

        elif _is_instance_of(span_data, FunctionSpanData) and capture_tools:
            payload.tool_arguments = _serialize_tool_value(
                getattr(span_data, "input", None)
            )
            payload.tool_result = _serialize_tool_value(
                getattr(span_data, "output", None)
            )

        return payload

    def _add_invocation_state(
        self,
        span_id: str,
        parent_span_id: Optional[str],
        invocation: Optional[
            Union[
                AgentCreation,
                AgentInvocation,
                LLMInvocation,
                ToolCall,
                Workflow,
            ]
        ] = None,
    ) -> _InvocationState:
        """Add state to tracking dict with parent-child relationship."""
        state = _InvocationState(
            invocation=invocation, parent_span_id=parent_span_id
        )
        self._invocations[span_id] = state

        # Establish parent-child relationship
        if parent_span_id is not None and parent_span_id in self._invocations:
            parent_state = self._invocations[parent_span_id]
            parent_state.children.append(span_id)

        return state

    def _set_invocation(
        self,
        span_id: str,
        invocation: Union[AgentInvocation, LLMInvocation, ToolCall],
    ) -> None:
        """Set invocation on existing state entry."""
        state = self._invocations.get(span_id)
        if state:
            state.invocation = invocation

    def _get_invocation_state(
        self, span_id: str
    ) -> Optional[_InvocationState]:
        """Get invocation state by span_id."""
        return self._invocations.get(span_id)

    def _delete_invocation_state(
        self, span_id: str
    ) -> Optional[_InvocationState]:
        """Delete an invocation state and return it."""
        return self._invocations.pop(span_id, None)

    def _fail_invocation(self, key: str, error: BaseException) -> None:
        """Fail an invocation with an error."""
        state = self._get_invocation_state(key)
        if state is None:
            return

        invocation = state.invocation
        gen_ai_error = Error(message=str(error), type=type(error))

        if isinstance(invocation, AgentInvocation):
            self._handler.fail_agent(invocation, gen_ai_error)
        elif isinstance(invocation, LLMInvocation):
            self._handler.fail_llm(invocation, gen_ai_error)
        elif isinstance(invocation, ToolCall):
            self._handler.fail_tool_call(invocation, gen_ai_error)
        elif isinstance(invocation, Workflow):
            self._handler.fail_workflow(invocation, gen_ai_error)

        self._delete_invocation_state(key)

    def _find_agent_parent_span_id(
        self, span_id: Optional[str]
    ) -> Optional[str]:
        """Return nearest ancestor span id that represents an agent."""
        current = span_id
        visited: set[str] = set()
        while current:
            if current in visited:
                break
            visited.add(current)
            state = self._invocations.get(current)
            if state:
                if isinstance(state.invocation, AgentInvocation):
                    return current
                current = state.parent_span_id
            else:
                break
        return None

    def _find_parent_agent_state(
        self, span_id: Optional[str]
    ) -> Optional[_InvocationState]:
        """Return the _InvocationState for the nearest agent ancestor."""
        agent_id = self._find_agent_parent_span_id(span_id)
        if agent_id:
            return self._invocations.get(agent_id)
        return None

    def _update_agent_aggregate(
        self, span: Span[Any], payload: ContentPayload
    ) -> None:
        """Accumulate child span content for parent agent span."""
        agent_id = self._find_agent_parent_span_id(span.parent_id)
        if not agent_id:
            return
        state = self._invocations.get(agent_id)
        if not state:
            return

        if payload.input_messages:
            state.input_messages = self._merge_content_sequence(
                state.input_messages, payload.input_messages
            )
        if payload.output_messages:
            state.output_messages = self._merge_content_sequence(
                state.output_messages, payload.output_messages
            )
        if payload.system_instructions:
            state.system_instructions = self._merge_content_sequence(
                state.system_instructions, payload.system_instructions
            )

        if not state.request_model:
            model = getattr(span.span_data, "model", None)
            if not model:
                response_obj = getattr(span.span_data, "response", None)
                model = getattr(response_obj, "model", None)
            if model:
                state.request_model = model

    def _infer_output_type(self, span_data: Any) -> str:
        """Infer gen_ai.output.type for multiple span kinds."""
        if _is_instance_of(span_data, FunctionSpanData):
            # Tool results are typically JSON
            return GenAIOutputType.JSON
        if _is_instance_of(span_data, TranscriptionSpanData):
            return GenAIOutputType.TEXT
        if _is_instance_of(span_data, SpeechSpanData):
            return GenAIOutputType.SPEECH
        if _is_instance_of(span_data, GuardrailSpanData):
            return GenAIOutputType.TEXT
        if _is_instance_of(span_data, HandoffSpanData):
            return GenAIOutputType.TEXT

        # Check for embeddings operation
        if _is_instance_of(span_data, GenerationSpanData):
            if hasattr(span_data, "embedding_dimension"):
                return (
                    GenAIOutputType.TEXT
                )  # Embeddings are numeric but represented as text

        # Generation/Response - check output structure
        output = getattr(span_data, "output", None) or getattr(
            getattr(span_data, "response", None), "output", None
        )
        if isinstance(output, Sequence) and output:
            first = output[0]
            if isinstance(first, dict):
                item_type = first.get("type")
                if isinstance(item_type, str):
                    normalized = item_type.strip().lower()
                    if normalized in {"image", "image_url"}:
                        return GenAIOutputType.IMAGE
                    if normalized in {"audio", "speech", "audio_url"}:
                        return GenAIOutputType.SPEECH
                    if normalized in {
                        "json",
                        "json_object",
                        "jsonschema",
                        "function_call",
                        "tool_call",
                        "tool_result",
                    }:
                        return GenAIOutputType.JSON
                    if normalized in {
                        "text",
                        "output_text",
                        "message",
                        "assistant",
                    }:
                        return GenAIOutputType.TEXT

                # Conversation style payloads
                if "role" in first:
                    parts = first.get("parts")
                    if isinstance(parts, Sequence) and parts:
                        # If all parts are textual (or missing explicit type), treat as text
                        textual = True
                        for part in parts:
                            if isinstance(part, dict):
                                part_type = str(part.get("type", "")).lower()
                                if part_type in {"image", "image_url"}:
                                    return GenAIOutputType.IMAGE
                                if part_type in {
                                    "audio",
                                    "speech",
                                    "audio_url",
                                }:
                                    return GenAIOutputType.SPEECH
                                if part_type and part_type not in {
                                    "text",
                                    "output_text",
                                    "assistant",
                                }:
                                    textual = False
                            elif not isinstance(part, str):
                                textual = False
                        if textual:
                            return GenAIOutputType.TEXT
                    content_value = first.get("content")
                    if isinstance(content_value, str):
                        return GenAIOutputType.TEXT

                # Detect structured data without explicit type
                json_like_keys = {
                    "schema",
                    "properties",
                    "arguments",
                    "result",
                    "data",
                    "json",
                    "output_json",
                }
                if json_like_keys.intersection(first.keys()):
                    return GenAIOutputType.JSON

        return GenAIOutputType.TEXT

    @staticmethod
    def _sanitize_usage_payload(usage: Any) -> None:
        """Remove non-spec usage fields (e.g., total tokens) in-place."""
        if not usage:
            return
        if isinstance(usage, dict):
            usage.pop("total_tokens", None)
            return
        if hasattr(usage, "total_tokens"):
            try:
                setattr(usage, "total_tokens", None)
            except Exception:  # pragma: no cover - defensive
                try:
                    delattr(usage, "total_tokens")
                except Exception:  # pragma: no cover - defensive
                    pass

    def _format_input_message(self, content: str) -> str:
        """Format input content as GenAI semantic convention message JSON."""
        # Check if already JSON formatted
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return content  # Already in correct format
        except (json.JSONDecodeError, TypeError):
            pass
        # Wrap in standard format
        input_msg = {
            "role": "user",
            "parts": [{"type": "text", "content": content}],
        }
        return json.dumps([input_msg])

    def _format_output_message(self, content: str) -> str:
        """Format output content as GenAI semantic convention message JSON."""
        # Check if already JSON formatted
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return content  # Already in correct format
        except (json.JSONDecodeError, TypeError):
            pass
        # Wrap in standard format
        output_msg = {
            "role": "assistant",
            "parts": [{"type": "text", "content": content}],
            "finish_reason": "stop",
        }
        return json.dumps([output_msg])

    def _make_input_messages(self, messages: list[Any]) -> list[InputMessage]:
        """Create InputMessage objects from message dicts (LangChain pattern)."""
        result: list[InputMessage] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user"))
                parts_data = msg.get("parts", [])
                parts: list[Any] = []
                for p in parts_data:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(Text(content=str(p.get("content", ""))))
                    else:
                        parts.append(p)
                if parts:
                    result.append(InputMessage(role=role, parts=parts))
        return result

    def _make_output_messages(
        self, messages: list[Any]
    ) -> list[OutputMessage]:
        """Create OutputMessage objects from message dicts (LangChain pattern)."""
        result: list[OutputMessage] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", "assistant"))
                parts_data = msg.get("parts", [])
                finish_reason = msg.get("finish_reason")
                parts: list[Any] = []
                for p in parts_data:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(Text(content=str(p.get("content", ""))))
                    else:
                        parts.append(p)
                if parts:
                    result.append(
                        OutputMessage(
                            role=role,
                            parts=parts,
                            finish_reason=finish_reason,
                        )
                    )
        return result

    def on_trace_start(self, trace: Trace) -> None:
        """Create workflow span when trace starts."""
        try:
            if self._workflow is None:
                self._workflow_first_input = None
                self._workflow_last_output = None
                metadata = getattr(trace, "metadata", None) or {}
                workflow_name = getattr(trace, "name", None) or "OpenAIAgents"

                # Check for initial_request in metadata to set workflow input
                initial_request = metadata.get("initial_request")
                input_messages: list[InputMessage] = []
                if initial_request:
                    input_messages = [
                        InputMessage(
                            role="user",
                            parts=[Text(content=str(initial_request))],
                        )
                    ]

                self._workflow = Workflow(
                    name=workflow_name,
                    workflow_type=metadata.get("workflow_type"),
                    description=metadata.get("description"),
                    input_messages=input_messages if input_messages else None,
                    attributes={},
                )
                invocation = self._handler.start_workflow(self._workflow)
                self._add_invocation_state(trace.trace_id, None, invocation)
        except Exception as e:
            logger.debug(
                "Failed to create workflow for trace %s: %s",
                getattr(trace, "trace_id", "<unknown>"),
                e,
                exc_info=True,
            )

    def on_trace_end(self, trace: Trace) -> None:
        """Stop workflow when trace ends."""
        key = str(trace.trace_id)
        state = self._get_invocation_state(key)
        if state is None:
            return

        workflow = state.invocation
        if isinstance(workflow, Workflow):
            # Parse and set input_messages from first agent's input (JSON string)
            if self._workflow_first_input and not workflow.input_messages:
                try:
                    parsed = json.loads(self._workflow_first_input)
                    if isinstance(parsed, list):
                        workflow.input_messages = self._make_input_messages(
                            parsed
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            # Parse and set output_messages from last agent's output (JSON string)
            if self._workflow_last_output:
                try:
                    parsed = json.loads(self._workflow_last_output)
                    if isinstance(parsed, list):
                        workflow.output_messages = self._make_output_messages(
                            parsed
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            self._handler.stop_workflow(workflow)

        self._workflow = None
        self._workflow_first_input = None
        self._workflow_last_output = None
        self._delete_invocation_state(key)

    def _handle_agent_span_start(
        self,
        span: Span[Any],
        key: str,
        parent_key: str,
        span_name: str,
        attributes: dict[str, Any],
        agent_name: Optional[str],
    ) -> None:
        """Handle AgentSpanData - create AgentInvocation."""
        try:
            agent_attrs: dict[str, Any] = dict(attributes)

            # Add workflow name to attributes before creating invocation
            if self._workflow is not None:
                if hasattr(self._workflow, "name") and self._workflow.name:
                    agent_attrs[GEN_AI_WORKFLOW_NAME] = self._workflow.name

            agent_invocation = AgentInvocation(
                name=agent_name or span_name,
                attributes=agent_attrs,
            )

            if self._workflow is not None:
                if hasattr(self._workflow, "span") and self._workflow.span:
                    agent_invocation.parent_span = self._workflow.span

            if agent_name:
                agent_invocation.agent_name = agent_name

            # Use span_id from the framework as agent_id
            agent_invocation.agent_id = str(span.span_id)

            agent_invocation.framework = "openai_agents"

            invocation = self._handler.start_agent(agent_invocation)
            self._add_invocation_state(key, parent_key, invocation)
        except Exception as e:
            logger.debug(
                "Failed to create AgentInvocation for span %s: %s",
                key,
                e,
                exc_info=True,
            )

    def _handle_llm_span_start(
        self,
        span: Span[Any],
        key: str,
        parent_key: str,
        span_name: str,
        model: Optional[str],
    ) -> None:
        """Handle GenerationSpanData/ResponseSpanData - create LLMInvocation."""
        try:
            # Get parent state for context
            parent_state = self._get_invocation_state(parent_key)
            parent_agent = (
                parent_state.invocation
                if parent_state
                and isinstance(parent_state.invocation, AgentInvocation)
                else None
            )

            # Get model - use from span or fallback to parent's stored model
            request_model: str = model if model else ""
            if not request_model and parent_state:
                request_model = parent_state.request_model or ""
            if not request_model:
                request_model = "unknown_model"

            # Build input messages from span data
            span_input = getattr(span.span_data, "input", None)
            input_messages: list[InputMessage] = []
            if span_input and isinstance(span_input, (list, tuple)):
                for msg in span_input:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        parts: list[Any] = []
                        # Handle different content formats
                        if isinstance(content, str):
                            parts.append(Text(content=content))
                        elif isinstance(content, list):
                            # Content is a list of parts
                            for part in content:
                                if isinstance(part, dict):
                                    part_type = part.get("type", "text")
                                    if part_type == "text":
                                        parts.append(
                                            Text(content=part.get("text", ""))
                                        )
                                    elif part_type == "input_text":
                                        parts.append(
                                            Text(content=part.get("text", ""))
                                        )
                                    else:
                                        # Keep other part types as-is
                                        parts.append(part)
                                elif isinstance(part, str):
                                    parts.append(Text(content=part))
                        if parts:
                            input_messages.append(
                                InputMessage(role=role, parts=parts)
                            )

            llm_invocation = LLMInvocation(
                request_model=request_model,
                input_messages=input_messages if input_messages else [],
            )

            # Set parent relationship - in agentic frameworks there's always a parent agent
            if parent_agent is not None:
                if hasattr(parent_agent, "span") and parent_agent.span:
                    llm_invocation.parent_span = parent_agent.span
                if (
                    hasattr(parent_agent, "agent_name")
                    and parent_agent.agent_name
                ):
                    llm_invocation.agent_name = parent_agent.agent_name
                # Use span_id from the framework as agent_id
                llm_invocation.agent_id = (
                    parent_agent.agent_id
                    if hasattr(parent_agent, "agent_id")
                    and parent_agent.agent_id
                    else None
                )

            llm_invocation.framework = "openai_agents"

            # Pass workflow name to LLM invocation
            if self._workflow is not None:
                if hasattr(self._workflow, "name") and self._workflow.name:
                    llm_invocation.attributes[GEN_AI_WORKFLOW_NAME] = (
                        self._workflow.name
                    )

            # Start LLM span immediately for correct duration
            invocation = self._handler.start_llm(llm_invocation)
            self._add_invocation_state(key, parent_key, invocation)
        except Exception as e:
            logger.debug(
                "Failed to create LLMInvocation for span %s: %s",
                key,
                e,
                exc_info=True,
            )

    def _handle_tool_span_start(
        self,
        span: Span[Any],
        key: str,
        parent_key: str,
        span_name: str,
        attributes: dict[str, Any],
    ) -> None:
        """Handle FunctionSpanData - create ToolCall."""
        try:
            tool_attrs: dict[str, Any] = dict(attributes)

            # Add workflow name to attributes before creating invocation
            if self._workflow is not None:
                if hasattr(self._workflow, "name") and self._workflow.name:
                    tool_attrs[GEN_AI_WORKFLOW_NAME] = self._workflow.name

            # Get tool arguments from span data
            tool_args = getattr(span.span_data, "input", None)

            # Get parent state for context
            parent_state = self._get_invocation_state(parent_key)
            parent_agent = (
                parent_state.invocation
                if parent_state
                and isinstance(parent_state.invocation, AgentInvocation)
                else None
            )

            # Get agent_id from parent agent (uses span_id from framework)
            actual_agent_id: Optional[str] = None
            if parent_agent is not None:
                if (
                    hasattr(parent_agent, "agent_name")
                    and parent_agent.agent_name
                ):
                    tool_attrs[GEN_AI_AGENT_NAME] = parent_agent.agent_name
                # Use agent_id (span_id from framework)
                actual_agent_id = (
                    parent_agent.agent_id
                    if hasattr(parent_agent, "agent_id")
                    and parent_agent.agent_id
                    else None
                )
                if actual_agent_id:
                    tool_attrs[GEN_AI_AGENT_ID] = actual_agent_id
                if GEN_AI_AGENT_DESCRIPTION in tool_attrs:
                    del tool_attrs[GEN_AI_AGENT_DESCRIPTION]

            tool_entity = ToolCall(
                name=getattr(span.span_data, "name", span_name),
                id=getattr(span.span_data, "call_id", None),
                arguments=tool_args,
                attributes=tool_attrs,
            )

            # Set parent relationship - in agentic frameworks there's always a parent agent
            if parent_agent is not None:
                tool_entity.agent_name = parent_agent.agent_name
                tool_entity.agent_id = actual_agent_id
                if hasattr(parent_agent, "span") and parent_agent.span:
                    tool_entity.parent_span = parent_agent.span

            tool_entity.framework = "openai_agents"

            invocation = self._handler.start_tool_call(tool_entity)
            self._add_invocation_state(key, parent_key, invocation)
        except Exception as e:
            logger.debug(
                "Failed to create ToolCall for span %s: %s",
                key,
                e,
                exc_info=True,
            )

    def on_span_start(self, span: Span[Any]) -> None:
        """Start invocation tracking for a span."""
        if not span.started_at:
            return

        key = str(span.span_id)
        parent_key = (
            str(span.parent_id) if span.parent_id else str(span.trace_id)
        )

        operation_name = self._get_operation_name(span.span_data)
        model = getattr(span.span_data, "model", None)
        if model is None:
            response_obj = getattr(span.span_data, "response", None)
            model = getattr(response_obj, "model", None)
        # Try model_config if model still not found
        if model is None:
            model_config = getattr(span.span_data, "model_config", None)
            if model_config and isinstance(model_config, dict):
                model = model_config.get("model")

        # Store model in parent agent's state for subsequent spans to use
        if model and _is_instance_of(
            span.span_data, (GenerationSpanData, ResponseSpanData)
        ):
            parent_agent_state = self._find_parent_agent_state(span.parent_id)
            if parent_agent_state and not parent_agent_state.request_model:
                parent_agent_state.request_model = model

        # Get agent name from span data or use default
        agent_name = None
        if _is_instance_of(span.span_data, AgentSpanData):
            agent_name = getattr(span.span_data, "name", None)
        if not agent_name:
            agent_name = self._agent_name_default

        tool_name = (
            getattr(span.span_data, "name", None)
            if _is_instance_of(span.span_data, FunctionSpanData)
            else None
        )

        # Generate spec-compliant span name
        span_name = get_span_name(operation_name, model, agent_name, tool_name)

        attributes: dict[str, Any] = {
            GEN_AI_PROVIDER_NAME: self.system_name,
            GEN_AI_SYSTEM_KEY: self.system_name,
            GEN_AI_OPERATION_NAME: operation_name,
        }

        attributes.update(self._get_server_attributes())

        # Dispatch to type-specific handlers
        if _is_instance_of(span.span_data, AgentSpanData):
            self._handle_agent_span_start(
                span, key, parent_key, span_name, attributes, agent_name
            )
        elif _is_instance_of(
            span.span_data, (GenerationSpanData, ResponseSpanData)
        ):
            self._handle_llm_span_start(
                span, key, parent_key, span_name, model
            )
        elif _is_instance_of(span.span_data, FunctionSpanData):
            self._handle_tool_span_start(
                span, key, parent_key, span_name, attributes
            )

    def _handle_agent_span_end(
        self,
        key: str,
        invocation: AgentInvocation,
        state: _InvocationState,
    ) -> None:
        """Handle AgentInvocation end - finalize agent span."""
        try:
            # Populate input_messages from accumulated state (LangChain pattern)
            if state.input_messages and not invocation.input_messages:
                invocation.input_messages = self._make_input_messages(
                    state.input_messages
                )

            # Populate output_messages from accumulated state (LangChain pattern)
            if state.output_messages:
                invocation.output_messages = self._make_output_messages(
                    state.output_messages
                )

            self._handler.stop_agent(invocation)
        except Exception as e:
            logger.debug(
                "Failed to stop AgentInvocation for span %s: %s",
                key,
                e,
                exc_info=True,
            )

    def _handle_llm_span_end(
        self,
        span: Span[Any],
        key: str,
        invocation: LLMInvocation,
        payload: ContentPayload,
    ) -> None:
        """Handle LLMInvocation end - finalize LLM span with response data."""
        try:
            # Update model from response if available
            response_obj = getattr(span.span_data, "response", None)
            if response_obj is not None:
                response_model = getattr(response_obj, "model", None)
                if response_model:
                    invocation.response_model_name = response_model
                    # Update request_model and span name if it was unknown
                    if invocation.request_model == "unknown_model":
                        invocation.request_model = response_model
                        # Update span name with actual model
                        if invocation.span is not None and hasattr(
                            invocation.span, "update_name"
                        ):
                            operation = getattr(
                                invocation, "operation", "chat"
                            )
                            invocation.span.update_name(
                                f"{operation} {response_model}"
                            )

            # Add input messages from payload (if not already set at start)
            if payload.input_messages and not invocation.input_messages:
                input_msgs: list[InputMessage] = []
                for msg in payload.input_messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        parts_data = msg.get("parts", [])
                        parts: list[Any] = []
                        for p in parts_data:
                            if isinstance(p, dict) and p.get("type") == "text":
                                parts.append(
                                    Text(content=p.get("content", ""))
                                )
                            else:
                                parts.append(p)
                        input_msgs.append(InputMessage(role=role, parts=parts))
                if input_msgs:
                    invocation.input_messages = input_msgs

            # Add output messages from payload
            if payload.output_messages:
                output_msgs: list[OutputMessage] = []
                for msg in payload.output_messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "assistant")
                        parts_data = msg.get("parts", [])
                        parts: list[Any] = []
                        for p in parts_data:
                            if isinstance(p, dict) and p.get("type") == "text":
                                parts.append(
                                    Text(content=p.get("content", ""))
                                )
                            else:
                                parts.append(p)
                        finish_reason = msg.get("finish_reason", "stop")
                        output_msgs.append(
                            OutputMessage(
                                role=role,
                                parts=parts,
                                finish_reason=finish_reason,
                            )
                        )
                if output_msgs:
                    invocation.output_messages = output_msgs

            # Add token usage from response
            if response_obj is not None:
                usage = getattr(response_obj, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", None)
                    if input_tokens is None:
                        input_tokens = getattr(usage, "prompt_tokens", None)
                    if input_tokens is not None:
                        invocation.input_tokens = input_tokens

                    output_tokens = getattr(usage, "output_tokens", None)
                    if output_tokens is None:
                        output_tokens = getattr(
                            usage, "completion_tokens", None
                        )
                    if output_tokens is not None:
                        invocation.output_tokens = output_tokens

            self._handler.stop_llm(invocation)
        except Exception as e:
            logger.debug(
                "Failed to stop LLMInvocation for span %s: %s",
                key,
                e,
                exc_info=True,
            )

    def _handle_tool_span_end(
        self,
        key: str,
        invocation: ToolCall,
        payload: ContentPayload,
    ) -> None:
        """Handle ToolCall end - finalize tool span with result."""
        try:
            # Add tool result from payload
            if payload.tool_result is not None:
                invocation.attributes.setdefault(
                    "tool.response",
                    safe_json_dumps(payload.tool_result),
                )
            self._handler.stop_tool_call(invocation)
        except Exception as e:
            logger.debug(
                "Failed to stop ToolCall for span %s: %s",
                key,
                e,
                exc_info=True,
            )

    def on_span_end(self, span: Span[Any]) -> None:
        """Finalize span with attributes, events, and metrics.

        Uses unified _invocations dict for all entity tracking.
        """
        key = str(span.span_id)
        state = self._get_invocation_state(key)
        if state is None:
            return

        payload = self._build_content_payload(span)
        self._update_agent_aggregate(span, payload)

        # Track workflow input/output from agent spans
        if isinstance(state.invocation, AgentInvocation):
            if state.input_messages:
                input_data = safe_json_dumps(state.input_messages)
                if self._workflow_first_input is None:
                    self._workflow_first_input = input_data
            if state.output_messages:
                self._workflow_last_output = safe_json_dumps(
                    state.output_messages
                )

        # Dispatch to type-specific handlers
        invocation = state.invocation
        if isinstance(invocation, AgentInvocation):
            self._handle_agent_span_end(key, invocation, state)
        elif isinstance(invocation, LLMInvocation):
            self._handle_llm_span_end(span, key, invocation, payload)
        elif isinstance(invocation, ToolCall):
            self._handle_tool_span_end(key, invocation, payload)

        # Remove from invocations
        self._delete_invocation_state(key)

    def on_span_error(self, span: Span[Any], error: BaseException) -> None:
        """Handle span error by failing the invocation."""
        key = str(span.span_id)
        self._fail_invocation(key, error)

    def on_trace_error(self, trace: Trace, error: BaseException) -> None:
        """Handle trace error by failing the workflow."""
        key = str(trace.trace_id)
        self._fail_invocation(key, error)

    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        # Stop any active workflow first (if not already stopped by on_trace_end)
        if self._workflow is not None:
            if (
                self._workflow_first_input
                and not self._workflow.input_messages
            ):
                try:
                    parsed = json.loads(self._workflow_first_input)
                    if isinstance(parsed, list):
                        self._workflow.input_messages = (
                            self._make_input_messages(parsed)
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            if self._workflow_last_output:
                try:
                    parsed = json.loads(self._workflow_last_output)
                    if isinstance(parsed, list):
                        self._workflow.output_messages = (
                            self._make_output_messages(parsed)
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            self._workflow = None
            self._workflow_first_input = None
            self._workflow_last_output = None

        self._invocations.clear()

    def force_flush(self) -> None:
        """Force flush (no-op for this processor)."""
        pass

    def _get_operation_name(self, span_data: Any) -> str:
        """Determine operation name from span data type."""
        if _is_instance_of(span_data, GenerationSpanData):
            # Check if it's embeddings
            if hasattr(span_data, "embedding_dimension"):
                return GenAIOperationName.EMBEDDINGS
            # Check if it's chat or completion
            if span_data.input:
                first_input = span_data.input[0] if span_data.input else None
                if isinstance(first_input, dict) and "role" in first_input:
                    return GenAIOperationName.CHAT
            return GenAIOperationName.TEXT_COMPLETION
        if _is_instance_of(span_data, AgentSpanData):
            # Could be create_agent or invoke_agent based on context
            operation = getattr(span_data, "operation", None)
            normalized = (
                operation.strip().lower()
                if isinstance(operation, str)
                else None
            )
            if normalized in {"create", "create_agent"}:
                return GenAIOperationName.CREATE_AGENT
            if normalized in {"invoke", "invoke_agent"}:
                return GenAIOperationName.INVOKE_AGENT
            return GenAIOperationName.INVOKE_AGENT
        if _is_instance_of(span_data, FunctionSpanData):
            return GenAIOperationName.EXECUTE_TOOL
        if _is_instance_of(span_data, ResponseSpanData):
            return GenAIOperationName.CHAT  # Response typically from chat
        if _is_instance_of(span_data, TranscriptionSpanData):
            return GenAIOperationName.TRANSCRIPTION
        if _is_instance_of(span_data, SpeechSpanData):
            return GenAIOperationName.SPEECH
        if _is_instance_of(span_data, GuardrailSpanData):
            return GenAIOperationName.GUARDRAIL
        if _is_instance_of(span_data, HandoffSpanData):
            return GenAIOperationName.HANDOFF
        return "unknown"

    def _extract_genai_attributes(
        self,
        span: Span[Any],
        payload: ContentPayload,
        agent_content: Optional[Dict[str, list[Any]]] = None,
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Yield (attr, value) pairs for GenAI semantic conventions."""
        span_data = span.span_data

        # Base attributes
        yield GEN_AI_PROVIDER_NAME, self.system_name
        yield GEN_AI_SYSTEM_KEY, self.system_name

        # Server attributes
        for key, value in self._get_server_attributes().items():
            yield key, value

        # Process different span types
        if _is_instance_of(span_data, GenerationSpanData):
            yield from self._get_attributes_from_generation_span_data(
                span_data, payload
            )
        elif _is_instance_of(span_data, AgentSpanData):
            yield from self._get_attributes_from_agent_span_data(
                span_data, agent_content
            )
        elif _is_instance_of(span_data, FunctionSpanData):
            yield from self._get_attributes_from_function_span_data(
                span_data, payload
            )
        elif _is_instance_of(span_data, ResponseSpanData):
            yield from self._get_attributes_from_response_span_data(
                span_data, payload
            )
        elif _is_instance_of(span_data, TranscriptionSpanData):
            yield from self._get_attributes_from_transcription_span_data(
                span_data
            )
        elif _is_instance_of(span_data, SpeechSpanData):
            yield from self._get_attributes_from_speech_span_data(span_data)
        elif _is_instance_of(span_data, GuardrailSpanData):
            yield from self._get_attributes_from_guardrail_span_data(span_data)
        elif _is_instance_of(span_data, HandoffSpanData):
            yield from self._get_attributes_from_handoff_span_data(span_data)

    def _get_attributes_from_generation_span_data(
        self, span_data: GenerationSpanData, payload: ContentPayload
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from generation span."""
        # Operation name
        operation_name = self._get_operation_name(span_data)
        yield GEN_AI_OPERATION_NAME, operation_name

        # Model information
        if span_data.model:
            yield GEN_AI_REQUEST_MODEL, span_data.model

        # Check for embeddings-specific attributes
        if hasattr(span_data, "embedding_dimension"):
            yield (
                GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
                span_data.embedding_dimension,
            )

        # Check for data source
        if hasattr(span_data, "data_source_id"):
            yield GEN_AI_DATA_SOURCE_ID, span_data.data_source_id

        finish_reasons: list[Any] = []
        if span_data.output:
            for part in span_data.output:
                if isinstance(part, dict):
                    fr = part.get("finish_reason") or part.get("stop_reason")
                else:
                    fr = getattr(part, "finish_reason", None)
                if fr:
                    finish_reasons.append(
                        fr if isinstance(fr, str) else str(fr)
                    )
        if finish_reasons:
            yield GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons

        # Usage information
        if span_data.usage:
            usage = span_data.usage
            self._sanitize_usage_payload(usage)
            if "prompt_tokens" in usage or "input_tokens" in usage:
                tokens = usage.get("prompt_tokens") or usage.get(
                    "input_tokens"
                )
                if tokens is not None:
                    yield GEN_AI_USAGE_INPUT_TOKENS, tokens
            if "completion_tokens" in usage or "output_tokens" in usage:
                tokens = usage.get("completion_tokens") or usage.get(
                    "output_tokens"
                )
                if tokens is not None:
                    yield GEN_AI_USAGE_OUTPUT_TOKENS, tokens

        # Model configuration
        if span_data.model_config:
            mc = span_data.model_config
            param_map = {
                "temperature": GEN_AI_REQUEST_TEMPERATURE,
                "top_p": GEN_AI_REQUEST_TOP_P,
                "top_k": GEN_AI_REQUEST_TOP_K,
                "max_tokens": GEN_AI_REQUEST_MAX_TOKENS,
                "presence_penalty": GEN_AI_REQUEST_PRESENCE_PENALTY,
                "frequency_penalty": GEN_AI_REQUEST_FREQUENCY_PENALTY,
                "seed": GEN_AI_REQUEST_SEED,
                "n": GEN_AI_REQUEST_CHOICE_COUNT,
                "stop": GEN_AI_REQUEST_STOP_SEQUENCES,
                "encoding_formats": GEN_AI_REQUEST_ENCODING_FORMATS,
            }
            for k, attr in param_map.items():
                if hasattr(mc, "__contains__") and k in mc:
                    value = mc[k]
                else:
                    value = getattr(mc, k, None)
                if value is not None:
                    yield attr, value

            if hasattr(mc, "get"):
                base_url = (
                    mc.get("base_url")
                    or mc.get("baseUrl")
                    or mc.get("endpoint")
                )
            else:
                base_url = (
                    getattr(mc, "base_url", None)
                    or getattr(mc, "baseUrl", None)
                    or getattr(mc, "endpoint", None)
                )
            for key, value in _infer_server_attributes(base_url).items():
                yield key, value

        # Sensitive data capture
        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and payload.input_messages
        ):
            yield (
                GEN_AI_INPUT_MESSAGES,
                safe_json_dumps(payload.input_messages),
            )

        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_system_instructions
            and payload.system_instructions
        ):
            yield (
                GEN_AI_SYSTEM_INSTRUCTIONS,
                safe_json_dumps(payload.system_instructions),
            )

        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and payload.output_messages
        ):
            yield (
                GEN_AI_OUTPUT_MESSAGES,
                safe_json_dumps(payload.output_messages),
            )

        # Output type
        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _merge_content_sequence(
        self,
        existing: list[Any],
        incoming: Sequence[Any],
    ) -> list[Any]:
        """Merge normalized message/content lists without duplicating snapshots."""
        if not incoming:
            return existing

        incoming_list = [self._clone_message(item) for item in incoming]

        if self.include_sensitive_data:
            filtered = [
                msg
                for msg in incoming_list
                if not self._is_placeholder_message(msg)
            ]
            if filtered:
                incoming_list = filtered

        if not existing:
            return incoming_list

        result = [self._clone_message(item) for item in existing]

        for idx, new_msg in enumerate(incoming_list):
            if idx < len(result):
                if (
                    self.include_sensitive_data
                    and self._is_placeholder_message(new_msg)
                    and not self._is_placeholder_message(result[idx])
                ):
                    continue
                if result[idx] != new_msg:
                    result[idx] = self._clone_message(new_msg)
            else:
                if (
                    self.include_sensitive_data
                    and self._is_placeholder_message(new_msg)
                ):
                    if (
                        any(
                            not self._is_placeholder_message(existing_msg)
                            for existing_msg in result
                        )
                        or new_msg in result
                    ):
                        continue
                result.append(self._clone_message(new_msg))

        return result

    def _clone_message(self, message: Any) -> Any:
        if isinstance(message, dict):
            return {
                key: (
                    self._clone_message(value)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in message.items()
            }
        if isinstance(message, list):
            return [self._clone_message(item) for item in message]
        return message

    def _is_placeholder_message(self, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        parts = message.get("parts")
        if not isinstance(parts, list) or not parts:
            return False
        for part in parts:
            if (
                not isinstance(part, dict)
                or part.get("type") != "text"
                or part.get("content") != "readacted"
            ):
                return False
        return True

    def _get_attributes_from_agent_span_data(
        self,
        span_data: AgentSpanData,
        agent_content: Optional[Dict[str, list[Any]]] = None,
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from agent span."""
        yield GEN_AI_OPERATION_NAME, self._get_operation_name(span_data)

        name = getattr(span_data, "name", None) or self._agent_name_default
        if name:
            yield GEN_AI_AGENT_NAME, name

        agent_id = (
            getattr(span_data, "agent_id", None) or self._agent_id_default
        )
        if agent_id:
            yield GEN_AI_AGENT_ID, agent_id

        description = (
            getattr(span_data, "description", None)
            or self._agent_description_default
        )
        if description:
            yield GEN_AI_AGENT_DESCRIPTION, description

        model = getattr(span_data, "model", None)
        if not model and agent_content:
            model = agent_content.get("request_model")
        if model:
            yield GEN_AI_REQUEST_MODEL, model

        if hasattr(span_data, "conversation_id") and span_data.conversation_id:
            yield GEN_AI_CONVERSATION_ID, span_data.conversation_id

        # Agent definitions
        if self._capture_tool_definitions and hasattr(
            span_data, "agent_definitions"
        ):
            yield (
                GEN_AI_ORCHESTRATOR_AGENT_DEFINITIONS,
                safe_json_dumps(span_data.agent_definitions),
            )

        # System instructions from agent definitions
        if self._capture_system_instructions and hasattr(
            span_data, "agent_definitions"
        ):
            try:
                defs = span_data.agent_definitions
                if isinstance(defs, (list, tuple)):
                    collected: list[dict[str, str]] = []
                    for d in defs:
                        if isinstance(d, dict):
                            msgs = d.get("messages") or d.get(
                                "system_messages"
                            )
                            if isinstance(msgs, (list, tuple)):
                                collected.extend(
                                    self._collect_system_instructions(msgs)
                                )
                    if collected:
                        yield (
                            GEN_AI_SYSTEM_INSTRUCTIONS,
                            safe_json_dumps(collected),
                        )
            except Exception:
                pass

        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and agent_content
        ):
            if agent_content.get("input_messages"):
                yield (
                    GEN_AI_INPUT_MESSAGES,
                    safe_json_dumps(agent_content["input_messages"]),
                )
            if agent_content.get("output_messages"):
                yield (
                    GEN_AI_OUTPUT_MESSAGES,
                    safe_json_dumps(agent_content["output_messages"]),
                )
        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_system_instructions
            and agent_content
            and agent_content.get("system_instructions")
        ):
            yield (
                GEN_AI_SYSTEM_INSTRUCTIONS,
                safe_json_dumps(agent_content["system_instructions"]),
            )

        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _get_attributes_from_function_span_data(
        self, span_data: FunctionSpanData, payload: ContentPayload
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from function/tool span."""
        yield GEN_AI_OPERATION_NAME, GenAIOperationName.EXECUTE_TOOL

        if span_data.name:
            yield GEN_AI_TOOL_NAME, span_data.name

        # Tool type - validate and normalize
        tool_type = "function"  # Default for function spans
        if hasattr(span_data, "tool_type"):
            tool_type = span_data.tool_type
        yield GEN_AI_TOOL_TYPE, validate_tool_type(tool_type)

        if hasattr(span_data, "call_id") and span_data.call_id:
            yield GEN_AI_TOOL_CALL_ID, span_data.call_id
        if hasattr(span_data, "description") and span_data.description:
            yield GEN_AI_TOOL_DESCRIPTION, span_data.description

        # Tool definitions
        if self._capture_tool_definitions and hasattr(
            span_data, "tool_definitions"
        ):
            yield (
                GEN_AI_TOOL_DEFINITIONS,
                safe_json_dumps(span_data.tool_definitions),
            )

        # Tool input/output (sensitive)
        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and payload.tool_arguments is not None
        ):
            yield GEN_AI_TOOL_CALL_ARGUMENTS, payload.tool_arguments

        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and payload.tool_result is not None
        ):
            yield GEN_AI_TOOL_CALL_RESULT, payload.tool_result

        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _get_attributes_from_response_span_data(
        self, span_data: ResponseSpanData, payload: ContentPayload
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from response span."""
        yield GEN_AI_OPERATION_NAME, GenAIOperationName.CHAT

        # Response information
        if span_data.response:
            if hasattr(span_data.response, "id") and span_data.response.id:
                yield GEN_AI_RESPONSE_ID, span_data.response.id

            # Model from response
            if (
                hasattr(span_data.response, "model")
                and span_data.response.model
            ):
                yield GEN_AI_RESPONSE_MODEL, span_data.response.model
                if not getattr(span_data, "model", None):
                    yield GEN_AI_REQUEST_MODEL, span_data.response.model

            # Finish reasons
            finish_reasons = []
            if (
                hasattr(span_data.response, "output")
                and span_data.response.output
            ):
                for part in span_data.response.output:
                    if isinstance(part, dict):
                        fr = part.get("finish_reason") or part.get(
                            "stop_reason"
                        )
                    else:
                        fr = getattr(part, "finish_reason", None)
                    if fr:
                        finish_reasons.append(fr)
            if finish_reasons:
                yield GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons

            # Usage from response
            if (
                hasattr(span_data.response, "usage")
                and span_data.response.usage
            ):
                usage = span_data.response.usage
                self._sanitize_usage_payload(usage)
                input_tokens = getattr(usage, "input_tokens", None)
                if input_tokens is None:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                if input_tokens is not None:
                    yield GEN_AI_USAGE_INPUT_TOKENS, input_tokens

                output_tokens = getattr(usage, "output_tokens", None)
                if output_tokens is None:
                    output_tokens = getattr(usage, "completion_tokens", None)
                if output_tokens is not None:
                    yield GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens

        # Input/output messages
        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and payload.input_messages
        ):
            yield (
                GEN_AI_INPUT_MESSAGES,
                safe_json_dumps(payload.input_messages),
            )

        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_system_instructions
            and payload.system_instructions
        ):
            yield (
                GEN_AI_SYSTEM_INSTRUCTIONS,
                safe_json_dumps(payload.system_instructions),
            )

        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and payload.output_messages
        ):
            yield (
                GEN_AI_OUTPUT_MESSAGES,
                safe_json_dumps(payload.output_messages),
            )

        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _get_attributes_from_transcription_span_data(
        self, span_data: TranscriptionSpanData
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from transcription span."""
        yield GEN_AI_OPERATION_NAME, GenAIOperationName.TRANSCRIPTION

        if hasattr(span_data, "model") and span_data.model:
            yield GEN_AI_REQUEST_MODEL, span_data.model

        # Audio format
        if hasattr(span_data, "format") and span_data.format:
            yield "gen_ai.audio.input.format", span_data.format

        # Transcript (sensitive)
        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and hasattr(span_data, "transcript")
        ):
            yield "gen_ai.transcription.text", span_data.transcript

        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _get_attributes_from_speech_span_data(
        self, span_data: SpeechSpanData
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from speech span."""
        yield GEN_AI_OPERATION_NAME, GenAIOperationName.SPEECH

        if hasattr(span_data, "model") and span_data.model:
            yield GEN_AI_REQUEST_MODEL, span_data.model

        if hasattr(span_data, "voice") and span_data.voice:
            yield "gen_ai.speech.voice", span_data.voice

        if hasattr(span_data, "format") and span_data.format:
            yield "gen_ai.audio.output.format", span_data.format

        # Input text (sensitive)
        if (
            self.include_sensitive_data
            and self._content_mode.capture_in_span
            and self._capture_messages
            and hasattr(span_data, "input_text")
        ):
            yield "gen_ai.speech.input_text", span_data.input_text

        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _get_attributes_from_guardrail_span_data(
        self, span_data: GuardrailSpanData
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from guardrail span."""
        yield GEN_AI_OPERATION_NAME, GenAIOperationName.GUARDRAIL

        if span_data.name:
            yield GEN_AI_GUARDRAIL_NAME, span_data.name

        yield GEN_AI_GUARDRAIL_TRIGGERED, span_data.triggered
        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _get_attributes_from_handoff_span_data(
        self, span_data: HandoffSpanData
    ) -> Iterator[tuple[str, AttributeValue]]:
        """Extract attributes from handoff span."""
        yield GEN_AI_OPERATION_NAME, GenAIOperationName.HANDOFF

        if span_data.from_agent:
            yield GEN_AI_HANDOFF_FROM_AGENT, span_data.from_agent

        if span_data.to_agent:
            yield GEN_AI_HANDOFF_TO_AGENT, span_data.to_agent

        yield (
            GEN_AI_OUTPUT_TYPE,
            normalize_output_type(self._infer_output_type(span_data)),
        )

    def _cleanup_spans_for_trace(self, trace_id: str) -> None:
        """Clean up entities for a trace to prevent memory leaks."""
        # Trace cleanup is a no-op since span_ids are UUIDs unrelated to trace_id.
        # Invocations are already cleaned up in on_span_end via _delete_invocation_state.
        # This method is kept for interface consistency but does nothing.
        pass


__all__ = [
    "GenAIProvider",
    "GenAIOperationName",
    "GenAIToolType",
    "GenAIOutputType",
    "GenAIEvaluationAttributes",
    "ContentCaptureMode",
    "ContentPayload",
    "GenAISemanticProcessor",
    "normalize_provider",
    "normalize_output_type",
    "validate_tool_type",
]
