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


import timeit
from contextvars import Token
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from opentelemetry.util.genai._error import (  # noqa: F401 - re-export
    Error,
    ErrorClassification,
    EvaluationResult,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai.handler import TelemetryHandler

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span, SpanContext

# Backward compatibility: older semconv builds may miss new GEN_AI attributes
if not hasattr(GenAIAttributes, "GEN_AI_PROVIDER_NAME"):
    GenAIAttributes.GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

# Import security attribute from centralized attributes module
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.attributes import (
    GEN_AI_REQUEST_STREAM,
    GEN_AI_SECURITY_EVENT_ID,
    GEN_AI_TOOL_DEFINITIONS,
)
from opentelemetry.util.types import AttributeValue

ContextToken = Token  # simple alias; avoid TypeAlias warning tools


class ContentCapturingMode(Enum):
    # Do not capture content (default).
    NO_CONTENT = 0
    # Only capture content in spans.
    SPAN_ONLY = 1
    # Only capture content in events.
    EVENT_ONLY = 2
    # Capture content in both spans and events.
    SPAN_AND_EVENT = 3


def _new_input_messages() -> list["InputMessage"]:  # quotes for forward ref
    return []


def _new_output_messages() -> list["OutputMessage"]:  # quotes for forward ref
    return []


def _new_str_any_dict() -> dict[str, Any]:
    return {}


@dataclass(kw_only=True)
class GenAI(GenAIInvocation):
    """Base type for all GenAI telemetry entities."""

    context_token: Optional[ContextToken] = None
    span: Optional[Span] = None
    span_context: Optional[SpanContext] = None
    trace_id: Optional[int] = None
    span_id: Optional[int] = None
    trace_flags: Optional[int] = None
    start_time: float = field(default_factory=timeit.default_timer)
    end_time: Optional[float] = None
    provider: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_PROVIDER_NAME},
    )
    framework: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    agent_name: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_AGENT_NAME},
    )
    agent_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_AGENT_ID},
    )
    system: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_SYSTEM},
    )
    conversation_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_CONVERSATION_ID},
    )
    conversation_root: Optional[bool] = field(
        default=None,
        metadata={"semconv": "gen_ai.conversation_root"},
    )
    data_source_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_DATA_SOURCE_ID},
    )
    # Association properties for context tracking.
    # Emitted on spans as gen_ai.association.properties.<key>.
    association_properties: Dict[str, Any] = field(default_factory=dict)
    sample_for_evaluation: Optional[bool] = field(default=True)
    evaluation_error: Optional[str] = None

    # Back-reference to the handler that started this invocation.
    # Set automatically by TelemetryHandler.start_*() methods.
    _handler: Optional["TelemetryHandler"] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        # Set safe defaults for GenAIInvocation internal fields.
        # The dataclass __init__ does not call super().__init__(), so
        # these fields would be missing on old-style instances.
        # Old-style types use _handler for lifecycle, not these.
        self._emitter = None
        self._agent_context_stack = []
        self._completion_callbacks = []
        self._sampler_fn = lambda trace_id: True
        self._meter_provider = None
        self._capture_refresh_fn = None
        self._otel_context_token = None
        self.parent_span = None
        self.error_type = None

    def stop(self) -> None:
        """Finalize the invocation successfully and end its span.

        Delegates to the handler's type-specific stop logic via
        ``_stop_invocation``.  This is the upstream-compatible API —
        callers use ``invocation.stop()`` instead of
        ``handler.stop_llm(invocation)``.
        """
        if self._handler is not None:
            self._handler._stop_invocation(self)

    def fail(self, error: "Error | BaseException") -> None:
        """Fail the invocation and end its span with error status.

        If *error* is a raw exception, it is wrapped in an ``Error``
        dataclass automatically.  Delegates to the handler's
        type-specific fail logic via ``_fail_invocation``.
        """
        if isinstance(error, BaseException):
            error = Error(message=str(error), type=type(error))
        if self._handler is not None:
            self._handler._fail_invocation(self, error)

    def semantic_convention_attributes(self) -> dict[str, Any]:
        """Return semantic convention attributes defined on this dataclass.

        Includes association properties emitted as
        ``gen_ai.association.properties.<key>``.
        """
        from opentelemetry.util.genai.attributes import (
            GEN_AI_ASSOCIATION_PROPERTIES_PREFIX,
        )

        result: dict[str, Any] = {}
        for data_field in dataclass_fields(self):
            semconv_key = data_field.metadata.get("semconv")
            if not semconv_key:
                continue
            value = getattr(self, data_field.name)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            result[semconv_key] = value

        # Emit association properties with prefix
        for key, value in self.association_properties.items():
            result[f"{GEN_AI_ASSOCIATION_PROPERTIES_PREFIX}.{key}"] = value

        return result


@dataclass()
class ToolCall(GenAI):
    """Represents a tool call invocation per execute_tool semantic conventions.

    See: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md#execute-tool-span

    Required attributes:
    - gen_ai.operation.name: Should be "execute_tool"

    Conditionally required:
    - error.type: If operation ended in error

    Recommended:
    - gen_ai.tool.name: Name of the tool
    - gen_ai.tool.call.id: Tool call identifier
    - gen_ai.tool.type: Type (function, extension, datastore)
    - gen_ai.tool.description: Tool description

    Opt-In:
    - gen_ai.tool.call.arguments: Parameters passed to tool
    - gen_ai.tool.call.result: Result returned by tool
    """

    # Required: gen_ai.tool.name
    name: str = field(metadata={"semconv": "gen_ai.tool.name"})
    # Opt-In: gen_ai.tool.call.arguments (set on start if content capture enabled)
    arguments: Any = field(
        default=None,
        metadata={"semconv_content": "gen_ai.tool.call.arguments"},
    )
    # Recommended: gen_ai.tool.call.id
    id: Optional[str] = field(
        default=None, metadata={"semconv": "gen_ai.tool.call.id"}
    )
    type: Literal["tool_call"] = "tool_call"

    # Recommended: gen_ai.tool.type ("function", "extension", or "datastore")
    tool_type: Optional[str] = field(
        default=None,
        metadata={"semconv": "gen_ai.tool.type"},
    )
    # Recommended: gen_ai.tool.description
    tool_description: Optional[str] = field(
        default=None,
        metadata={"semconv": "gen_ai.tool.description"},
    )
    # Opt-In: gen_ai.tool.call.result (set on finish if content capture enabled)
    tool_result: Optional[Any] = field(
        default=None,
        metadata={"semconv_content": "gen_ai.tool.call.result"},
    )
    # Conditionally Required: error.type (set if error occurred)
    error_type: Optional[str] = field(
        default=None,
        metadata={"semconv": "error.type"},
    )

    # Internal: new-style invocation delegate
    _tool_invocation: Any = field(
        default=None, init=False, repr=False, compare=False
    )

    def _start_with_handler(self, components: dict) -> None:
        """Create and start a ToolInvocation from this data container."""
        from opentelemetry.util.genai._tool_invocation import ToolInvocation

        self._tool_invocation = ToolInvocation(
            **components,
            provider=self.provider,
            framework=self.framework,
            system=self.system,
            name=self.name,
            arguments=self.arguments,
            id=self.id,
            tool_type=self.tool_type,
            tool_description=self.tool_description,
            tool_result=self.tool_result,
            error_type=self.error_type,
            attributes=dict(self.attributes),
        )
        self.span = self._tool_invocation.span
        self.span_context = self._tool_invocation.span_context
        self.trace_id = self._tool_invocation.trace_id
        self.span_id = self._tool_invocation.span_id
        self.start_time = self._tool_invocation.start_time

    def _sync_to_invocation(self) -> None:
        """Sync mutable fields from wrapper to underlying ToolInvocation."""
        inv = self._tool_invocation
        if inv is None:
            return
        inv.name = self.name
        inv.arguments = self.arguments
        inv.id = self.id
        inv.tool_type = self.tool_type
        inv.tool_description = self.tool_description
        inv.tool_result = self.tool_result
        inv.error_type = self.error_type
        inv.attributes = dict(self.attributes) if self.attributes else {}


@dataclass()
class MCPOperation(GenAI):
    """Represents any MCP protocol operation (non-tool-call).

    Covers tools/list, resources/read, resources/list, prompts/get,
    prompts/list, initialize, ping, etc.
    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp

    Span name: ``{mcp.method.name} {target}`` (target omitted when empty).
    SpanKind: CLIENT if ``is_client`` else SERVER.
    """

    target: str = ""

    # --- Required ---
    mcp_method_name: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.method.name"},
    )

    # --- Conditionally required ---
    jsonrpc_request_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "jsonrpc.request.id"},
    )
    mcp_resource_uri: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.resource.uri"},
    )
    gen_ai_prompt_name: Optional[str] = field(
        default=None,
        metadata={"semconv": "gen_ai.prompt.name"},
    )
    rpc_response_status_code: Optional[str] = field(
        default=None,
        metadata={"semconv": "rpc.response.status_code"},
    )

    # --- Recommended ---
    mcp_protocol_version: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.protocol.version"},
    )
    mcp_session_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.session.id"},
    )
    network_transport: Optional[str] = field(
        default=None,
        metadata={"semconv": "network.transport"},
    )
    network_protocol_name: Optional[str] = field(
        default=None,
        metadata={"semconv": "network.protocol.name"},
    )
    network_protocol_version: Optional[str] = field(
        default=None,
        metadata={"semconv": "network.protocol.version"},
    )
    server_address: Optional[str] = field(
        default=None,
        metadata={"semconv": "server.address"},
    )
    server_port: Optional[int] = field(
        default=None,
        metadata={"semconv": "server.port"},
    )
    client_address: Optional[str] = field(
        default=None,
        metadata={"semconv": "client.address"},
    )
    client_port: Optional[int] = field(
        default=None,
        metadata={"semconv": "client.port"},
    )

    # --- SDOT custom (not in OTel semconv) ---
    sdot_mcp_server_name: Optional[str] = field(
        default=None,
        metadata={"semconv": "sdot.mcp.server_name"},
    )

    # --- Internal (no semconv) ---
    is_client: bool = True
    duration_s: Optional[float] = None
    is_error: bool = False
    mcp_error_type: Optional[str] = None


@dataclass()
class MCPToolCall(MCPOperation, ToolCall):
    """MCP tool call operation (``tools/call``).

    Inherits MCP transport/protocol attributes from :class:`MCPOperation`
    and tool content fields (arguments, result, etc.) from :class:`ToolCall`.

    MRO: MCPToolCall -> MCPOperation -> ToolCall -> GenAI
    """

    output_size_bytes: Optional[int] = None
    output_size_tokens: Optional[int] = None


@dataclass()
class ToolCallResponse:
    response: Any
    id: Optional[str]
    type: Literal["tool_call_response"] = "tool_call_response"


FinishReason = Literal[
    "content_filter", "error", "length", "stop", "tool_calls"
]


@dataclass()
class Text:
    content: str
    type: Literal["text"] = "text"


MessagePart = Union[Text, "ToolCall", ToolCallResponse, Any]


@dataclass()
class InputMessage:
    role: str
    parts: list[MessagePart]


@dataclass()
class OutputMessage:
    role: str
    parts: list[MessagePart]
    finish_reason: Optional[Union[str, FinishReason]] = (
        None  # Only for LLM responses
    )


@dataclass
class LLMInvocation(GenAI):
    """Represents a single large language model invocation.

    Only fields tagged with ``metadata["semconv"]`` are emitted as
    semantic-convention attributes by the span emitters. Additional fields are
    util-only helpers or inputs to alternative span flavors (e.g. Traceloop).
    """

    request_model: str = field(
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL}
    )
    server_address: Optional[str] = field(
        default=None,
        metadata={"semconv": ServerAttributes.SERVER_ADDRESS},
    )
    server_port: Optional[int] = field(
        default=None,
        metadata={"semconv": ServerAttributes.SERVER_PORT},
    )
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    # Traceloop compatibility relies on enumerating these lists into prefixed attributes.
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )
    operation: str = field(
        default=GenAIAttributes.GenAiOperationNameValues.CHAT.value,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    response_model_name: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_RESPONSE_MODEL},
    )
    response_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_RESPONSE_ID},
    )
    input_tokens: Optional[AttributeValue] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS},
    )
    output_tokens: Optional[AttributeValue] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS},
    )
    # Structured function/tool definitions for semantic convention emission
    request_functions: list[dict[str, Any]] = field(default_factory=list)
    # Opt-In: gen_ai.tool.definitions (JSON-serialized tool schemas)
    tool_definitions: Optional[str] = field(
        default=None,
        metadata={"semconv_content": GEN_AI_TOOL_DEFINITIONS},
    )
    request_temperature: Optional[float] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE},
    )
    request_top_p: Optional[float] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_TOP_P},
    )
    request_top_k: Optional[int] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_TOP_K},
    )
    request_frequency_penalty: Optional[float] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY},
    )
    request_presence_penalty: Optional[float] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY},
    )
    request_stop_sequences: List[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES},
    )
    request_max_tokens: Optional[int] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS},
    )
    request_choice_count: Optional[int] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT},
    )
    request_seed: Optional[int] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_SEED},
    )
    request_encoding_formats: List[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS},
    )
    output_type: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_OUTPUT_TYPE},
    )
    response_finish_reasons: List[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS},
    )
    request_service_tier: Optional[str] = field(
        default=None,
        metadata={
            "semconv": GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER
        },
    )
    response_service_tier: Optional[str] = field(
        default=None,
        metadata={
            "semconv": GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
        },
    )
    response_system_fingerprint: Optional[str] = field(
        default=None,
        metadata={
            "semconv": GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        },
    )
    # Security inspection attribute (Cisco AI Defense)
    security_event_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GEN_AI_SECURITY_EVENT_ID},
    )
    # Streaming attribute (custom, not in semconv yet)
    request_stream: Optional[bool] = field(
        default=None,
        metadata={"semconv": GEN_AI_REQUEST_STREAM},
    )
    # Note: gen_ai.response.time_to_first_chunk is captured as an attribute
    # directly in the instrumentation (e.g., callback_handler.py) rather than
    # as a dedicated field, since it's computed dynamically during streaming.

    # Internal: new-style InferenceInvocation delegate (set by _start_with_handler)
    _inference_invocation: Any = field(
        default=None, init=False, repr=False, compare=False
    )

    def _start_with_handler(self, components: dict) -> None:
        """Create and start an InferenceInvocation from this data container.

        Called by ``handler.start_llm()`` to delegate to the new invocation.
        """
        from opentelemetry.util.genai._inference_invocation import (
            InferenceInvocation,
        )

        self._inference_invocation = InferenceInvocation(
            **components,
            provider=self.provider,
            framework=self.framework,
            system=self.system,
            request_model=self.request_model,
            server_address=self.server_address,
            server_port=self.server_port,
            input_messages=self.input_messages,
            output_messages=self.output_messages,
            operation=self.operation,
            response_model_name=self.response_model_name,
            response_id=self.response_id,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            request_functions=self.request_functions,
            tool_definitions=self.tool_definitions,
            request_temperature=self.request_temperature,
            request_top_p=self.request_top_p,
            request_top_k=self.request_top_k,
            request_frequency_penalty=self.request_frequency_penalty,
            request_presence_penalty=self.request_presence_penalty,
            request_stop_sequences=list(self.request_stop_sequences),
            request_max_tokens=self.request_max_tokens,
            request_choice_count=self.request_choice_count,
            request_seed=self.request_seed,
            request_encoding_formats=list(self.request_encoding_formats),
            output_type=self.output_type,
            response_finish_reasons=list(self.response_finish_reasons),
            request_service_tier=self.request_service_tier,
            response_service_tier=self.response_service_tier,
            response_system_fingerprint=self.response_system_fingerprint,
            security_event_id=self.security_event_id,
            request_stream=self.request_stream,
            attributes=dict(self.attributes),
        )
        # Sync back span/context info
        self.span = self._inference_invocation.span
        self.span_context = self._inference_invocation.span_context
        self.trace_id = self._inference_invocation.trace_id
        self.span_id = self._inference_invocation.span_id
        self.start_time = self._inference_invocation.start_time

    def _sync_to_invocation(self) -> None:
        """Sync mutable fields from wrapper to underlying InferenceInvocation."""
        inv = self._inference_invocation
        if inv is None:
            return
        inv.provider = self.provider
        inv.request_model = self.request_model
        inv.input_messages = self.input_messages
        inv.output_messages = self.output_messages
        inv.response_model_name = self.response_model_name
        inv.response_id = self.response_id
        inv.input_tokens = self.input_tokens
        inv.output_tokens = self.output_tokens
        inv.response_finish_reasons = (
            list(self.response_finish_reasons)
            if self.response_finish_reasons
            else []
        )
        inv.request_temperature = self.request_temperature
        inv.request_top_p = self.request_top_p
        inv.request_frequency_penalty = self.request_frequency_penalty
        inv.request_presence_penalty = self.request_presence_penalty
        inv.request_max_tokens = self.request_max_tokens
        inv.request_stop_sequences = (
            list(self.request_stop_sequences)
            if self.request_stop_sequences
            else []
        )
        inv.request_seed = self.request_seed
        inv.server_address = self.server_address
        inv.server_port = self.server_port
        inv.attributes = dict(self.attributes) if self.attributes else {}
        inv.security_event_id = self.security_event_id
        inv.request_stream = self.request_stream


# ErrorClassification, Error, and EvaluationResult are imported from ._error
# and re-exported for backward compatibility (see imports at top of file).


@dataclass
class EmbeddingInvocation(GenAI):
    """Represents a single embedding model invocation."""

    operation_name: str = field(
        default=GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    request_model: str = field(
        default="",
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL},
    )
    input_texts: list[str] = field(default_factory=list)
    dimension_count: Optional[int] = None
    server_port: Optional[int] = None
    server_address: Optional[str] = None
    input_tokens: Optional[int] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS},
    )
    encoding_formats: list[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS},
    )
    error_type: Optional[str] = None

    # Internal: new-style invocation delegate
    _real_invocation: Any = field(
        default=None, init=False, repr=False, compare=False
    )

    def _start_with_handler(self, components: dict) -> None:
        """Create and start a new EmbeddingInvocation from this data container."""
        from opentelemetry.util.genai._embedding_invocation import (
            EmbeddingInvocation as NewEmbedding,
        )

        self._real_invocation = NewEmbedding(
            **components,
            provider=self.provider,
            framework=self.framework,
            system=self.system,
            operation_name=self.operation_name,
            request_model=self.request_model,
            server_address=self.server_address,
            server_port=self.server_port,
            input_texts=list(self.input_texts),
            dimension_count=self.dimension_count,
            input_tokens=self.input_tokens,
            encoding_formats=list(self.encoding_formats),
            error_type=self.error_type,
            attributes=dict(self.attributes),
        )
        self.span = self._real_invocation.span
        self.span_context = self._real_invocation.span_context
        self.trace_id = self._real_invocation.trace_id
        self.span_id = self._real_invocation.span_id
        self.start_time = self._real_invocation.start_time

    def _sync_to_invocation(self) -> None:
        """Sync mutable fields from wrapper to underlying invocation."""
        inv = self._real_invocation
        if inv is None:
            return
        inv.provider = self.provider
        inv.request_model = self.request_model
        inv.input_texts = list(self.input_texts) if self.input_texts else []
        inv.dimension_count = self.dimension_count
        inv.input_tokens = self.input_tokens
        inv.encoding_formats = (
            list(self.encoding_formats) if self.encoding_formats else []
        )
        inv.error_type = self.error_type
        inv.attributes = dict(self.attributes) if self.attributes else {}


@dataclass
class RetrievalInvocation(GenAI):
    """Represents a single retrieval/search invocation."""

    # Required attribute
    operation_name: str = field(
        default="retrieval",
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )

    # Recommended attributes
    retriever_type: Optional[str] = field(
        default=None,
        metadata={"semconv": "gen_ai.retrieval.type"},
    )
    request_model: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL},
    )
    query: Optional[str] = field(
        default=None,
        metadata={"semconv": "gen_ai.retrieval.query.text"},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"semconv": "gen_ai.retrieval.top_k"},
    )
    documents_retrieved: Optional[int] = field(
        default=None,
        metadata={"semconv": "gen_ai.retrieval.documents_retrieved"},
    )

    # Opt-in attribute
    results: list[dict[str, Any]] = field(
        default_factory=list,
        metadata={"semconv": "gen_ai.retrieval.documents"},
    )

    # Additional utility fields (not in semantic conventions)
    query_vector: Optional[list[float]] = None
    server_port: Optional[int] = None
    server_address: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class Workflow(GenAI):
    """Represents a workflow orchestrating multiple agents and steps.

    A workflow is the top-level orchestration unit in agentic AI systems,
    coordinating agents and steps to achieve a complex goal. Workflows are optional
    and typically used in multi-agent or multi-step scenarios.

    Attributes:
        name: Identifier for the workflow (e.g., "customer_support_pipeline")
        workflow_type: Type of orchestration (e.g., "sequential", "parallel", "graph", "dynamic")
        description: Human-readable description of the workflow's purpose
        framework: Framework implementing the workflow (e.g., "langgraph", "crewai", "autogen")
        input_messages: Structured input messages for the workflow
        output_messages: Structured output messages from the workflow
        attributes: Additional custom attributes for workflow-specific metadata
        start_time: Timestamp when workflow started
        end_time: Timestamp when workflow completed
        span: OpenTelemetry span associated with this workflow
        context_token: Context token for span management
    """

    name: str
    workflow_type: Optional[str] = None  # sequential, parallel, graph, dynamic
    description: Optional[str] = None
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )

    # Internal: new-style invocation delegate
    _workflow_invocation: Any = field(
        default=None, init=False, repr=False, compare=False
    )

    def _start_with_handler(self, components: dict) -> None:
        """Create and start a WorkflowInvocation from this data container."""
        from opentelemetry.util.genai._workflow_invocation import (
            WorkflowInvocation,
        )

        self._workflow_invocation = WorkflowInvocation(
            **components,
            provider=self.provider,
            framework=self.framework,
            system=self.system,
            name=self.name,
            workflow_type=self.workflow_type,
            description=self.description,
            input_messages=list(self.input_messages),
            output_messages=list(self.output_messages),
            attributes=dict(self.attributes),
        )
        self.span = self._workflow_invocation.span
        self.span_context = self._workflow_invocation.span_context
        self.trace_id = self._workflow_invocation.trace_id
        self.span_id = self._workflow_invocation.span_id
        self.start_time = self._workflow_invocation.start_time

    def _sync_to_invocation(self) -> None:
        """Sync mutable fields from wrapper to underlying WorkflowInvocation."""
        inv = self._workflow_invocation
        if inv is None:
            return
        inv.name = self.name
        inv.workflow_type = self.workflow_type
        inv.description = self.description
        inv.input_messages = (
            list(self.input_messages) if self.input_messages else []
        )
        inv.output_messages = (
            list(self.output_messages) if self.output_messages else []
        )
        inv.attributes = dict(self.attributes) if self.attributes else {}


@dataclass
class _BaseAgent(GenAI):
    """Shared fields for agent lifecycle phases."""

    name: str
    agent_type: Optional[str] = (
        None  # researcher, planner, executor, critic, etc.
    )
    description: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_AGENT_DESCRIPTION},
    )
    model: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL},
    )  # primary model if applicable
    tools: list[str] = field(default_factory=list)  # available tool names
    system_instructions: Optional[str] = None  # System prompt/instructions


@dataclass
class AgentCreation(_BaseAgent):
    """Represents agent creation/initialisation."""

    operation: Literal["create_agent"] = field(
        init=False,
        default="create_agent",
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )


@dataclass
class AgentInvocation(_BaseAgent):
    """Represents agent execution (`invoke_agent`)."""

    operation: Literal["invoke_agent"] = field(
        init=False,
        default="invoke_agent",
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )


@dataclass
class Step(GenAI):
    """Represents a discrete unit of work in an agentic AI system.

    Steps can be orchestrated at the workflow level (assigned to agents) or
    decomposed internally by agents during execution. This design supports both
    scenarios through flexible parent relationships.
    """

    name: str
    objective: Optional[str] = None  # what the step aims to achieve
    step_type: Optional[str] = (
        None  # planning, execution, reflection, tool_use, etc.
    )
    source: Optional[Literal["workflow", "agent"]] = (
        None  # where step originated
    )
    assigned_agent: Optional[str] = None  # for workflow-assigned steps
    status: Optional[str] = None  # pending, in_progress, completed, failed
    description: Optional[str] = None


__all__ = [
    # existing exports intentionally implicit before; making explicit for new additions
    "ContentCapturingMode",
    "ToolCall",
    "ToolCallResponse",
    "Text",
    "InputMessage",
    "OutputMessage",
    "GenAI",
    "LLMInvocation",
    "EmbeddingInvocation",
    "RetrievalInvocation",
    "Error",
    "EvaluationResult",
    # agentic AI types
    "Workflow",
    "AgentCreation",
    "AgentInvocation",
    "Step",
    # MCP types
    "MCPOperation",
    "MCPToolCall",
    # Security semconv constant (Cisco AI Defense) - re-exported from attributes
    "GEN_AI_SECURITY_EVENT_ID",
]
