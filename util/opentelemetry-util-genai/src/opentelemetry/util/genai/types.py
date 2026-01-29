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


import time
from contextvars import Token
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import UUID, uuid4

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
from opentelemetry.util.genai.attributes import GEN_AI_SECURITY_EVENT_ID
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
class GenAI:
    """Base type for all GenAI telemetry entities."""

    context_token: Optional[ContextToken] = None
    span: Optional[Span] = None
    span_context: Optional[SpanContext] = None
    trace_id: Optional[int] = None
    span_id: Optional[int] = None
    trace_flags: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    provider: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_PROVIDER_NAME},
    )
    framework: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    run_id: UUID = field(default_factory=uuid4)
    parent_run_id: Optional[UUID] = None
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
    data_source_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_DATA_SOURCE_ID},
    )
    sample_for_evaluation: Optional[bool] = field(default=True)
    evaluation_error: Optional[str] = None

    def semantic_convention_attributes(self) -> dict[str, Any]:
        """Return semantic convention attributes defined on this dataclass."""

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


@dataclass()
class MCPToolCall(ToolCall):
    """Represents an MCP (Model Context Protocol) tool call invocation.

    Extends ToolCall with MCP-specific semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp

    MCP Semantic Convention Attributes:
    - mcp.method.name: The name of the request or notification method
    - mcp.session.id: Session identifier for the MCP connection
    - mcp.protocol.version: MCP protocol version
    - mcp.server.name: Name of the MCP server
    - network.transport: Transport type ("pipe" for stdio, "tcp" for HTTP)

    Metrics-only fields (not span attributes):
    - output_size_bytes: Output size for metrics tracking
    - output_size_tokens: Token count for metrics tracking
    - duration_s: Duration for standalone metrics emission
    """

    # mcp.method.name: The name of the request or notification method
    mcp_method_name: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.method.name"},
    )
    # network.transport: "pipe" for stdio, "tcp" for HTTP
    network_transport: Optional[str] = field(
        default=None,
        metadata={"semconv": "network.transport"},
    )
    # mcp.session.id: Session identifier
    mcp_session_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.session.id"},
    )
    # mcp.protocol.version: MCP protocol version
    mcp_protocol_version: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.protocol.version"},
    )
    # mcp.server.name: Name of the MCP server
    mcp_server_name: Optional[str] = field(
        default=None,
        metadata={"semconv": "mcp.server.name"},
    )
    # Metrics-only fields (no semconv metadata - not span attributes)
    output_size_bytes: Optional[int] = None
    output_size_tokens: Optional[int] = None
    duration_s: Optional[float] = None
    # Internal state tracking (no semconv)
    is_client: bool = True
    is_error: bool = False


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


@dataclass
class Error:
    message: str
    type: Type[BaseException]


@dataclass
class EvaluationResult:
    """Represents the outcome of a single evaluation metric.

    Additional fields (e.g., judge model, threshold) can be added without
    breaking callers that rely only on the current contract.
    """

    metric_name: str
    score: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[Error] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


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
        run_id: Unique identifier for this workflow execution
        parent_run_id: Optional parent workflow/trace identifier
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
    # Security semconv constant (Cisco AI Defense) - re-exported from attributes
    "GEN_AI_SECURITY_EVENT_ID",
]
