from __future__ import annotations

from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

from .._invocation import GenAIInvocation
from ..instruments import Instruments
from ..interfaces import EmitterMeta
from ..types import (
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    LLMInvocation,
    MCPOperation,
    MCPToolCall,
    RetrievalInvocation,
    ToolCall,
    Workflow,
)
from .utils import (
    _get_metric_attributes,
    _record_duration,
    _record_token_metrics,
    get_context_metric_attributes,
)


class MetricsEmitter(EmitterMeta):
    """Emits GenAI metrics (duration + token usage).

    Supports LLMInvocation, EmbeddingInvocation, ToolCall, Workflow, and Agent.
    """

    role = "metric"
    name = "semconv_metrics"

    def __init__(self, meter: Optional[Meter] = None):
        _meter: Meter = meter or get_meter(__name__)
        instruments = Instruments(_meter)
        self._duration_histogram: Histogram = (
            instruments.operation_duration_histogram
        )
        self._token_histogram: Histogram = instruments.token_usage_histogram
        self._time_to_first_chunk_histogram: Histogram = (
            instruments.time_to_first_chunk_histogram
        )
        self._workflow_duration_histogram: Histogram = (
            instruments.workflow_duration_histogram
        )
        self._agent_duration_histogram: Histogram = (
            instruments.agent_duration_histogram
        )
        self._retrieval_duration_histogram: Histogram = (
            instruments.retrieval_duration_histogram
        )
        # MCP-specific histograms
        self._mcp_client_operation_duration: Histogram = (
            instruments.mcp_client_operation_duration
        )
        self._mcp_server_operation_duration: Histogram = (
            instruments.mcp_server_operation_duration
        )
        self._mcp_client_session_duration: Histogram = (
            instruments.mcp_client_session_duration
        )
        self._mcp_server_session_duration: Histogram = (
            instruments.mcp_server_session_duration
        )
        self._mcp_tool_output_size: Histogram = (
            instruments.mcp_tool_output_size
        )

    def on_start(self, obj: Any) -> None:  # no-op for metrics
        return None

    def on_end(self, obj: Any) -> None:
        # GenAIInvocation subclasses with an emitter provide their own
        # metric attributes via hooks.
        if isinstance(obj, GenAIInvocation) and obj._emitter is not None:
            self._record_invocation_metrics(obj)
            return

        if isinstance(obj, Workflow):
            self._record_workflow_metrics(obj)
            return
        if isinstance(obj, AgentInvocation):
            self._record_agent_metrics(obj)
            return
        # Step metrics removed

        if isinstance(obj, LLMInvocation):
            llm_invocation = obj
            metric_attrs = _get_metric_attributes(
                llm_invocation.request_model,
                llm_invocation.response_model_name,
                llm_invocation.operation,
                llm_invocation.provider,
                llm_invocation.framework,
                server_address=llm_invocation.server_address,
                server_port=llm_invocation.server_port,
            )
            # Add agent context if available
            if llm_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    llm_invocation.agent_name
                )
            if llm_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = llm_invocation.agent_id

            # Add session context if configured
            metric_attrs.update(get_context_metric_attributes(llm_invocation))

            _record_token_metrics(
                self._token_histogram,
                llm_invocation.input_tokens,
                llm_invocation.output_tokens,
                metric_attrs,
                span=getattr(llm_invocation, "span", None),
            )
            _record_duration(
                self._duration_histogram,
                llm_invocation,
                metric_attrs,
                span=getattr(llm_invocation, "span", None),
            )
            # Record time to first chunk for streaming operations
            if llm_invocation.request_stream:
                ttfc = llm_invocation.attributes.get(
                    "gen_ai.response.time_to_first_chunk"
                )
                if ttfc is not None:
                    span = getattr(llm_invocation, "span", None)
                    context = None
                    if span is not None:
                        try:
                            context = trace.set_span_in_context(span)
                        except (
                            TypeError,
                            ValueError,
                            AttributeError,
                        ):  # pragma: no cover - defensive
                            context = None
                    self._time_to_first_chunk_histogram.record(
                        ttfc, attributes=metric_attrs, context=context
                    )
            return
        if isinstance(obj, MCPOperation):
            self._record_mcp_operation_metrics(obj)
            if isinstance(obj, MCPToolCall) and obj.is_client:
                self._record_execute_tool_metrics(obj)
            return

        if isinstance(obj, ToolCall):
            self._record_execute_tool_metrics(obj)

        if isinstance(obj, EmbeddingInvocation):
            embedding_invocation = obj
            metric_attrs = _get_metric_attributes(
                embedding_invocation.request_model,
                None,
                embedding_invocation.operation_name,
                embedding_invocation.provider,
                embedding_invocation.framework,
                server_address=embedding_invocation.server_address,
                server_port=embedding_invocation.server_port,
            )
            # Add agent context if available
            if embedding_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    embedding_invocation.agent_name
                )
            if embedding_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = (
                    embedding_invocation.agent_id
                )

            # Add session context if configured
            metric_attrs.update(
                get_context_metric_attributes(embedding_invocation)
            )

            _record_token_metrics(
                self._token_histogram,
                embedding_invocation.input_tokens,
                None,  # embeddings don't produce output tokens
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )
            _record_duration(
                self._duration_histogram,
                embedding_invocation,
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )

        if isinstance(obj, RetrievalInvocation):
            self._record_retrieval_metrics(obj)

    def on_error(self, error: Error, obj: Any) -> None:
        # GenAIInvocation subclasses with an emitter provide their own
        # metric attributes via hooks.
        if isinstance(obj, GenAIInvocation) and obj._emitter is not None:
            self._record_invocation_metrics(obj, error=error)
            return

        # Handle agentic types
        if isinstance(obj, Workflow):
            self._record_workflow_metrics(obj)
            return
        if isinstance(obj, AgentInvocation):
            self._record_agent_metrics(obj, error=error)
            return
        # Step metrics removed

        # Handle existing types with agent context
        if isinstance(obj, LLMInvocation):
            llm_invocation = obj
            metric_attrs = _get_metric_attributes(
                llm_invocation.request_model,
                llm_invocation.response_model_name,
                llm_invocation.operation,
                llm_invocation.provider,
                llm_invocation.framework,
                server_address=llm_invocation.server_address,
                server_port=llm_invocation.server_port,
            )
            # Add agent context if available
            if llm_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    llm_invocation.agent_name
                )
            if llm_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = llm_invocation.agent_id
            if getattr(error, "type", None) is not None:
                metric_attrs[ErrorAttributes.ERROR_TYPE] = (
                    error.type.__qualname__
                )

            _record_duration(
                self._duration_histogram, llm_invocation, metric_attrs
            )
            return

        if isinstance(obj, MCPOperation):
            obj.is_error = True
            if getattr(error, "type", None) is not None:
                obj.mcp_error_type = error.type.__qualname__
            self._record_mcp_operation_metrics(obj)
            if isinstance(obj, MCPToolCall) and obj.is_client:
                metric_attrs = _get_metric_attributes(
                    None,
                    None,
                    GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                    obj.provider,
                    obj.framework,
                )
                metric_attrs[GenAI.GEN_AI_TOOL_NAME] = obj.name
                if obj.agent_name:
                    metric_attrs[GenAI.GEN_AI_AGENT_NAME] = obj.agent_name
                if obj.agent_id:
                    metric_attrs[GenAI.GEN_AI_AGENT_ID] = obj.agent_id
                if getattr(error, "type", None) is not None:
                    metric_attrs[ErrorAttributes.ERROR_TYPE] = (
                        error.type.__qualname__
                    )
                metric_attrs.update(get_context_metric_attributes(obj))
                duration = obj.duration_s
                if duration is None and obj.end_time is not None:
                    duration = obj.end_time - obj.start_time
                if duration is not None:
                    context = None
                    span = getattr(obj, "span", None)
                    if span is not None:
                        try:
                            context = trace.set_span_in_context(span)
                        except (TypeError, ValueError, AttributeError):
                            context = None
                    self._duration_histogram.record(
                        duration, attributes=metric_attrs, context=context
                    )
            return

        if isinstance(obj, ToolCall):
            tool_invocation = obj
            metric_attrs = _get_metric_attributes(
                None,
                None,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                tool_invocation.provider,
                tool_invocation.framework,
            )
            metric_attrs[GenAI.GEN_AI_TOOL_NAME] = tool_invocation.name
            if tool_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    tool_invocation.agent_name
                )
            if tool_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = tool_invocation.agent_id
            if getattr(error, "type", None) is not None:
                metric_attrs[ErrorAttributes.ERROR_TYPE] = (
                    error.type.__qualname__
                )

            _record_duration(
                self._duration_histogram,
                tool_invocation,
                metric_attrs,
                span=getattr(tool_invocation, "span", None),
            )

        if isinstance(obj, EmbeddingInvocation):
            embedding_invocation = obj
            metric_attrs = _get_metric_attributes(
                embedding_invocation.request_model,
                None,
                embedding_invocation.operation_name,
                embedding_invocation.provider,
                embedding_invocation.framework,
                server_address=embedding_invocation.server_address,
                server_port=embedding_invocation.server_port,
            )
            # Add agent context if available
            if embedding_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    embedding_invocation.agent_name
                )
            if embedding_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = (
                    embedding_invocation.agent_id
                )
            if getattr(error, "type", None) is not None:
                metric_attrs[ErrorAttributes.ERROR_TYPE] = (
                    error.type.__qualname__
                )

            _record_token_metrics(
                self._token_histogram,
                embedding_invocation.input_tokens,
                None,  # embeddings don't produce output tokens
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )
            _record_duration(
                self._duration_histogram,
                embedding_invocation,
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )

        if isinstance(obj, RetrievalInvocation):
            self._record_retrieval_metrics(obj, error)

    def handles(self, obj: Any) -> bool:
        return isinstance(
            obj,
            (
                GenAIInvocation,
                LLMInvocation,
                ToolCall,
                MCPOperation,
                Workflow,
                AgentInvocation,
                EmbeddingInvocation,
                RetrievalInvocation,
            ),
        )

    # Helper methods for new agentic types
    def _record_workflow_metrics(self, workflow: Workflow) -> None:
        """Record metrics for a workflow."""
        if workflow.end_time is None:
            return
        duration = workflow.end_time - workflow.start_time
        metric_attrs = {
            "gen_ai.workflow.name": workflow.name,
        }
        if workflow.workflow_type:
            metric_attrs["gen_ai.workflow.type"] = workflow.workflow_type
        if workflow.framework:
            metric_attrs["gen_ai.framework"] = workflow.framework

        context = None
        span = getattr(workflow, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                context = None

        self._workflow_duration_histogram.record(
            duration, attributes=metric_attrs, context=context
        )

    def _record_agent_metrics(
        self, agent: AgentInvocation, error: Optional[Error] = None
    ) -> None:
        """Record metrics for an agent operation."""
        if agent.end_time is None:
            return
        duration = agent.end_time - agent.start_time
        metric_attrs = {
            GenAI.GEN_AI_OPERATION_NAME: agent.operation,
            GenAI.GEN_AI_AGENT_NAME: agent.name,
            GenAI.GEN_AI_AGENT_ID: (
                f"{agent.span_id:016x}"
                if agent.span_id is not None
                else str(id(agent))
            ),
        }
        if agent.agent_type:
            metric_attrs["gen_ai.agent.type"] = agent.agent_type
        if agent.framework:
            metric_attrs["gen_ai.framework"] = agent.framework

        context = None
        span = getattr(agent, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                context = None

        self._agent_duration_histogram.record(
            duration, attributes=metric_attrs, context=context
        )

        # Additionally record MCP session duration if this is an MCP session
        if agent.system == "mcp" and agent.agent_type in (
            "mcp_client",
            "mcp_server",
        ):
            self._record_mcp_session_metrics(agent, error=error)

    def _record_mcp_session_metrics(
        self, agent: AgentInvocation, error: Optional[Error] = None
    ) -> None:
        """Record mcp.client.session.duration or mcp.server.session.duration.

        Per OTel semconv: https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp

        Attributes (from agent.attributes):
        - error.type (conditionally required, if error)
        - network.transport (recommended)
        - mcp.protocol.version (recommended)
        - server.address (recommended)
        - server.port (recommended)
        """
        if agent.end_time is None:
            return
        duration = agent.end_time - agent.start_time

        # Build attributes per semconv
        mcp_attrs: dict[str, Any] = {}

        # Conditionally required: error.type
        error_type = agent.attributes.get("error.type")
        if error_type:
            mcp_attrs["error.type"] = error_type
        elif error is not None and getattr(error, "type", None) is not None:
            mcp_attrs["error.type"] = error.type.__qualname__

        # Recommended attributes
        for attr_key in (
            "network.transport",
            "mcp.protocol.version",
            "network.protocol.name",
            "network.protocol.version",
            "server.address",
            "server.port",
        ):
            val = agent.attributes.get(attr_key)
            if val is not None:
                mcp_attrs[attr_key] = val

        # Get span context for metric correlation
        context = None
        span = getattr(agent, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                context = None

        # Choose client or server histogram
        is_client = agent.agent_type == "mcp_client"
        histogram = (
            self._mcp_client_session_duration
            if is_client
            else self._mcp_server_session_duration
        )
        histogram.record(duration, attributes=mcp_attrs, context=context)

    def _record_retrieval_metrics(
        self, retrieval: RetrievalInvocation, error: Optional[Error] = None
    ) -> None:
        """Record metrics for a retrieval operation."""
        if retrieval.end_time is None:
            return
        duration = retrieval.end_time - retrieval.start_time
        metric_attrs = {
            GenAI.GEN_AI_OPERATION_NAME: retrieval.operation_name,
        }
        if retrieval.retriever_type:
            metric_attrs["gen_ai.retrieval.type"] = retrieval.retriever_type
        if retrieval.framework:
            metric_attrs["gen_ai.framework"] = retrieval.framework
        if retrieval.provider:
            metric_attrs[GenAI.GEN_AI_PROVIDER_NAME] = retrieval.provider
        # Add agent context if available
        if retrieval.agent_name:
            metric_attrs[GenAI.GEN_AI_AGENT_NAME] = retrieval.agent_name
        if retrieval.agent_id:
            metric_attrs[GenAI.GEN_AI_AGENT_ID] = retrieval.agent_id
        # Add error type if present
        if error is not None and getattr(error, "type", None) is not None:
            metric_attrs[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

        context = None
        span = getattr(retrieval, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                context = None

        self._retrieval_duration_histogram.record(
            duration, attributes=metric_attrs, context=context
        )

    def _record_execute_tool_metrics(self, tool: ToolCall) -> None:
        """Record ``gen_ai.client.operation.duration`` for an execute_tool."""
        metric_attrs = _get_metric_attributes(
            None,
            None,
            GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            tool.provider,
            tool.framework,
        )
        metric_attrs[GenAI.GEN_AI_TOOL_NAME] = tool.name
        if tool.agent_name:
            metric_attrs[GenAI.GEN_AI_AGENT_NAME] = tool.agent_name
        if tool.agent_id:
            metric_attrs[GenAI.GEN_AI_AGENT_ID] = tool.agent_id
        metric_attrs.update(get_context_metric_attributes(tool))
        _record_duration(
            self._duration_histogram,
            tool,
            metric_attrs,
            span=getattr(tool, "span", None),
        )

    def _record_mcp_operation_metrics(self, op: MCPOperation) -> None:
        """Record MCP-specific metrics for any MCP operation.

        Per OTel semconv: https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp

        Metrics:
        - mcp.client.operation.duration / mcp.server.operation.duration
        - mcp.tool.output.size (custom: tracks output bytes for LLM context)
        """
        if not op.mcp_method_name:
            return

        mcp_attrs: dict[str, Any] = {
            "mcp.method.name": op.mcp_method_name,
        }

        if isinstance(op, MCPToolCall):
            mcp_attrs[GenAI.GEN_AI_TOOL_NAME] = op.name
            if op.mcp_method_name == "tools/call":
                mcp_attrs[GenAI.GEN_AI_OPERATION_NAME] = (
                    GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
                )

        if op.network_transport:
            mcp_attrs["network.transport"] = op.network_transport
        if op.mcp_protocol_version:
            mcp_attrs["mcp.protocol.version"] = op.mcp_protocol_version
        if op.is_error and op.mcp_error_type:
            mcp_attrs["error.type"] = op.mcp_error_type
        elif op.is_error:
            mcp_attrs["error.type"] = "operation_error"

        context = None
        span = getattr(op, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):
                context = None

        duration = op.duration_s
        if duration is None and op.end_time is not None:
            duration = op.end_time - op.start_time
        if duration is not None:
            histogram = (
                self._mcp_client_operation_duration
                if op.is_client
                else self._mcp_server_operation_duration
            )
            histogram.record(duration, attributes=mcp_attrs, context=context)

        if (
            isinstance(op, MCPToolCall)
            and op.output_size_bytes is not None
            and op.output_size_bytes > 0
        ):
            self._mcp_tool_output_size.record(
                op.output_size_bytes, attributes=mcp_attrs, context=context
            )

    def _record_invocation_metrics(
        self, obj: GenAIInvocation, error: Optional[Error] = None
    ) -> None:
        """Record metrics for a GenAIInvocation using its hook methods."""
        if obj.end_time is None:
            return

        metric_attrs = obj._get_metric_attributes()

        # Add session context if configured
        metric_attrs.update(get_context_metric_attributes(obj))

        # Add error type if present
        if error is not None and getattr(error, "type", None) is not None:
            metric_attrs[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

        # Get span context for metric correlation
        span = getattr(obj, "span", None)
        context = None
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (TypeError, ValueError, AttributeError):
                context = None

        # Record token metrics from hook
        token_counts = obj._get_metric_token_counts()
        if token_counts:
            for token_type, count in token_counts.items():
                if isinstance(count, (int, float)):
                    token_attrs = {GenAI.GEN_AI_TOKEN_TYPE: token_type}
                    token_attrs.update(metric_attrs)
                    self._token_histogram.record(
                        count, attributes=token_attrs, context=context
                    )

        # Record duration
        duration = obj.end_time - obj.start_time
        self._duration_histogram.record(
            duration, attributes=metric_attrs, context=context
        )
