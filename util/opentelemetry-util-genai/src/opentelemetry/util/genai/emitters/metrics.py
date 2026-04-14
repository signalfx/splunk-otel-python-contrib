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
            if isinstance(obj, MCPToolCall):
                metric_attrs = _get_metric_attributes(
                    obj.name,
                    None,
                    GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                    obj.provider,
                    obj.framework,
                )
                if obj.agent_name:
                    metric_attrs[GenAI.GEN_AI_AGENT_NAME] = obj.agent_name
                if obj.agent_id:
                    metric_attrs[GenAI.GEN_AI_AGENT_ID] = obj.agent_id
                metric_attrs.update(get_context_metric_attributes(obj))
                _record_duration(
                    self._duration_histogram,
                    obj,
                    metric_attrs,
                    span=getattr(obj, "span", None),
                )
            return

        if isinstance(obj, ToolCall):
            tool_invocation = obj
            metric_attrs = _get_metric_attributes(
                tool_invocation.name,
                None,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                tool_invocation.provider,
                tool_invocation.framework,
            )
            if tool_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    tool_invocation.agent_name
                )
            if tool_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = tool_invocation.agent_id
            metric_attrs.update(get_context_metric_attributes(tool_invocation))
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

            _record_duration(
                self._duration_histogram,
                embedding_invocation,
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )

        if isinstance(obj, RetrievalInvocation):
            self._record_retrieval_metrics(obj)

    def on_error(self, error: Error, obj: Any) -> None:
        # Handle new agentic types
        if isinstance(obj, Workflow):
            self._record_workflow_metrics(obj)
            return
        if isinstance(obj, AgentInvocation):
            self._record_agent_metrics(obj)
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
            return

        if isinstance(obj, ToolCall):
            tool_invocation = obj
            metric_attrs = _get_metric_attributes(
                tool_invocation.name,
                None,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                tool_invocation.provider,
                tool_invocation.framework,
            )
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
                LLMInvocation,
                ToolCall,
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

    def _record_agent_metrics(self, agent: AgentInvocation) -> None:
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
