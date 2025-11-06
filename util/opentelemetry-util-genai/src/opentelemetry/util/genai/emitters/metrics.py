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
    ToolCall,
    Workflow,
)
from .utils import (
    _get_metric_attributes,
    _record_duration,
    _record_token_metrics,
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
        self._workflow_duration_histogram: Histogram = (
            instruments.workflow_duration_histogram
        )
        self._agent_duration_histogram: Histogram = (
            instruments.agent_duration_histogram
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
            )
            # Add agent context if available
            if llm_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    llm_invocation.agent_name
                )
            if llm_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = llm_invocation.agent_id

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
            # Add agent context if available
            if tool_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = (
                    tool_invocation.agent_name
                )
            if tool_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = tool_invocation.agent_id

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
        if isinstance(obj, ToolCall):
            tool_invocation = obj
            metric_attrs = _get_metric_attributes(
                tool_invocation.name,
                None,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                tool_invocation.provider,
                tool_invocation.framework,
            )
            # Add agent context if available
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

    def handles(self, obj: Any) -> bool:
        return isinstance(
            obj,
            (
                LLMInvocation,
                ToolCall,
                Workflow,
                AgentInvocation,
                EmbeddingInvocation,
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
            GenAI.GEN_AI_AGENT_ID: str(agent.run_id),
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
