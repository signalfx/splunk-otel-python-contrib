"""CrewAI + OpenAI-v2 context inheritance tests.

These tests validate that when CrewAI and OpenAI-v2 share the singleton
TelemetryHandler, downstream chat spans/metrics emitted via the OpenAI-v2-style
LLM lifecycle inherit agent identity from the parent CrewAI agent span.
"""

import os
from unittest import mock

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import InputMessage, LLMInvocation, OutputMessage, Text

import opentelemetry.instrumentation.crewai.instrumentation as crewai_module


def test_openai_v2_chat_inherits_agent_context_to_spans_and_metrics(
    tracer_provider, meter_provider, metric_reader, span_exporter
):
    # Ensure fresh singleton with test providers.
    if hasattr(get_telemetry_handler, "_default_handler"):
        delattr(get_telemetry_handler, "_default_handler")

    handler = get_telemetry_handler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    crewai_module._handler = handler

    mock_agent = mock.MagicMock()
    mock_agent.role = "Crew Router Agent"
    mock_task = mock.MagicMock()
    mock_task.description = "Route support query"
    mock_task.expected_output = "Routing decision"

    # Simulates the core OpenAI-v2 instrumentation flow:
    # build LLMInvocation -> start_llm -> set outputs/tokens -> stop_llm
    def _agent_exec_side_effect(*_args, **_kwargs):
        llm = LLMInvocation(
            request_model="gpt-4o-mini",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="hello")])
            ],
            provider="openai",
            framework="openai-sdk",
            system="openai",
        )
        handler.start_llm(llm)
        llm.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="ok")],
                finish_reason="stop",
            )
        ]
        llm.response_model_name = "gpt-4o-mini-2024-07-18"
        llm.input_tokens = 12
        llm.output_tokens = 5
        handler.stop_llm(llm)
        return "agent-complete"

    wrapped = mock.MagicMock(side_effect=_agent_exec_side_effect)
    result = crewai_module._wrap_agent_execute_task(
        wrapped, mock_agent, (), {"task": mock_task}
    )
    assert result == "agent-complete"

    # Validate chat span inherited agent identity.
    spans = span_exporter.get_finished_spans()
    chat_span = next((span for span in spans if span.name == "chat gpt-4o-mini"), None)
    assert chat_span is not None
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "Crew Router Agent"
    )
    inherited_agent_id = chat_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_ID)
    assert isinstance(inherited_agent_id, str) and inherited_agent_id

    # Validate chat metrics inherited agent identity.
    try:
        meter_provider.force_flush()
    except Exception:
        pass
    metric_reader.collect()

    resource_metrics = metric_reader.get_metrics_data().resource_metrics
    assert resource_metrics
    metrics = resource_metrics[0].scope_metrics[0].metrics

    duration_metric = next(
        (
            metric
            for metric in metrics
            if metric.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
        ),
        None,
    )
    token_metric = next(
        (
            metric
            for metric in metrics
            if metric.name == gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE
        ),
        None,
    )

    assert duration_metric is not None
    assert token_metric is not None

    found_duration_with_agent = False
    for point in duration_metric.data.data_points:
        attrs = point.attributes
        if (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == GenAIAttributes.GenAiOperationNameValues.CHAT.value
            and attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME)
            == "Crew Router Agent"
            and attrs.get(GenAIAttributes.GEN_AI_AGENT_ID) == inherited_agent_id
        ):
            found_duration_with_agent = True
            break
    assert found_duration_with_agent

    found_input_with_agent = False
    found_output_with_agent = False
    for point in token_metric.data.data_points:
        attrs = point.attributes
        if (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            != GenAIAttributes.GenAiOperationNameValues.CHAT.value
        ):
            continue
        if (
            attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME)
            != "Crew Router Agent"
            or attrs.get(GenAIAttributes.GEN_AI_AGENT_ID) != inherited_agent_id
        ):
            continue
        token_type = attrs.get(GenAIAttributes.GEN_AI_TOKEN_TYPE)
        if token_type == GenAIAttributes.GenAiTokenTypeValues.INPUT.value:
            found_input_with_agent = True
        if token_type == GenAIAttributes.GenAiTokenTypeValues.COMPLETION.value:
            found_output_with_agent = True

    assert found_input_with_agent
    assert found_output_with_agent

    # Keep environment clean for neighboring tests that may inspect globals.
    crewai_module._handler = None
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
