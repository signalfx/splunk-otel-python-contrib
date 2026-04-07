# pylint: disable=wrong-import-position,wrong-import-order,import-error,no-name-in-module,unexpected-keyword-arg,no-value-for-parameter,redefined-outer-name

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

TESTS_ROOT = Path(__file__).resolve().parent
stub_path = TESTS_ROOT / "stubs"
if str(stub_path) not in sys.path:
    sys.path.insert(0, str(stub_path))

import agents.tracing as agents_tracing  # noqa: E402
from agents.tracing import (  # noqa: E402
    agent_span,
    function_span,
    generation_span,
    response_span,
    set_trace_processors,
    trace,
)

from opentelemetry.instrumentation.openai_agents import (  # noqa: E402
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.openai_agents.span_processor import (  # noqa: E402
    ContentPayload,
    GenAISemanticProcessor,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402

try:
    from opentelemetry.sdk.trace.export import (  # type: ignore[attr-defined]
        InMemorySpanExporter,
        SimpleSpanProcessor,
    )
except ImportError:  # pragma: no cover - support older/newer SDK layouts
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,  # noqa: E402
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
        InMemorySpanExporter,
    )
from opentelemetry.semconv._incubating.attributes import (  # noqa: E402
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind  # noqa: E402

GEN_AI_INPUT_MESSAGES = getattr(
    GenAI, "GEN_AI_INPUT_MESSAGES", "gen_ai.input.messages"
)
GEN_AI_OUTPUT_MESSAGES = getattr(
    GenAI, "GEN_AI_OUTPUT_MESSAGES", "gen_ai.output.messages"
)


def _instrument_with_provider(**instrument_kwargs):
    # Reset singleton so each test gets a fresh handler/exporter pipeline.
    from opentelemetry.util.genai.handler import TelemetryHandler

    TelemetryHandler._reset_for_testing()

    set_trace_processors([])
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, **instrument_kwargs)

    return instrumentor, exporter


def test_generation_span_creates_client_span():
    instrumentor, exporter = _instrument_with_provider()

    try:
        with trace("workflow"):
            with generation_span(
                input=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                model_config={
                    "temperature": 0.2,
                    "base_url": "https://api.openai.com",
                },
                usage={"input_tokens": 12, "output_tokens": 3},
            ):
                pass

        spans = exporter.get_finished_spans()
        client_span = next(
            span for span in spans if span.kind is SpanKind.CLIENT
        )

        assert (
            client_span.attributes[GenAI.GEN_AI_OPERATION_NAME]
            == GenAI.GenAiOperationNameValues.CHAT.value
        )
        assert (
            client_span.attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
        )
        assert client_span.name == "chat gpt-4o-mini"
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def test_generation_span_without_roles_uses_text_completion():
    instrumentor, exporter = _instrument_with_provider()

    try:
        with trace("workflow"):
            with generation_span(
                input=[{"content": "tell me a joke"}],
                model="gpt-4o-mini",
                model_config={"temperature": 0.7},
            ):
                pass

        spans = exporter.get_finished_spans()
        chat_span = next(
            span
            for span in spans
            if span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
            == GenAI.GenAiOperationNameValues.CHAT.value
        )

        assert chat_span.kind is SpanKind.CLIENT
        assert (
            chat_span.attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
        )
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def test_function_span_records_tool_attributes():
    instrumentor, exporter = _instrument_with_provider()

    try:
        with trace("workflow"):
            with function_span(
                name="fetch_weather", input='{"city": "Paris"}'
            ):
                pass

        spans = exporter.get_finished_spans()
        tool_span = next(
            span for span in spans if span.kind is SpanKind.INTERNAL
        )

        assert (
            tool_span.attributes[GenAI.GEN_AI_OPERATION_NAME]
            == GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
        )
        assert tool_span.attributes[GenAI.GEN_AI_TOOL_NAME] == "fetch_weather"
        assert tool_span.attributes[GenAI.GEN_AI_TOOL_TYPE] == "function"
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def test_agent_invoke_span_records_attributes():
    instrumentor, exporter = _instrument_with_provider()

    try:
        with trace("workflow"):
            with agent_span(name="support_bot"):
                pass

        spans = exporter.get_finished_spans()
        agent_span_record = next(
            span
            for span in spans
            if span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
            == GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
        )

        assert agent_span_record.kind is SpanKind.CLIENT
        assert agent_span_record.name == "invoke_agent support_bot"
        assert (
            agent_span_record.attributes[GenAI.GEN_AI_AGENT_NAME]
            == "support_bot"
        )
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def _make_processor(**kwargs):
    """Create a GenAISemanticProcessor with a fresh handler for unit tests."""
    from opentelemetry.util.genai.handler import (
        TelemetryHandler,
        get_telemetry_handler,
    )

    TelemetryHandler._reset_for_testing()
    handler = get_telemetry_handler(tracer_provider=TracerProvider())
    return GenAISemanticProcessor(handler=handler, **kwargs)


def _placeholder_message() -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [{"type": "text", "content": "readacted"}],
    }


def test_normalize_messages_skips_empty_when_sensitive_enabled():
    processor = _make_processor()
    normalized = processor._normalize_messages_to_role_parts(
        [{"role": "user", "content": None}]
    )
    assert not normalized


def test_normalize_messages_emits_placeholder_when_sensitive_disabled():
    processor = _make_processor(include_sensitive_data=False)
    normalized = processor._normalize_messages_to_role_parts(
        [{"role": "user", "content": None}]
    )
    assert normalized == [_placeholder_message()]


def _setup_agent_state(processor, agent_id="agent-span"):
    """Register a mock AgentInvocation state in processor._invocations."""
    from opentelemetry.instrumentation.openai_agents.span_processor import (
        _InvocationState,
    )
    from opentelemetry.util.genai.types import AgentInvocation

    state = _InvocationState(
        invocation=AgentInvocation(name="test-agent"),
    )
    processor._invocations[agent_id] = state
    return state


def test_agent_content_aggregation_skips_duplicate_snapshots():
    processor = _make_processor()
    agent_id = "agent-span"
    state = _setup_agent_state(processor, agent_id)

    payload = ContentPayload(
        input_messages=[
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
            {
                "role": "user",
                "parts": [{"type": "text", "content": "readacted"}],
            },
        ]
    )

    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-1", parent_id=agent_id, span_data=None),
        payload,
    )
    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-2", parent_id=agent_id, span_data=None),
        payload,
    )

    aggregated = state.input_messages
    assert aggregated == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    # ensure data copied rather than reused to prevent accidental mutation
    assert aggregated is not payload.input_messages


def test_agent_content_aggregation_filters_placeholder_append_when_sensitive():
    processor = _make_processor()
    agent_id = "agent-span"
    state = _setup_agent_state(processor, agent_id)

    initial_payload = ContentPayload(
        input_messages=[
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
        ]
    )
    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-1", parent_id=agent_id, span_data=None),
        initial_payload,
    )

    placeholder_payload = ContentPayload(
        input_messages=[_placeholder_message()]
    )
    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-2", parent_id=agent_id, span_data=None),
        placeholder_payload,
    )

    aggregated = state.input_messages
    assert aggregated == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]


def test_agent_content_aggregation_retains_placeholder_when_sensitive_disabled():
    processor = _make_processor(include_sensitive_data=False)
    agent_id = "agent-span"
    state = _setup_agent_state(processor, agent_id)

    placeholder_payload = ContentPayload(
        input_messages=[_placeholder_message()]
    )
    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-1", parent_id=agent_id, span_data=None),
        placeholder_payload,
    )

    aggregated = state.input_messages
    assert aggregated == [_placeholder_message()]


def test_agent_content_aggregation_appends_new_messages_once():
    processor = _make_processor()
    agent_id = "agent-span"
    state = _setup_agent_state(processor, agent_id)

    initial_payload = ContentPayload(
        input_messages=[
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
        ]
    )
    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-1", parent_id=agent_id, span_data=None),
        initial_payload,
    )

    extended_messages = [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "hi there"}],
        },
    ]
    extended_payload = ContentPayload(input_messages=extended_messages)
    processor._update_agent_aggregate(
        SimpleNamespace(span_id="child-2", parent_id=agent_id, span_data=None),
        extended_payload,
    )

    aggregated = state.input_messages
    assert aggregated == extended_messages
    assert extended_payload.input_messages == extended_messages


def test_agent_span_collects_child_messages():
    capture_content_env = {
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    }
    with patch.dict(os.environ, capture_content_env):
        instrumentor, exporter = _instrument_with_provider(
            capture_message_content=True,
        )

        try:
            provider = agents_tracing.get_trace_provider()

            with trace("workflow") as workflow:
                agent_span_obj = provider.create_span(
                    agents_tracing.AgentSpanData(
                        name="helper",
                        input=[{"role": "user", "content": "hi"}],
                    ),
                    parent=workflow,
                )
                agent_span_obj.start()

                generation = agents_tracing.GenerationSpanData(
                    input=[{"role": "user", "content": "hi"}],
                    output=[{"type": "text", "content": "hello"}],
                    model="gpt-4o-mini",
                )
                gen_span = provider.create_span(
                    generation, parent=agent_span_obj
                )
                gen_span.start()
                gen_span.finish()

                agent_span_obj.finish()

            spans = exporter.get_finished_spans()
            agent_span_found = next(
                span
                for span in spans
                if span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
                == GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
            )

            assert agent_span_found.name == "invoke_agent helper"

            prompt = json.loads(
                agent_span_found.attributes[GEN_AI_INPUT_MESSAGES]
            )
            assert prompt == [
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": "hi"}],
                }
            ]

            completion = json.loads(
                agent_span_found.attributes[GEN_AI_OUTPUT_MESSAGES]
            )
            assert completion == [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": "hello"}],
                }
            ]
        finally:
            instrumentor.uninstrument()
            exporter.clear()


def test_agent_name_override_applied_to_agent_spans():
    instrumentor, exporter = _instrument_with_provider(
        agent_name="Travel Concierge"
    )

    try:
        with trace("workflow"):
            with agent_span(name="support_bot"):
                pass
            with agent_span():
                pass

        spans = exporter.get_finished_spans()
        agent_spans = [
            span
            for span in spans
            if span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
            == GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
        ]
        assert len(agent_spans) == 2

        named_span = next(s for s in agent_spans if "support_bot" in s.name)
        assert named_span.kind is SpanKind.CLIENT
        assert named_span.name == "invoke_agent support_bot"
        assert named_span.attributes[GenAI.GEN_AI_AGENT_NAME] == "support_bot"

        default_span = next(
            s for s in agent_spans if "Travel Concierge" in s.name
        )
        assert default_span.kind is SpanKind.CLIENT
        assert default_span.name == "invoke_agent Travel Concierge"
        assert (
            default_span.attributes[GenAI.GEN_AI_AGENT_NAME]
            == "Travel Concierge"
        )
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def test_capture_mode_can_be_disabled():
    instrumentor, exporter = _instrument_with_provider(
        capture_message_content="no_content"
    )

    try:
        with trace("workflow"):
            with generation_span(
                input=[{"role": "user", "content": "hi"}],
                output=[{"role": "assistant", "content": "hello"}],
                model="gpt-4o-mini",
            ):
                pass

        spans = exporter.get_finished_spans()
        client_span = next(
            span for span in spans if span.kind is SpanKind.CLIENT
        )

        assert GEN_AI_INPUT_MESSAGES not in client_span.attributes
        assert GEN_AI_OUTPUT_MESSAGES not in client_span.attributes
        for event in client_span.events:
            assert GEN_AI_INPUT_MESSAGES not in event.attributes
            assert GEN_AI_OUTPUT_MESSAGES not in event.attributes
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def test_response_span_records_response_attributes():
    instrumentor, exporter = _instrument_with_provider()

    class _Usage:
        def __init__(self, input_tokens: int, output_tokens: int) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class _Response:
        def __init__(self) -> None:
            self.id = "resp-123"
            self.model = "gpt-4o-mini"
            self.usage = _Usage(42, 9)
            self.output = [{"finish_reason": "stop"}]

    try:
        with trace("workflow"):
            with response_span(response=_Response()):
                pass

        spans = exporter.get_finished_spans()
        response = next(
            span
            for span in spans
            if span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
            == GenAI.GenAiOperationNameValues.CHAT.value
        )

        assert response.kind is SpanKind.CLIENT
        assert response.name == "chat gpt-4o-mini"
    finally:
        instrumentor.uninstrument()
        exporter.clear()
