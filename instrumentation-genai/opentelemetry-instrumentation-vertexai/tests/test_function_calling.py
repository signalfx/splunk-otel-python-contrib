import json

import pytest
from tests.shared_test_utils import (
    ask_about_weather,
    ask_about_weather_function_response,
)

from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

# Backward compatibility for InMemoryLogExporter -> InMemoryLogRecordExporter rename
try:
    from opentelemetry.sdk._logs._internal.export.in_memory_log_exporter import (  # pylint: disable=no-name-in-module
        InMemoryLogRecordExporter,
    )
except ImportError:
    # Fallback to old name for compatibility with older SDK versions
    from opentelemetry.sdk._logs._internal.export.in_memory_log_exporter import (
        InMemoryLogExporter as InMemoryLogRecordExporter,
    )
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def assert_handler_event(log, parent_span):
    """Assert log record is the unified GenAI handler event and return its body."""
    assert (
        log.log_record.event_name
        == "gen_ai.client.inference.operation.details"
    )
    assert log.log_record.body is not None
    if parent_span:
        span_context = parent_span.get_span_context()
        assert log.log_record.trace_id == span_context.trace_id
        assert log.log_record.span_id == span_context.span_id
        assert log.log_record.trace_flags == span_context.trace_flags
    return dict(log.log_record.body)


@pytest.mark.vcr()
def test_function_call_choice(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    instrument_with_content: VertexAIInstrumentor,
    generate_content: callable,
):
    ask_about_weather(generate_content)

    # Emits span
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gemini-2.5-pro"
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.response.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["gen_ai.usage.input_tokens"] == 74
    assert attrs["gen_ai.usage.output_tokens"] == 16
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443

    # Tool definitions are always emitted (independent of content capture)
    assert attrs["gen_ai.request.function.0.name"] == "get_current_weather"
    assert (
        attrs["gen_ai.request.function.0.description"]
        == "Get the current weather in a given location"
    )
    assert "gen_ai.request.function.0.parameters" in attrs

    # Content on span
    assert "gen_ai.input.messages" in attrs
    input_msgs = json.loads(attrs["gen_ai.input.messages"])
    assert input_msgs == [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": "Get weather details in New Delhi and San Francisco?",
                }
            ],
        }
    ]

    # Output messages on span — function_call parts now appear as ToolCall
    assert "gen_ai.output.messages" in attrs
    output_msgs = json.loads(attrs["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "model"
    assert output_msgs[0]["finish_reason"] == "stop"
    assert len(output_msgs[0]["parts"]) == 2
    assert output_msgs[0]["parts"][0]["type"] == "tool_call"
    assert output_msgs[0]["parts"][0]["name"] == "get_current_weather"
    assert output_msgs[0]["parts"][0]["arguments"] == {"location": "New Delhi"}
    assert output_msgs[0]["parts"][1]["type"] == "tool_call"
    assert output_msgs[0]["parts"][1]["name"] == "get_current_weather"
    assert output_msgs[0]["parts"][1]["arguments"] == {
        "location": "San Francisco"
    }

    # Content events emitter emits a single event
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    body = assert_handler_event(logs[0], span)
    assert "gen_ai.input.messages" in body
    assert "gen_ai.output.messages" in body


@pytest.mark.vcr()
def test_function_call_choice_no_content(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    instrument_no_content: VertexAIInstrumentor,
    generate_content: callable,
):
    ask_about_weather(generate_content)

    # Emits span without content
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.provider.name"] == "vertex_ai"

    # Tool definitions are always emitted (independent of content capture)
    assert attrs["gen_ai.request.function.0.name"] == "get_current_weather"
    assert (
        attrs["gen_ai.request.function.0.description"]
        == "Get the current weather in a given location"
    )
    assert "gen_ai.request.function.0.parameters" in attrs

    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs

    # No events emitted when content is disabled
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


@pytest.mark.vcr()
def test_tool_events(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    instrument_with_content: VertexAIInstrumentor,
    generate_content: callable,
):
    ask_about_weather_function_response(generate_content)

    # Emits span
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gemini-2.5-pro"
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.response.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["gen_ai.usage.input_tokens"] == 128
    assert attrs["gen_ai.usage.output_tokens"] == 26
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443

    # Tool definitions are always emitted
    assert attrs["gen_ai.request.function.0.name"] == "get_current_weather"
    assert (
        attrs["gen_ai.request.function.0.description"]
        == "Get the current weather in a given location"
    )
    assert "gen_ai.request.function.0.parameters" in attrs

    # Content on span: user text, model function_call, user tool responses, model text response
    assert "gen_ai.input.messages" in attrs
    input_msgs = json.loads(attrs["gen_ai.input.messages"])
    assert len(input_msgs) == 3
    # First message: user text
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"] == [
        {
            "type": "text",
            "content": "Get weather details in New Delhi and San Francisco?",
        }
    ]
    # Second message: model with function_call parts now mapped to ToolCall
    assert input_msgs[1]["role"] == "model"
    assert len(input_msgs[1]["parts"]) == 2
    assert input_msgs[1]["parts"][0]["type"] == "tool_call"
    assert input_msgs[1]["parts"][0]["name"] == "get_current_weather"
    assert input_msgs[1]["parts"][0]["arguments"] == {"location": "New Delhi"}
    assert input_msgs[1]["parts"][1]["type"] == "tool_call"
    assert input_msgs[1]["parts"][1]["name"] == "get_current_weather"
    assert input_msgs[1]["parts"][1]["arguments"] == {
        "location": "San Francisco"
    }
    # Third message: user with tool call responses
    assert input_msgs[2]["role"] == "user"
    assert len(input_msgs[2]["parts"]) == 2
    assert input_msgs[2]["parts"][0]["type"] == "tool_call_response"
    assert input_msgs[2]["parts"][1]["type"] == "tool_call_response"

    # Output messages on span
    assert "gen_ai.output.messages" in attrs
    output_msgs = json.loads(attrs["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "model"
    assert output_msgs[0]["finish_reason"] == "stop"
    assert len(output_msgs[0]["parts"]) == 1
    assert output_msgs[0]["parts"][0]["type"] == "text"

    # Content events emitter emits a single event
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    body = assert_handler_event(logs[0], span)
    assert "gen_ai.input.messages" in body
    assert "gen_ai.output.messages" in body


@pytest.mark.vcr()
def test_tool_events_no_content(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    instrument_no_content: VertexAIInstrumentor,
    generate_content: callable,
):
    ask_about_weather_function_response(generate_content)

    # Emits span without content
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.response.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["gen_ai.usage.input_tokens"] == 128
    assert attrs["gen_ai.usage.output_tokens"] == 22
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443

    # Tool definitions are always emitted (independent of content capture)
    assert attrs["gen_ai.request.function.0.name"] == "get_current_weather"
    assert (
        attrs["gen_ai.request.function.0.description"]
        == "Get the current weather in a given location"
    )
    assert "gen_ai.request.function.0.parameters" in attrs

    # finish_reason stays "stop" because the *response* is a final text
    # answer (no function_call parts in the response candidates)
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs

    # No events emitted when content is disabled
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0
