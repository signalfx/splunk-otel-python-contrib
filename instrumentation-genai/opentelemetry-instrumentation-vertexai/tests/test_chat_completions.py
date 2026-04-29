from __future__ import annotations

import json

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)
from vertexai.preview.generative_models import (
    GenerativeModel as PreviewGenerativeModel,
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
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode


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
def test_generate_content(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    generate_content: callable,
    instrument_with_content: VertexAIInstrumentor,
):
    model = GenerativeModel("gemini-2.5-pro")
    generate_content(
        model,
        [
            Content(
                role="user",
                parts=[
                    Part.from_text("Say this is a test"),
                ],
            ),
        ],
    )

    # Emits span
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gemini-2.5-pro"

    # Core span attributes
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.response.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443

    # Content on span (JSON-encoded input/output messages)
    assert "gen_ai.input.messages" in attrs
    input_msgs = json.loads(attrs["gen_ai.input.messages"])
    assert input_msgs == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Say this is a test"}],
        }
    ]

    assert "gen_ai.output.messages" in attrs
    output_msgs = json.loads(attrs["gen_ai.output.messages"])
    assert output_msgs == [
        {
            "role": "model",
            "parts": [{"type": "text", "content": "This is a test."}],
            "finish_reason": "stop",
        }
    ]

    # Content events emitter emits a single event
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    body = assert_handler_event(logs[0], span)
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Say this is a test"}],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "model",
            "parts": [{"type": "text", "content": "This is a test."}],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.vcr()
def test_generate_content_no_content(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    generate_content: callable,
    instrument_no_content: VertexAIInstrumentor,
):
    model = GenerativeModel("gemini-2.5-pro")
    generate_content(
        model,
        [
            Content(role="user", parts=[Part.from_text("Say this is a test")]),
        ],
    )

    # Emits span without content attributes
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
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443

    # No content attributes on span
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs

    # No events emitted when content is disabled
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


@pytest.mark.vcr()
def test_generate_content_empty_model(
    span_exporter: InMemorySpanExporter,
    generate_content: callable,
    instrument_with_content: VertexAIInstrumentor,
):
    model = GenerativeModel("")
    try:
        generate_content(
            model,
            [
                Content(
                    role="user", parts=[Part.from_text("Say this is a test")]
                )
            ],
        )
    except ValueError:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat "
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == ""
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443
    assert_span_error(spans[0])


@pytest.mark.vcr()
def test_generate_content_missing_model(
    span_exporter: InMemorySpanExporter,
    generate_content: callable,
    instrument_with_content: VertexAIInstrumentor,
):
    model = GenerativeModel("gemini-does-not-exist")
    try:
        generate_content(
            model,
            [
                Content(
                    role="user", parts=[Part.from_text("Say this is a test")]
                )
            ],
        )
    except NotFound:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat gemini-does-not-exist"
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-does-not-exist"
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443
    assert_span_error(spans[0])


@pytest.mark.vcr()
def test_generate_content_invalid_temperature(
    span_exporter: InMemorySpanExporter,
    generate_content: callable,
    instrument_with_content: VertexAIInstrumentor,
):
    model = GenerativeModel("gemini-2.5-pro")
    try:
        # Temperature out of range causes error
        generate_content(
            model,
            [
                Content(
                    role="user", parts=[Part.from_text("Say this is a test")]
                )
            ],
            generation_config=GenerationConfig(temperature=1000),
        )
    except BadRequest:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat gemini-2.5-pro"
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.request.temperature"] == 1000.0
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443
    assert_span_error(spans[0])


@pytest.mark.vcr()
def test_generate_content_extra_params(
    span_exporter,
    instrument_no_content,
    generate_content: callable,
):
    generation_config = GenerationConfig(
        top_k=2,
        top_p=0.95,
        temperature=0.2,
        stop_sequences=["\n\n\n"],
        max_output_tokens=5,
        presence_penalty=-1.5,
        frequency_penalty=1.0,
        seed=12345,
    )
    model = GenerativeModel("gemini-2.5-pro")
    generate_content(
        model,
        [
            Content(role="user", parts=[Part.from_text("Say this is a test")]),
        ],
        generation_config=generation_config,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.frequency_penalty"] == 1.0
    assert attrs["gen_ai.request.max_tokens"] == 5
    assert attrs["gen_ai.request.seed"] == 12345
    assert attrs["gen_ai.request.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.request.presence_penalty"] == -1.5
    assert attrs["gen_ai.request.stop_sequences"] == ("\n\n\n",)
    assert attrs["gen_ai.request.temperature"] == 0.20000000298023224
    assert attrs["gen_ai.request.top_p"] == 0.949999988079071
    assert attrs["gen_ai.response.finish_reasons"] == ("length",)
    assert attrs["gen_ai.response.model"] == "gemini-2.5-pro"
    assert attrs["gen_ai.provider.name"] == "vertex_ai"
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 0
    assert attrs["server.address"] == "us-central1-aiplatform.googleapis.com"
    assert attrs["server.port"] == 443


def assert_span_error(span: ReadableSpan, error_type: str = None) -> None:
    # Sets error status
    assert span.status.status_code == StatusCode.ERROR

    # TelemetryHandler sets error.type attribute
    attrs = dict(span.attributes)
    assert "error.type" in attrs
    if error_type:
        assert attrs["error.type"] == error_type


@pytest.mark.vcr()
def test_generate_content_all_events(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    generate_content: callable,
    instrument_with_content: VertexAIInstrumentor,
):
    generate_content_all_input_messages(
        GenerativeModel(
            "gemini-2.5-pro",
            system_instruction=Part.from_text(
                "You are a clever language model"
            ),
        ),
        span_exporter,
        log_exporter,
    )


@pytest.mark.vcr()
def test_preview_generate_content_all_input_events(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    generate_content: callable,
    instrument_with_content: VertexAIInstrumentor,
):
    generate_content_all_input_messages(
        PreviewGenerativeModel(
            "gemini-2.5-pro",
            system_instruction=Part.from_text(
                "You are a clever language model"
            ),
        ),
        span_exporter,
        log_exporter,
    )


def generate_content_all_input_messages(
    model: GenerativeModel | PreviewGenerativeModel,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
):
    model.generate_content(
        [
            Content(
                role="user", parts=[Part.from_text("My name is OpenTelemetry")]
            ),
            Content(
                role="model", parts=[Part.from_text("Hello OpenTelemetry!")]
            ),
            Content(
                role="user",
                parts=[
                    Part.from_text("Address me by name and say this is a test")
                ],
            ),
        ],
        generation_config=GenerationConfig(
            seed=12345, response_mime_type="text/plain"
        ),
    )

    # Emits span with content attributes
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes)

    # Verify system instructions are on the span
    assert "gen_ai.system_instructions" in attrs

    # Verify input messages on span (system messages excluded from input)
    assert "gen_ai.input.messages" in attrs
    input_msgs = json.loads(attrs["gen_ai.input.messages"])
    assert len(input_msgs) == 3
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"] == [
        {"type": "text", "content": "My name is OpenTelemetry"}
    ]
    assert input_msgs[1]["role"] == "model"
    assert input_msgs[1]["parts"] == [
        {"type": "text", "content": "Hello OpenTelemetry!"}
    ]
    assert input_msgs[2]["role"] == "user"
    assert input_msgs[2]["parts"] == [
        {
            "type": "text",
            "content": "Address me by name and say this is a test",
        }
    ]

    # Verify output messages on span
    assert "gen_ai.output.messages" in attrs
    output_msgs = json.loads(attrs["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "model"
    assert output_msgs[0]["parts"] == [
        {"type": "text", "content": "OpenTelemetry, this is a test."}
    ]
    assert output_msgs[0]["finish_reason"] == "stop"

    # Content events emitter emits a single event
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    body = assert_handler_event(logs[0], span)
    assert "gen_ai.input.messages" in body
    assert "gen_ai.output.messages" in body
