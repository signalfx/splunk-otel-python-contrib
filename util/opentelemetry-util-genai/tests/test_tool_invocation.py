import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai._error import Error, ErrorClassification
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    _current_genai_span,
    get_telemetry_handler,
)


@pytest.fixture(autouse=True)
def reset_handler():
    TelemetryHandler._reset_for_testing()
    _current_genai_span.set(None)
    yield
    TelemetryHandler._reset_for_testing()
    _current_genai_span.set(None)


def _make_handler():
    TelemetryHandler._reset_for_testing()
    tp = TracerProvider()
    return TelemetryHandler(tracer_provider=tp)


def test_tool_invocation_creates_span():
    handler = get_telemetry_handler()
    inv = handler.start_tool("search_web", arguments={"query": "python"})
    assert inv.span is not None
    inv.stop()
    assert inv.end_time is not None


def test_tool_invocation_fail():
    handler = get_telemetry_handler()
    inv = handler.start_tool(
        "fetch_url", arguments={"url": "https://example.com"}
    )
    inv.fail(
        Error(
            message="connection refused",
            type=ConnectionError,
            classification=ErrorClassification.REAL_ERROR,
        )
    )
    assert inv.end_time is not None


def test_tool_invocation_fail_with_exception():
    handler = get_telemetry_handler()
    inv = handler.start_tool("parse_json")
    inv.fail(ValueError("invalid JSON"))
    assert inv.end_time is not None


def test_tool_invocation_span_kind():
    handler = _make_handler()
    inv = handler.start_tool("lookup")
    assert inv.span.kind == SpanKind.INTERNAL
    inv.stop()


def test_tool_invocation_optional_fields():
    handler = get_telemetry_handler()
    inv = handler.start_tool(
        "translate",
        arguments={"text": "hello"},
        tool_call_id="tc-123",
        tool_type="function",
        tool_description="Translates text",
    )
    assert inv.id == "tc-123"
    assert inv.tool_type == "function"
    assert inv.tool_description == "Translates text"
    inv.tool_result = "hola"
    inv.stop()
    assert inv.end_time is not None


def test_tool_context_manager():
    handler = get_telemetry_handler()
    with handler.tool("calculate", arguments={"expr": "1+1"}) as inv:
        inv.tool_result = "2"
    assert inv.end_time is not None
    assert inv.tool_result == "2"


def test_tool_context_manager_propagates_exception():
    handler = get_telemetry_handler()
    with pytest.raises(TypeError):
        with handler.tool("bad_tool") as inv:
            raise TypeError("wrong type")
    assert inv.end_time is not None
