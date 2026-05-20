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


def test_workflow_invocation_creates_span():
    handler = get_telemetry_handler()
    inv = handler.start_workflow(name="travel-planner")
    assert inv.span is not None
    inv.stop()
    assert inv.end_time is not None


def test_workflow_invocation_fail():
    handler = get_telemetry_handler()
    inv = handler.start_workflow(name="booking-flow")
    inv.fail(
        Error(
            message="planning failed",
            type=RuntimeError,
            classification=ErrorClassification.REAL_ERROR,
        )
    )
    assert inv.end_time is not None


def test_workflow_invocation_fail_with_exception():
    handler = get_telemetry_handler()
    inv = handler.start_workflow(name="research-flow")
    inv.fail(RuntimeError("upstream error"))
    assert inv.end_time is not None


def test_workflow_invocation_span_kind():
    handler = _make_handler()
    inv = handler.start_workflow(name="my-workflow")
    assert inv.span.kind == SpanKind.INTERNAL
    inv.stop()


def test_workflow_invocation_marks_conversation_root():
    handler = get_telemetry_handler()
    inv = handler.start_workflow(name="root-workflow")
    assert inv.conversation_root is True
    inv.stop()


def test_workflow_context_manager():
    handler = get_telemetry_handler()
    with handler.workflow(name="my-workflow", framework="langchain") as inv:
        pass
    assert inv.end_time is not None
    assert inv.framework == "langchain"


def test_workflow_context_manager_propagates_exception():
    handler = get_telemetry_handler()
    with pytest.raises(KeyError):
        with handler.workflow(name="failing-workflow") as inv:
            raise KeyError("missing key")
    assert inv.end_time is not None
