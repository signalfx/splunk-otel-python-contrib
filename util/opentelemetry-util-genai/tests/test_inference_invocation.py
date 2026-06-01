import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
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


def test_inference_invocation_creates_span():
    handler = get_telemetry_handler()
    inv = handler.start_inference("openai", request_model="gpt-4o")
    assert inv.span is not None
    inv.stop()
    assert inv.end_time is not None


def test_inference_invocation_fail():
    handler = get_telemetry_handler()
    inv = handler.start_inference("anthropic", request_model="claude-3")
    inv.fail(
        Error(
            message="timeout",
            type=TimeoutError,
            classification=ErrorClassification.REAL_ERROR,
        )
    )
    assert inv.end_time is not None


def test_inference_invocation_fail_with_exception():
    handler = get_telemetry_handler()
    inv = handler.start_inference("openai")
    inv.fail(RuntimeError("boom"))
    assert inv.end_time is not None


def test_inference_invocation_span_kind():
    handler = _make_handler()
    inv = handler.start_inference("openai", request_model="gpt-4o")
    assert inv.span.kind == SpanKind.CLIENT
    inv.stop()


def test_inference_context_manager():
    handler = get_telemetry_handler()
    with handler.inference("openai", request_model="gpt-4o") as inv:
        inv.input_tokens = 10
        inv.output_tokens = 20
    assert inv.end_time is not None
    assert inv.input_tokens == 10


def test_inference_context_manager_propagates_exception():
    handler = get_telemetry_handler()
    with pytest.raises(ValueError):
        with handler.inference("openai") as inv:
            raise ValueError("bad input")
    assert inv.end_time is not None


def test_inference_direct_use_without_context_manager():
    handler = get_telemetry_handler()
    inv = handler.inference("openai", request_model="gpt-4o")
    inv.input_tokens = 5
    inv.stop()
    assert inv.end_time is not None
    assert inv.input_tokens == 5


def test_start_inference_operation_name():
    handler = get_telemetry_handler()
    inv = handler.start_inference("openai", operation_name="completion")
    inv.stop()
    attrs = inv.semantic_convention_attributes()
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "completion"


def test_inference_operation_name_default_is_chat():
    handler = get_telemetry_handler()
    inv = handler.start_inference("openai")
    inv.stop()
    attrs = inv.semantic_convention_attributes()
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "chat"


def test_inference_context_manager_operation_name():
    handler = get_telemetry_handler()
    with handler.inference("openai", operation_name="completion") as inv:
        pass
    attrs = inv.semantic_convention_attributes()
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "completion"
