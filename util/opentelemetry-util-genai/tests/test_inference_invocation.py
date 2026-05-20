import pytest
from opentelemetry.trace import SpanKind

from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai._inference_invocation import InferenceInvocation
from opentelemetry.util.genai._error import Error, ErrorClassification


def test_inference_invocation_creates_span():
    handler = get_telemetry_handler()
    inv = handler.start_inference("openai", request_model="gpt-4o")
    assert inv.span is not None
    inv.stop()
    assert inv.end_time is not None


def test_inference_invocation_fail():
    handler = get_telemetry_handler()
    inv = handler.start_inference("anthropic", request_model="claude-3")
    inv.fail(Error(message="timeout", type=TimeoutError, classification=ErrorClassification.REAL_ERROR))
    assert inv.end_time is not None


def test_inference_invocation_fail_with_exception():
    handler = get_telemetry_handler()
    inv = handler.start_inference("openai")
    inv.fail(RuntimeError("boom"))
    assert inv.end_time is not None


def test_inference_invocation_span_kind():
    handler = get_telemetry_handler()
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
