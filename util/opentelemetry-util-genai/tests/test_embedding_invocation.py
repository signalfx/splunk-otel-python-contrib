import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai._error import Error, ErrorClassification
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    _current_genai_span,
    get_telemetry_handler,
)
from opentelemetry.util.genai.types import EmbeddingInvocation


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


def test_embedding_invocation_creates_span():
    handler = get_telemetry_handler()
    emb = EmbeddingInvocation(
        request_model="embedding-model",
        input_texts=["a"],
        provider="emb-provider",
    )
    handler.start_embedding(emb)
    assert emb.span is not None
    # ensure stop works without error
    handler.stop_embedding(emb)
    # span should have ended (recording possibly false depending on SDK impl)
    # we at least assert the object reference still exists
    assert emb.span is not None


def test_embedding_creates_span():
    handler = get_telemetry_handler()
    inv = handler.embedding("openai", request_model="text-embedding-3-small")
    assert inv.span is not None
    inv.stop()
    assert inv.end_time is not None


def test_embedding_fail():
    handler = get_telemetry_handler()
    inv = handler.embedding("openai", request_model="text-embedding-3-small")
    inv.fail(
        Error(
            message="rate limit exceeded",
            type=RuntimeError,
            classification=ErrorClassification.REAL_ERROR,
        )
    )
    assert inv.end_time is not None


def test_embedding_fail_with_exception():
    handler = get_telemetry_handler()
    inv = handler.embedding("openai")
    inv.fail(ValueError("invalid input"))
    assert inv.end_time is not None


def test_embedding_span_kind():
    handler = _make_handler()
    inv = handler.embedding("openai", request_model="text-embedding-3-small")
    assert inv.span.kind == SpanKind.CLIENT
    inv.stop()


def test_embedding_context_manager():
    handler = get_telemetry_handler()
    with handler.embedding(
        "openai", request_model="text-embedding-3-small"
    ) as inv:
        pass
    assert inv.end_time is not None


def test_embedding_context_manager_propagates_exception():
    handler = get_telemetry_handler()
    with pytest.raises(ValueError):
        with handler.embedding("openai") as inv:
            raise ValueError("bad input")
    assert inv.end_time is not None


def test_embedding_direct_use_without_context_manager():
    handler = get_telemetry_handler()
    inv = handler.embedding("openai", request_model="text-embedding-3-small")
    inv.stop()
    assert inv.end_time is not None


def test_embedding_with_server_address():
    handler = get_telemetry_handler()
    inv = handler.embedding(
        "openai",
        request_model="text-embedding-3-small",
        server_address="api.openai.com",
        server_port=443,
    )
    inv.stop()
    assert inv.end_time is not None
