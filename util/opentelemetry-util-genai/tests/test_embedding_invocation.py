"""Tests for EmbeddingInvocation lifecycle, defaults, and telemetry."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import EmbeddingInvocation, Error


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


def test_embedding_invocation_default_operation_name():
    """EmbeddingInvocation should default operation_name to 'embeddings'."""
    emb = EmbeddingInvocation(
        request_model="text-embedding-ada-002",
        input_texts=["hello"],
    )
    assert (
        emb.operation_name == GenAI.GenAiOperationNameValues.EMBEDDINGS.value
    )
    assert emb.operation_name == "embeddings"


def test_embedding_invocation_semantic_convention_attributes():
    """semantic_convention_attributes() should include the default operation_name."""
    emb = EmbeddingInvocation(
        request_model="text-embedding-3-small",
        input_texts=["test input"],
        provider="openai",
    )
    semconv_attrs = emb.semantic_convention_attributes()

    assert GenAI.GEN_AI_OPERATION_NAME in semconv_attrs
    assert semconv_attrs[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"
    assert GenAI.GEN_AI_REQUEST_MODEL in semconv_attrs
    assert (
        semconv_attrs[GenAI.GEN_AI_REQUEST_MODEL] == "text-embedding-3-small"
    )


def test_embedding_invocation_span_attributes():
    """Spans should carry the correct operation_name attribute from the default."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    emb = EmbeddingInvocation(
        request_model="text-embedding-ada-002",
        input_texts=["hello world"],
        provider="openai",
    )

    handler.start_embedding(emb)
    handler.stop_embedding(emb)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    attrs = span.attributes

    # operation_name should be "embeddings" (the default from types.py)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"
    assert attrs[GenAI.GEN_AI_REQUEST_MODEL] == "text-embedding-ada-002"


def test_embedding_invocation_span_name():
    """Span name should be '{operation_name} {request_model}'."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    emb = EmbeddingInvocation(
        request_model="text-embedding-3-large",
        input_texts=["test"],
        provider="openai",
    )

    handler.start_embedding(emb)
    handler.stop_embedding(emb)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embeddings text-embedding-3-large"


def test_embedding_invocation_with_error():
    """Error path should still produce a span with correct operation_name."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    emb = EmbeddingInvocation(
        request_model="text-embedding-ada-002",
        input_texts=["test"],
        provider="openai",
    )

    handler.start_embedding(emb)
    handler.fail_embedding(emb, Error(message="API error", type=RuntimeError))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"


def test_embedding_invocation_custom_operation_name_override():
    """If a caller explicitly sets operation_name, the override should be honoured."""
    emb = EmbeddingInvocation(
        operation_name="custom_embedding",
        request_model="my-model",
        input_texts=["x"],
    )
    assert emb.operation_name == "custom_embedding"


def test_embedding_invocation_without_explicit_operation_name_matches_langchain_usage():
    """Verify the pattern used by langchain instrumentation (no operation_name kwarg)
    produces the correct default."""
    # This mirrors the construction in langchain __init__.py after the fix:
    # UtilEmbeddingInvocation(request_model=..., input_texts=..., provider=..., attributes=...)
    emb = EmbeddingInvocation(
        request_model="text-embedding-ada-002",
        input_texts=["hello world"],
        provider="openai",
        attributes={"framework": "langchain"},
    )
    assert emb.operation_name == "embeddings"
