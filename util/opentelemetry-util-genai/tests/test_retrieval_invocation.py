"""Tests for RetrievalInvocation lifecycle and telemetry."""

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.attributes import (
    GEN_AI_RETRIEVAL_DOCUMENTS_RETRIEVED,
    GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_RETRIEVAL_TYPE,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import Error, RetrievalInvocation


def test_retrieval_invocation_basic_lifecycle():
    """Test basic start/stop lifecycle for retrieval invocation."""
    handler = get_telemetry_handler()
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="test query",
        top_k=5,
        retriever_type="vector_store",
        provider="pinecone",
    )

    # Start should assign span
    result = handler.start_retrieval(retrieval)
    assert result is retrieval
    assert retrieval.span is not None
    assert retrieval.start_time is not None

    # Stop should set end_time and end span
    retrieval.documents_retrieved = 5
    handler.stop_retrieval(retrieval)
    assert retrieval.end_time is not None
    assert retrieval.end_time >= retrieval.start_time


def test_retrieval_invocation_with_error():
    """Test error handling for retrieval invocation."""
    handler = get_telemetry_handler()
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="failing query",
        top_k=10,
        retriever_type="vector_store",
        provider="chroma",
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None

    # Fail the retrieval
    error = Error(message="Connection timeout", type=TimeoutError)
    handler.fail_retrieval(retrieval, error)
    assert retrieval.end_time is not None


def test_retrieval_invocation_creates_span_with_attributes():
    """Test that retrieval invocation creates span with correct attributes."""
    # Set up in-memory span exporter
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="OpenTelemetry documentation",
        top_k=7,
        retriever_type="semantic_search",
        provider="weaviate",
        framework="langchain",
    )

    handler.start_retrieval(retrieval)
    retrieval.documents_retrieved = 7
    handler.stop_retrieval(retrieval)

    # Get exported spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    attrs = span.attributes

    # Check required attributes
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "retrieval"

    # Check recommended attributes
    assert attrs[GEN_AI_RETRIEVAL_TYPE] == "semantic_search"
    assert attrs[GEN_AI_RETRIEVAL_TOP_K] == 7
    assert attrs[GEN_AI_RETRIEVAL_DOCUMENTS_RETRIEVED] == 7

    # Check provider and framework
    assert attrs[GenAI.GEN_AI_PROVIDER_NAME] == "weaviate"
    assert attrs.get("gen_ai.framework") == "langchain"


def test_retrieval_invocation_with_vector_search():
    """Test retrieval with query vector."""
    handler = get_telemetry_handler()
    query_vector = [0.1, 0.2, 0.3] * 256  # 768-dim vector

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query_vector=query_vector,
        top_k=10,
        retriever_type="vector_store",
        provider="pinecone",
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None
    assert retrieval.query_vector == query_vector

    retrieval.documents_retrieved = 10
    handler.stop_retrieval(retrieval)
    assert retrieval.end_time is not None


def test_retrieval_invocation_with_hybrid_search():
    """Test retrieval with both text query and vector."""
    handler = get_telemetry_handler()

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="machine learning",
        query_vector=[0.5] * 384,
        top_k=15,
        retriever_type="hybrid_search",
        provider="elasticsearch",
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None
    assert retrieval.query == "machine learning"
    assert len(retrieval.query_vector) == 384

    retrieval.documents_retrieved = 15
    handler.stop_retrieval(retrieval)


def test_retrieval_invocation_with_agent_context():
    """Test retrieval within agent context."""
    handler = get_telemetry_handler()

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="product information",
        top_k=5,
        retriever_type="vector_store",
        provider="milvus",
        agent_name="product_assistant",
        agent_id="agent-123",
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None
    assert retrieval.agent_name == "product_assistant"
    assert retrieval.agent_id == "agent-123"

    retrieval.documents_retrieved = 5
    handler.stop_retrieval(retrieval)


def test_retrieval_invocation_with_custom_attributes():
    """Test retrieval with custom attributes."""
    handler = get_telemetry_handler()

    custom_attrs = {
        "collection_name": "docs",
        "user_id": "user-456",
        "session_id": "session-789",
    }

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="custom search",
        top_k=3,
        retriever_type="vector_store",
        provider="qdrant",
        attributes=custom_attrs,
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None
    assert retrieval.attributes == custom_attrs

    retrieval.documents_retrieved = 3
    handler.stop_retrieval(retrieval)


def test_retrieval_invocation_with_results():
    """Test retrieval with result documents."""
    handler = get_telemetry_handler()

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="test",
        top_k=2,
        retriever_type="vector_store",
        provider="pinecone",
    )

    handler.start_retrieval(retrieval)

    # Populate results
    retrieval.documents_retrieved = 2
    retrieval.results = [
        {"id": "doc1", "score": 0.95, "content": "First document"},
        {"id": "doc2", "score": 0.87, "content": "Second document"},
    ]

    handler.stop_retrieval(retrieval)
    assert len(retrieval.results) == 2
    assert retrieval.results[0]["score"] == 0.95


def test_retrieval_invocation_semantic_convention_attributes():
    """Test that semantic convention attributes are properly extracted."""
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        request_model="text-embedding-ada-002",
        query="semantic test",
        top_k=5,
        retriever_type="vector_store",
        provider="test_provider",
    )

    semconv_attrs = retrieval.semantic_convention_attributes()

    # Check that semantic convention attributes are present
    assert GenAI.GEN_AI_OPERATION_NAME in semconv_attrs
    assert semconv_attrs[GenAI.GEN_AI_OPERATION_NAME] == "retrieval"
    assert GenAI.GEN_AI_REQUEST_MODEL in semconv_attrs
    assert (
        semconv_attrs[GenAI.GEN_AI_REQUEST_MODEL] == "text-embedding-ada-002"
    )
    assert "gen_ai.retrieval.type" in semconv_attrs
    assert semconv_attrs["gen_ai.retrieval.type"] == "vector_store"
    assert "gen_ai.retrieval.query.text" in semconv_attrs
    assert semconv_attrs["gen_ai.retrieval.query.text"] == "semantic test"
    assert "gen_ai.retrieval.top_k" in semconv_attrs
    assert semconv_attrs["gen_ai.retrieval.top_k"] == 5


def test_retrieval_invocation_minimal_required_fields():
    """Test retrieval with only required fields."""
    handler = get_telemetry_handler()

    # Only operation_name is required
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None

    handler.stop_retrieval(retrieval)
    assert retrieval.end_time is not None


def test_retrieval_invocation_multiple_sequential():
    """Test multiple sequential retrieval invocations."""
    handler = get_telemetry_handler()

    queries = ["query1", "query2", "query3"]
    retrievals = []

    for query in queries:
        retrieval = RetrievalInvocation(
            operation_name="retrieval",
            query=query,
            top_k=5,
            retriever_type="vector_store",
            provider="pinecone",
        )
        handler.start_retrieval(retrieval)
        retrieval.documents_retrieved = 5
        handler.stop_retrieval(retrieval)
        retrievals.append(retrieval)

    # All should have completed successfully
    assert len(retrievals) == 3
    for retrieval in retrievals:
        assert retrieval.span is not None
        assert retrieval.end_time is not None


def test_generic_start_finish_for_retrieval():
    """Test generic handler methods route to retrieval lifecycle."""
    handler = get_telemetry_handler()

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="generic test",
        top_k=5,
        retriever_type="vector_store",
        provider="test",
    )

    # Generic methods should route to retrieval lifecycle
    handler.start(retrieval)
    assert retrieval.span is not None

    handler.finish(retrieval)
    assert retrieval.end_time is not None

    # Test fail path
    retrieval2 = RetrievalInvocation(
        operation_name="retrieval",
        query="fail test",
        top_k=3,
    )
    handler.start(retrieval2)
    handler.fail(retrieval2, Error(message="test error", type=RuntimeError))
    assert retrieval2.end_time is not None


def test_retrieval_invocation_span_name():
    """Test that span name is correctly formatted."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="test",
        provider="pinecone",
    )

    handler.start_retrieval(retrieval)
    handler.stop_retrieval(retrieval)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    # Span name should be "retrieval pinecone"
    assert spans[0].name == "retrieval pinecone"


def test_retrieval_invocation_without_provider():
    """Test retrieval without provider (span name should be just operation)."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="test",
    )

    handler.start_retrieval(retrieval)
    handler.stop_retrieval(retrieval)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    # Span name should be just "retrieval"
    assert spans[0].name == "retrieval"


@pytest.mark.parametrize(
    "retriever_type,provider",
    [
        ("vector_store", "pinecone"),
        ("semantic_search", "weaviate"),
        ("hybrid_search", "elasticsearch"),
        ("keyword_search", "opensearch"),
    ],
)
def test_retrieval_invocation_different_types(retriever_type, provider):
    """Test retrieval with different retriever types and providers."""
    handler = get_telemetry_handler()

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query=f"test {retriever_type}",
        top_k=5,
        retriever_type=retriever_type,
        provider=provider,
    )

    handler.start_retrieval(retrieval)
    assert retrieval.span is not None
    assert retrieval.retriever_type == retriever_type
    assert retrieval.provider == provider

    retrieval.documents_retrieved = 5
    handler.stop_retrieval(retrieval)
    assert retrieval.end_time is not None


def test_retrieval_invocation_with_server_and_model_attributes():
    """Test retrieval with server address, port, and model attributes."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        request_model="text-embedding-ada-002",
        query="test query",
        top_k=5,
        retriever_type="vector_store",
        provider="weaviate",
        server_address="localhost",
        server_port=8080,
    )

    handler.start_retrieval(retrieval)
    retrieval.documents_retrieved = 5
    handler.stop_retrieval(retrieval)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    attrs = span.attributes

    # Check new attributes
    assert attrs[GenAI.GEN_AI_REQUEST_MODEL] == "text-embedding-ada-002"
    assert attrs["server.address"] == "localhost"
    assert attrs["server.port"] == 8080


def test_retrieval_invocation_with_error_type():
    """Test retrieval with error_type attribute."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    handler = get_telemetry_handler()
    span_emitters = list(handler._emitter.emitters_for("span"))
    if span_emitters:
        span_emitters[0]._tracer = tracer_provider.get_tracer(__name__)

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="test query",
        top_k=5,
        retriever_type="vector_store",
        provider="pinecone",
        error_type="ConnectionError",
    )

    handler.start_retrieval(retrieval)
    error = Error(message="Connection failed", type=ConnectionError)
    handler.fail_retrieval(retrieval, error)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    attrs = span.attributes

    # Check error type attribute (should be set from invocation.error_type)
    assert attrs["error.type"] == "ConnectionError"
