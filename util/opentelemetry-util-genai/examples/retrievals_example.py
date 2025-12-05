#!/usr/bin/env python3
"""Example demonstrating OpenTelemetry GenAI telemetry for retrieval operations.

This example shows:
1. Basic retrieval invocation lifecycle
2. Retrieval with vector search
3. Retrieval with text query and metadata
4. Retrieval with custom attributes
5. Error handling for retrieval operations
6. Retrieval with agent context
7. Metrics and span emission for retrievals
"""

import time

from opentelemetry import _logs as logs
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import Error, RetrievalInvocation


def setup_telemetry():
    """Set up OpenTelemetry providers for tracing, metrics, and logging."""
    # Set up tracing
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(trace_provider)

    # Set up metrics
    metric_reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(), export_interval_millis=5000
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    # Set up logging (for events)
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(ConsoleLogExporter())
    )
    logs.set_logger_provider(logger_provider)

    return trace_provider, meter_provider, logger_provider


def example_basic_retrieval():
    """Example 1: Basic retrieval invocation with text query."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Retrieval Invocation")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create retrieval invocation
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="What is OpenTelemetry?",
        top_k=5,
        retriever_type="vector_store",
        provider="pinecone",
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.05)  # Simulate API call

    # Simulate response - populate results
    retrieval.documents_retrieved = 5
    retrieval.results = [
        {"id": "doc1", "score": 0.95, "content": "OpenTelemetry is..."},
        {"id": "doc2", "score": 0.89, "content": "OTEL provides..."},
        {"id": "doc3", "score": 0.85, "content": "Observability with..."},
        {"id": "doc4", "score": 0.82, "content": "Tracing and metrics..."},
        {"id": "doc5", "score": 0.78, "content": "Distributed tracing..."},
    ]

    # Finish the retrieval operation
    handler.stop_retrieval(retrieval)

    print("✓ Completed retrieval for text query")
    print(f"  Query: {retrieval.query}")
    print(f"  Documents retrieved: {retrieval.documents_retrieved}")
    print(f"  Provider: {retrieval.provider}")


def example_vector_search():
    """Example 2: Retrieval with vector search."""
    print("\n" + "=" * 60)
    print("Example 2: Vector Search Retrieval")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create retrieval with query vector
    query_vector = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim vector

    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query_vector=query_vector,
        top_k=10,
        retriever_type="vector_store",
        provider="chroma",
        framework="langchain",
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.08)  # Simulate API call

    # Simulate response
    retrieval.documents_retrieved = 10
    retrieval.results = [
        {"id": f"doc{i}", "score": 0.95 - i * 0.05} for i in range(10)
    ]

    # Finish the retrieval operation
    handler.stop_retrieval(retrieval)

    print("✓ Completed vector search retrieval")
    print(f"  Vector dimensions: {len(query_vector)}")
    print(f"  Documents retrieved: {retrieval.documents_retrieved}")
    print(f"  Framework: {retrieval.framework}")


def example_retrieval_with_metadata():
    """Example 3: Retrieval with result metadata."""
    print("\n" + "=" * 60)
    print("Example 3: Retrieval with Metadata")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create retrieval
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="machine learning tutorials",
        top_k=3,
        retriever_type="hybrid_search",
        provider="weaviate",
        framework="langchain",
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.06)  # Simulate API call

    # Simulate response with metadata
    retrieval.documents_retrieved = 3
    retrieval.results = [
        {
            "id": "tut1",
            "score": 0.92,
            "content": "Intro to ML",
            "metadata": {"category": "tutorial", "difficulty": "beginner"},
        },
        {
            "id": "tut2",
            "score": 0.88,
            "content": "Python ML basics",
            "metadata": {"category": "tutorial", "difficulty": "beginner"},
        },
        {
            "id": "tut3",
            "score": 0.85,
            "content": "Getting started with ML",
            "metadata": {"category": "tutorial", "difficulty": "beginner"},
        },
    ]

    # Finish the retrieval operation
    handler.stop_retrieval(retrieval)

    print("✓ Completed retrieval with metadata")
    print(f"  Query: {retrieval.query}")
    print(f"  Retriever type: {retrieval.retriever_type}")
    print(f"  Documents retrieved: {retrieval.documents_retrieved}")


def example_retrieval_with_custom_attributes():
    """Example 4: Retrieval with custom attributes."""
    print("\n" + "=" * 60)
    print("Example 4: Retrieval with Custom Attributes")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create retrieval with custom attributes
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="customer support documentation",
        top_k=5,
        retriever_type="semantic_search",
        provider="qdrant",
        attributes={
            "collection_name": "support_docs",
            "user_id": "user-789",
            "session_id": "session-456",
            "search_type": "semantic",
        },
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.07)  # Simulate API call

    # Simulate response
    retrieval.documents_retrieved = 5

    # Finish the retrieval operation
    handler.stop_retrieval(retrieval)

    print("✓ Completed retrieval with custom attributes")
    print(f"  Query: {retrieval.query}")
    print(f"  Custom attributes: {retrieval.attributes}")


def example_retrieval_with_agent_context():
    """Example 5: Retrieval within an agent context."""
    print("\n" + "=" * 60)
    print("Example 5: Retrieval with Agent Context")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create retrieval with agent context
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="latest product updates",
        top_k=7,
        retriever_type="vector_store",
        provider="milvus",
        framework="langchain",
        agent_name="product_assistant",
        agent_id="agent-123",
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.05)  # Simulate API call

    # Simulate response
    retrieval.documents_retrieved = 7

    # Finish the retrieval operation
    handler.stop_retrieval(retrieval)

    print("✓ Completed retrieval with agent context")
    print(f"  Agent: {retrieval.agent_name} (ID: {retrieval.agent_id})")
    print(f"  Query: {retrieval.query}")
    print(f"  Documents retrieved: {retrieval.documents_retrieved}")


def example_retrieval_error():
    """Example 6: Handling retrieval errors."""
    print("\n" + "=" * 60)
    print("Example 6: Retrieval Error Handling")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create retrieval invocation
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="test query",
        top_k=5,
        retriever_type="vector_store",
        provider="pinecone",
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.03)  # Simulate API call

    # Simulate an error
    error = Error(
        message="Connection timeout to vector store",
        type=TimeoutError,
    )

    # Fail the retrieval operation
    handler.fail_retrieval(retrieval, error)

    print("✗ Retrieval failed with error")
    print(f"  Error: {error.message}")
    print(f"  Provider: {retrieval.provider}")


def example_multiple_retrievals():
    """Example 7: Multiple sequential retrievals."""
    print("\n" + "=" * 60)
    print("Example 7: Multiple Sequential Retrievals")
    print("=" * 60)

    handler = get_telemetry_handler()

    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
    ]

    for idx, query_text in enumerate(queries, 1):
        retrieval = RetrievalInvocation(
            operation_name="retrieval",
            query=query_text,
            top_k=5,
            retriever_type="vector_store",
            provider="pinecone",
            attributes={"query_index": idx},
        )

        handler.start_retrieval(retrieval)
        time.sleep(0.04)  # Simulate API call

        # Simulate response
        retrieval.documents_retrieved = 5

        handler.stop_retrieval(retrieval)
        print(f"  ✓ Completed retrieval {idx}/{len(queries)}")

    print(f"✓ Completed all {len(queries)} retrievals")


def example_hybrid_retrieval():
    """Example 8: Hybrid retrieval combining text and vector search."""
    print("\n" + "=" * 60)
    print("Example 8: Hybrid Retrieval")
    print("=" * 60)

    handler = get_telemetry_handler()

    # Create hybrid retrieval with both query and vector
    retrieval = RetrievalInvocation(
        operation_name="retrieval",
        query="artificial intelligence applications",
        query_vector=[0.2] * 768,  # 768-dim vector
        top_k=8,
        retriever_type="hybrid_search",
        provider="elasticsearch",
        framework="langchain",
        attributes={
            "alpha": 0.5,  # Balance between text and vector search
            "boost_query": True,
        },
    )

    # Start the retrieval operation
    handler.start_retrieval(retrieval)
    time.sleep(0.09)  # Simulate API call

    # Simulate response
    retrieval.documents_retrieved = 8
    retrieval.results = [
        {"id": f"doc{i}", "score": 0.9 - i * 0.05, "type": "hybrid"}
        for i in range(8)
    ]

    # Finish the retrieval operation
    handler.stop_retrieval(retrieval)

    print("✓ Completed hybrid retrieval")
    print(f"  Query: {retrieval.query}")
    print(f"  Vector dimensions: {len(retrieval.query_vector)}")
    print(f"  Retriever type: {retrieval.retriever_type}")
    print(f"  Documents retrieved: {retrieval.documents_retrieved}")


def main():
    """Run all retrieval examples."""
    print("\n" + "=" * 60)
    print("OpenTelemetry GenAI Retrievals Examples")
    print("=" * 60)

    # Set up telemetry
    trace_provider, meter_provider, logger_provider = setup_telemetry()

    # Run examples
    example_basic_retrieval()
    example_vector_search()
    example_retrieval_with_metadata()
    example_retrieval_with_custom_attributes()
    example_retrieval_with_agent_context()
    example_retrieval_error()
    example_multiple_retrievals()
    example_hybrid_retrieval()

    # Force flush to ensure all telemetry is exported
    print("\n" + "=" * 60)
    print("Flushing telemetry data...")
    print("=" * 60)
    trace_provider.force_flush()
    meter_provider.force_flush()
    logger_provider.force_flush()

    print("\n✓ All examples completed successfully!")
    print("Check the console output above for spans, metrics, and events.\n")


if __name__ == "__main__":
    main()
