"""Test embedding instrumentation for LlamaIndex."""

import os

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor


# Global setup - shared across tests
metric_reader = None
instrumentor = None


def setup_telemetry():
    """Setup OpenTelemetry with span and metric exporters (once)."""
    global metric_reader, instrumentor

    if metric_reader is not None:
        return metric_reader

    # Enable metrics
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"

    # Setup tracing
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    # Setup metrics with InMemoryMetricReader
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Enable instrumentation once
    instrumentor = LlamaindexInstrumentor()
    instrumentor.instrument(
        tracer_provider=trace.get_tracer_provider(),
        meter_provider=metrics.get_meter_provider(),
    )

    return metric_reader


def test_embedding_single_text():
    """Test single text embedding instrumentation."""
    print("\nTest: Single Text Embedding")
    print("=" * 60)

    metric_reader = setup_telemetry()

    # Configure embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    Settings.embed_model = embed_model

    # Make sure callback manager is initialized
    if Settings.callback_manager is None:
        Settings.callback_manager = CallbackManager()

    # Generate single embedding
    text = "LlamaIndex is a data framework for LLM applications"
    embedding = embed_model.get_text_embedding(text)

    print(f"\nText: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Validate metrics
    print("\nMetrics:")
    metrics_data = metric_reader.get_metrics_data()
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                print(f"\nMetric: {metric.name}")
                for data_point in metric.data.data_points:
                    if hasattr(data_point, "bucket_counts"):
                        # Histogram
                        print(f"  Count: {sum(data_point.bucket_counts)}")
                    else:
                        # Counter
                        print(f"  Value: {data_point.value}")

    print("\nTest completed successfully")


def test_embedding_batch():
    """Test batch embedding instrumentation."""
    print("\nTest: Batch Embeddings")
    print("=" * 60)

    metric_reader = setup_telemetry()

    # Configure embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    Settings.embed_model = embed_model

    # Make sure callback manager is initialized
    if Settings.callback_manager is None:
        Settings.callback_manager = CallbackManager()

    # Generate batch embeddings
    texts = [
        "Paris is the capital of France",
        "Berlin is the capital of Germany",
        "Rome is the capital of Italy",
    ]
    embeddings = embed_model.get_text_embedding_batch(texts)

    print(f"\nEmbedded {len(embeddings)} texts")
    print(f"Dimension: {len(embeddings[0])}")

    # Validate metrics
    print("\nMetrics:")
    metrics_data = metric_reader.get_metrics_data()
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                print(f"\nMetric: {metric.name}")
                for data_point in metric.data.data_points:
                    if hasattr(data_point, "bucket_counts"):
                        # Histogram
                        print(f"  Count: {sum(data_point.bucket_counts)}")
                    else:
                        # Counter
                        print(f"  Value: {data_point.value}")

    print("\nTest completed successfully")


if __name__ == "__main__":
    test_embedding_single_text()
    print("\n" + "=" * 60 + "\n")
    test_embedding_batch()

    # Cleanup
    if instrumentor:
        instrumentor.uninstrument()
