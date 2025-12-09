"""Unit tests configuration module."""

import pytest

from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="function")
def span_exporter():
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture(scope="function")
def tracer_provider(span_exporter):
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrumentor():
    """Create and return a WeaviateInstrumentor instance."""
    return WeaviateInstrumentor()


@pytest.fixture(scope="function", autouse=True)
def reset_instrumentor(instrumentor):
    """Ensure instrumentor is uninstrumented after each test."""
    yield
    try:
        instrumentor.uninstrument()
    except Exception:
        pass  # Ignore errors if not instrumented
