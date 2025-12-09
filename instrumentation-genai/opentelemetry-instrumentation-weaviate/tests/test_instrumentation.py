"""Tests for Weaviate instrumentation."""

import pytest

from opentelemetry import trace
from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor


class TestWeaviateInstrumentation:
    """Test basic instrumentation functionality."""

    def test_instrumentor_initialization(self, instrumentor):
        """Test that instrumentor can be initialized."""
        assert instrumentor is not None
        assert isinstance(instrumentor, WeaviateInstrumentor)

    def test_instrument_uninstrument(self, instrumentor, tracer_provider):
        """Test that instrumentation can be applied and removed."""
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument
        instrumentor.instrument(tracer_provider=tracer_provider)
        
        # Uninstrument
        instrumentor.uninstrument()

    def test_instrumentation_dependencies(self, instrumentor):
        """Test that instrumentation dependencies are correctly specified."""
        dependencies = instrumentor.instrumentation_dependencies()
        assert dependencies is not None
        assert len(dependencies) > 0
        assert any("weaviate-client" in dep for dep in dependencies)

    def test_double_instrument(self, instrumentor, tracer_provider):
        """Test that double instrumentation doesn't cause errors."""
        trace.set_tracer_provider(tracer_provider)
        
        instrumentor.instrument(tracer_provider=tracer_provider)
        # Second instrumentation should be idempotent
        instrumentor.instrument(tracer_provider=tracer_provider)
        
        instrumentor.uninstrument()

    def test_uninstrument_without_instrument(self, instrumentor):
        """Test that uninstrument works even if not instrumented."""
        # Should not raise an error
        instrumentor.uninstrument()
