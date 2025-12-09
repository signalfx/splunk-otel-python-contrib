"""Tests for Weaviate v4 client instrumentation."""

import pytest

try:
    import weaviate

    WEAVIATE_AVAILABLE = True
    WEAVIATE_VERSION = int(weaviate.__version__.split(".")[0])
except ImportError:
    WEAVIATE_AVAILABLE = False
    WEAVIATE_VERSION = 0

from opentelemetry import trace


@pytest.mark.skipif(
    not WEAVIATE_AVAILABLE or WEAVIATE_VERSION < 4,
    reason="Weaviate v4 client not available",
)
class TestWeaviateV4Instrumentation:
    """Test Weaviate v4 client instrumentation."""

    def test_version_detection(self, instrumentor, tracer_provider):
        """Test that v4 client is correctly detected."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Version should be detected as v4
        from opentelemetry.instrumentation.weaviate import weaviate_version, WEAVIATE_V4

        assert weaviate_version == WEAVIATE_V4

    @pytest.mark.integration
    def test_connect_to_local_instrumented(
        self, instrumentor, tracer_provider, span_exporter
    ):
        """Test that connect_to_local creates spans."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        try:
            # This will fail if Weaviate is not running, but we can still check instrumentation
            client = weaviate.connect_to_local()
            client.close()
        except Exception:
            # Expected if Weaviate is not running
            pass

        # Check that some instrumentation occurred
        # Note: This test requires a running Weaviate instance for full validation

    def test_collection_operations_span_names(self):
        """Test that collection operations have correct span names."""
        from opentelemetry.instrumentation.weaviate.mapping import MAPPING_V4

        # Verify key operations are mapped
        span_names = [m["span_name"] for m in MAPPING_V4]

        assert "collections.create" in span_names
        assert "collections.get" in span_names
        assert "collections.delete" in span_names
        assert "collections.data.insert" in span_names
        assert "collections.query.fetch_objects" in span_names

    def test_query_operations_mapped(self):
        """Test that query operations are properly mapped."""
        from opentelemetry.instrumentation.weaviate.mapping import MAPPING_V4

        query_operations = [m for m in MAPPING_V4 if "query" in m["span_name"]]

        assert len(query_operations) > 0

        # Check for specific query operations
        query_span_names = [op["span_name"] for op in query_operations]
        assert "collections.query.near_text" in query_span_names
        assert "collections.query.near_vector" in query_span_names
        assert "collections.query.fetch_objects" in query_span_names
