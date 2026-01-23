"""Tests for Weaviate v3 client instrumentation."""

import pytest

try:
    import weaviate

    WEAVIATE_AVAILABLE = True
    WEAVIATE_VERSION = int(weaviate.__version__.split(".")[0])
except ImportError:
    WEAVIATE_AVAILABLE = False
    WEAVIATE_VERSION = 0

from opentelemetry import trace


@pytest.mark.skipif(not WEAVIATE_AVAILABLE, reason="Weaviate client not available")
class TestWeaviateV3Instrumentation:
    """Test Weaviate v3 client instrumentation."""

    def test_v3_operations_mapped(self):
        """Test that v3 operations are properly mapped."""
        from opentelemetry.instrumentation.weaviate.mapping import MAPPING_V3

        # Verify key v3 operations are mapped
        span_names = [m["span_name"] for m in MAPPING_V3]

        # Schema operations
        assert "schema.get" in span_names
        assert "schema.create_class" in span_names
        assert "schema.delete_class" in span_names

        # Data operations
        assert "data.crud_data.create" in span_names
        assert "data.crud_data.get" in span_names

        # Batch operations
        assert "batch.crud_batch.add_data_object" in span_names

        # Query operations
        assert "gql.query.get" in span_names
        assert "gql.query.aggregate" in span_names
        assert "gql.query.raw" in span_names

    def test_v3_modules_structure(self):
        """Test that v3 module paths are correct."""
        from opentelemetry.instrumentation.weaviate.mapping import MAPPING_V3

        # Check that module paths follow v3 structure
        modules = [m["module"] for m in MAPPING_V3]

        assert "weaviate.schema" in modules
        assert "weaviate.data.crud_data" in modules
        assert "weaviate.batch.crud_batch" in modules
        assert "weaviate.gql.query" in modules
        assert "weaviate.gql.get" in modules

    @pytest.mark.skipif(
        WEAVIATE_VERSION >= 4, reason="Test only applicable for v3 client"
    )
    def test_v3_client_detection(self, instrumentor, tracer_provider):
        """Test that v3 client is correctly detected when installed."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        from opentelemetry.instrumentation.weaviate import weaviate_version, WEAVIATE_V3

        assert weaviate_version == WEAVIATE_V3
