"""Tests for span attributes and semantic conventions."""

import pytest

from opentelemetry.instrumentation.weaviate.mapping import (
    MAPPING_V3,
    MAPPING_V4,
    SPAN_NAME_PREFIX,
)


class TestSpanAttributes:
    """Test span naming and attribute conventions."""

    def test_span_name_prefix(self):
        """Test that span name prefix is correct."""
        assert SPAN_NAME_PREFIX == "db.weaviate"

    def test_v3_span_names_have_prefix(self):
        """Test that all v3 span names follow naming convention."""
        for mapping in MAPPING_V3:
            span_name = mapping.get("span_name")
            assert span_name is not None
            # Span names should not include the prefix (it's added at runtime)
            assert not span_name.startswith("db.weaviate.")

    def test_v4_span_names_have_prefix(self):
        """Test that all v4 span names follow naming convention."""
        for mapping in MAPPING_V4:
            span_name = mapping.get("span_name")
            assert span_name is not None
            # Span names should not include the prefix (it's added at runtime)
            assert not span_name.startswith("db.weaviate.")

    def test_v3_mappings_have_required_fields(self):
        """Test that all v3 mappings have required fields."""
        required_fields = ["module", "name", "function", "span_name"]
        
        for mapping in MAPPING_V3:
            for field in required_fields:
                assert field in mapping, f"Missing {field} in mapping: {mapping}"
                assert mapping[field], f"Empty {field} in mapping: {mapping}"

    def test_v4_mappings_have_required_fields(self):
        """Test that all v4 mappings have required fields."""
        required_fields = ["module", "name", "function", "span_name"]
        
        for mapping in MAPPING_V4:
            for field in required_fields:
                assert field in mapping, f"Missing {field} in mapping: {mapping}"
                assert mapping[field], f"Empty {field} in mapping: {mapping}"

    def test_span_names_are_unique_v3(self):
        """Test that v3 span names are unique."""
        span_names = [m["span_name"] for m in MAPPING_V3]
        assert len(span_names) == len(set(span_names)), "Duplicate span names found in v3 mappings"

    def test_span_names_are_unique_v4(self):
        """Test that v4 span names are unique."""
        span_names = [m["span_name"] for m in MAPPING_V4]
        assert len(span_names) == len(set(span_names)), "Duplicate span names found in v4 mappings"

    def test_v3_operation_categories(self):
        """Test that v3 operations are properly categorized."""
        span_names = [m["span_name"] for m in MAPPING_V3]
        
        # Should have schema operations
        schema_ops = [s for s in span_names if s.startswith("schema.")]
        assert len(schema_ops) > 0, "No schema operations found"
        
        # Should have data operations
        data_ops = [s for s in span_names if "data" in s]
        assert len(data_ops) > 0, "No data operations found"
        
        # Should have query operations
        query_ops = [s for s in span_names if "query" in s or "gql" in s]
        assert len(query_ops) > 0, "No query operations found"
        
        # Should have batch operations
        batch_ops = [s for s in span_names if "batch" in s]
        assert len(batch_ops) > 0, "No batch operations found"

    def test_v4_operation_categories(self):
        """Test that v4 operations are properly categorized."""
        span_names = [m["span_name"] for m in MAPPING_V4]
        
        # Should have collection operations
        collection_ops = [s for s in span_names if s.startswith("collections.")]
        assert len(collection_ops) > 0, "No collection operations found"
        
        # Should have query operations
        query_ops = [s for s in span_names if "query" in s]
        assert len(query_ops) > 0, "No query operations found"
        
        # Should have data operations
        data_ops = [s for s in span_names if "data" in s]
        assert len(data_ops) > 0, "No data operations found"

    def test_module_paths_are_valid_python(self):
        """Test that module paths follow Python naming conventions."""
        all_mappings = MAPPING_V3 + MAPPING_V4
        
        for mapping in all_mappings:
            module = mapping["module"]
            # Should be valid Python module path
            assert module.replace(".", "").replace("_", "").isalnum(), \
                f"Invalid module path: {module}"
            # Should not start or end with dot
            assert not module.startswith("."), f"Module starts with dot: {module}"
            assert not module.endswith("."), f"Module ends with dot: {module}"
