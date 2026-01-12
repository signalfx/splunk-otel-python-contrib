"""Tests for span attributes and semantic conventions."""

import pytest

try:
    import weaviate
    import weaviate.classes as wvc

    WEAVIATE_AVAILABLE = True
    WEAVIATE_VERSION = int(weaviate.__version__.split(".")[0])
except ImportError:
    WEAVIATE_AVAILABLE = False
    WEAVIATE_VERSION = 0

from opentelemetry import trace
from opentelemetry.instrumentation.weaviate.mapping import (
    MAPPING_V3,
    MAPPING_V4,
    SPAN_NAME_PREFIX,
)
from opentelemetry.semconv.attributes import (
    db_attributes as DbAttributes,
    server_attributes as ServerAttributes,
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
        assert len(span_names) == len(
            set(span_names)
        ), "Duplicate span names found in v3 mappings"

    def test_span_names_are_unique_v4(self):
        """Test that v4 span names are unique."""
        span_names = [m["span_name"] for m in MAPPING_V4]
        assert len(span_names) == len(
            set(span_names)
        ), "Duplicate span names found in v4 mappings"

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
            assert (
                module.replace(".", "").replace("_", "").isalnum()
            ), f"Invalid module path: {module}"
            # Should not start or end with dot
            assert not module.startswith("."), f"Module starts with dot: {module}"
            assert not module.endswith("."), f"Module ends with dot: {module}"

    def test_v3_all_operations_covered(self):
        """Test that all major v3 operations are covered."""
        span_names = [m["span_name"] for m in MAPPING_V3]

        # Schema operations
        assert "schema.get" in span_names
        assert "schema.create_class" in span_names
        assert "schema.delete_class" in span_names
        assert "schema.delete_all" in span_names

        # Data CRUD operations
        assert "data.crud_data.create" in span_names
        assert "data.crud_data.get" in span_names
        assert "data.crud_data.validate" in span_names

        # Batch operations
        assert "batch.crud_batch.add_data_object" in span_names
        assert "batch.crud_batch.flush" in span_names

        # Query operations
        assert "gql.query.get" in span_names
        assert "gql.query.aggregate" in span_names
        assert "gql.query.raw" in span_names
        assert "gql.query.get.do" in span_names

    def test_v4_all_operations_covered(self):
        """Test that all major v4 operations are covered."""
        span_names = [m["span_name"] for m in MAPPING_V4]

        # Collection management
        assert "collections.get" in span_names
        assert "collections.create" in span_names
        assert "collections.delete" in span_names
        assert "collections.delete_all" in span_names
        assert "collections.create_from_dict" in span_names

        # Data operations
        assert "collections.data.insert" in span_names
        assert "collections.data.replace" in span_names
        assert "collections.data.update" in span_names

        # Query operations
        assert "collections.query.near_text" in span_names
        assert "collections.query.near_vector" in span_names
        assert "collections.query.fetch_objects" in span_names
        assert "collections.query.get" in span_names

        # Batch operations
        assert "collections.batch.add_object" in span_names


class TestOperationSpanAttributes:
    """Test that operations create spans with correct attributes."""

    def test_span_has_required_db_attributes(self):
        """Test that spans should have db.system.name and db.operation.name."""
        # All operations should set these attributes
        # db.system.name = "weaviate"
        # db.operation.name = function name
        pass

    def test_span_has_server_attributes(self):
        """Test that spans should include server.address and server.port when available."""
        # Server attributes are extracted from connection URL
        # server.address = hostname
        # server.port = port number
        pass

    def test_collection_name_attribute(self):
        """Test that collection/class name should be captured in db.weaviate.collection.name."""
        # For operations on collections/classes, should include:
        # db.weaviate.collection.name = collection/class name
        pass

    def test_similarity_search_attributes(self):
        """Test that similarity search operations should capture additional attributes."""
        # Should capture:
        # - db.weaviate.documents.count (span attribute)
        # - db.weaviate.document.content (event attribute)
        # - db.weaviate.document.distance (event attribute, if present)
        # - db.weaviate.document.certainty (event attribute, if present)
        # - db.weaviate.document.score (event attribute, if present)
        # - db.weaviate.document.query (event attribute, if present)
        pass


@pytest.mark.integration
@pytest.mark.skipif(
    not WEAVIATE_AVAILABLE or WEAVIATE_VERSION < 4,
    reason="Weaviate v4 client not available",
)
class TestV4IntegrationAttributes:
    """Integration tests for v4 span attributes with actual Weaviate operations."""

    @pytest.fixture(scope="class")
    def weaviate_client(self):
        """Create a Weaviate v4 client for testing."""
        try:
            client = weaviate.connect_to_local()
            yield client
            client.close()
        except Exception as e:
            pytest.skip(f"Weaviate server not available: {e}")

    def test_collection_create_span_attributes(
        self, instrumentor, tracer_provider, span_exporter, weaviate_client
    ):
        """Test that collections.create generates correct span attributes."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create a collection
        try:
            weaviate_client.collections.create(
                name="TestCollection",
                description="Test collection for span attributes",
            )
        finally:
            try:
                weaviate_client.collections.delete("TestCollection")
            except Exception:
                pass

        spans = span_exporter.get_finished_spans()
        create_spans = [s for s in spans if "collections.create" in s.name]

        assert len(create_spans) > 0, "No collections.create span found"

        span = create_spans[0]
        attributes = dict(span.attributes or {})

        # Verify required attributes
        assert attributes.get(DbAttributes.DB_SYSTEM_NAME) == "weaviate"
        assert attributes.get(DbAttributes.DB_OPERATION_NAME) == "create"
        assert ServerAttributes.SERVER_ADDRESS in attributes
        assert ServerAttributes.SERVER_PORT in attributes

    def test_data_insert_span_attributes(
        self, instrumentor, tracer_provider, span_exporter, weaviate_client
    ):
        """Test that collections.data.insert generates correct span attributes."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Setup: Create collection
        try:
            weaviate_client.collections.delete("TestInsert")
        except Exception:
            pass

        weaviate_client.collections.create(
            name="TestInsert",
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )

        span_exporter.clear()

        # Test: Insert data
        collection = weaviate_client.collections.get("TestInsert")
        collection.data.insert({"text": "test content"})

        spans = span_exporter.get_finished_spans()
        insert_spans = [s for s in spans if "data.insert" in s.name]

        assert len(insert_spans) > 0, "No data.insert span found"

        span = insert_spans[0]
        attributes = dict(span.attributes or {})

        # Verify required attributes
        assert attributes.get(DbAttributes.DB_SYSTEM_NAME) == "weaviate"
        assert attributes.get(DbAttributes.DB_OPERATION_NAME) == "insert"
        assert attributes.get("db.weaviate.collection.name") == "TestInsert"
        assert ServerAttributes.SERVER_ADDRESS in attributes
        assert ServerAttributes.SERVER_PORT in attributes

        # Cleanup
        weaviate_client.collections.delete("TestInsert")

    def test_fetch_objects_span_attributes(
        self, instrumentor, tracer_provider, span_exporter, weaviate_client
    ):
        """Test that collections.query.fetch_objects generates correct span attributes."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Setup: Create collection and insert data
        try:
            weaviate_client.collections.delete("TestQuery")
        except Exception:
            pass

        weaviate_client.collections.create(
            name="TestQuery",
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )
        collection = weaviate_client.collections.get("TestQuery")
        collection.data.insert({"text": "test content"})

        span_exporter.clear()

        # Test: Fetch objects
        collection.query.fetch_objects(limit=1)

        spans = span_exporter.get_finished_spans()
        fetch_spans = [s for s in spans if "fetch_objects" in s.name]

        assert len(fetch_spans) > 0, "No fetch_objects span found"

        span = fetch_spans[0]
        attributes = dict(span.attributes or {})

        # Verify required attributes
        assert attributes.get(DbAttributes.DB_SYSTEM_NAME) == "weaviate"
        assert attributes.get(DbAttributes.DB_OPERATION_NAME) == "fetch_objects"
        assert attributes.get("db.weaviate.collection.name") == "TestQuery"
        assert ServerAttributes.SERVER_ADDRESS in attributes
        assert ServerAttributes.SERVER_PORT in attributes

        # Cleanup
        weaviate_client.collections.delete("TestQuery")

    def test_fetch_objects_creates_nested_get_span(
        self, instrumentor, tracer_provider, span_exporter, weaviate_client
    ):
        """Test that fetch_objects creates a nested collections.query.get span."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Setup
        try:
            weaviate_client.collections.delete("TestNested")
        except Exception:
            pass

        weaviate_client.collections.create(
            name="TestNested",
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )
        collection = weaviate_client.collections.get("TestNested")
        collection.data.insert({"text": "test"})

        span_exporter.clear()

        # Test
        collection.query.fetch_objects(limit=1)

        spans = span_exporter.get_finished_spans()

        # Should have both fetch_objects and get spans
        fetch_span = next((s for s in spans if "fetch_objects" in s.name), None)
        get_span = next((s for s in spans if s.name.endswith("query.get")), None)

        assert fetch_span is not None, "No fetch_objects span found"
        assert get_span is not None, "No query.get span found"

        # get span should be a child of fetch_objects span
        if get_span and fetch_span:
            assert get_span.start_time >= fetch_span.start_time
            assert get_span.end_time <= fetch_span.end_time

        # Cleanup
        weaviate_client.collections.delete("TestNested")


@pytest.mark.integration
@pytest.mark.skipif(
    not WEAVIATE_AVAILABLE or WEAVIATE_VERSION >= 4,
    reason="Weaviate v3 client not available",
)
class TestV3IntegrationAttributes:
    """Integration tests for v3 span attributes with actual Weaviate operations."""

    @pytest.fixture(scope="class")
    def weaviate_client(self):
        """Create a Weaviate v3 client for testing."""
        try:
            client = weaviate.Client("http://localhost:8080")
            yield client
        except Exception as e:
            pytest.skip(f"Weaviate server not available: {e}")

    def test_schema_create_span_attributes(
        self, instrumentor, tracer_provider, span_exporter, weaviate_client
    ):
        """Test that schema.create_class generates correct span attributes."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create a schema
        try:
            weaviate_client.schema.create_class(
                {
                    "class": "TestClass",
                    "description": "Test class for span attributes",
                }
            )
        finally:
            try:
                weaviate_client.schema.delete_class("TestClass")
            except Exception:
                pass

        spans = span_exporter.get_finished_spans()
        create_spans = [s for s in spans if "schema.create_class" in s.name]

        assert len(create_spans) > 0, "No schema.create_class span found"

        span = create_spans[0]
        attributes = dict(span.attributes or {})

        # Verify required attributes
        assert attributes.get(DbAttributes.DB_SYSTEM_NAME) == "weaviate"
        assert attributes.get(DbAttributes.DB_OPERATION_NAME) == "create_class"
        assert ServerAttributes.SERVER_ADDRESS in attributes
        assert ServerAttributes.SERVER_PORT in attributes

    def test_data_create_span_attributes(
        self, instrumentor, tracer_provider, span_exporter, weaviate_client
    ):
        """Test that data.crud_data.create generates correct span attributes."""
        trace.set_tracer_provider(tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Setup: Create schema
        try:
            weaviate_client.schema.delete_class("TestData")
        except Exception:
            pass

        weaviate_client.schema.create_class(
            {
                "class": "TestData",
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                    }
                ],
            }
        )

        span_exporter.clear()

        # Test: Create data object
        weaviate_client.data_object.create(
            data_object={"text": "test content"},
            class_name="TestData",
        )

        spans = span_exporter.get_finished_spans()
        create_spans = [s for s in spans if "crud_data.create" in s.name]

        assert len(create_spans) > 0, "No crud_data.create span found"

        span = create_spans[0]
        attributes = dict(span.attributes or {})

        # Verify required attributes
        assert attributes.get(DbAttributes.DB_SYSTEM_NAME) == "weaviate"
        assert attributes.get(DbAttributes.DB_OPERATION_NAME) == "create"
        assert attributes.get("db.weaviate.collection.name") == "TestData"
        assert ServerAttributes.SERVER_ADDRESS in attributes
        assert ServerAttributes.SERVER_PORT in attributes

        # Cleanup
        weaviate_client.schema.delete_class("TestData")
