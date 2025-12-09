"""
Example demonstrating OpenTelemetry instrumentation for Weaviate v4 client.

This example shows various Weaviate operations including schema management,
data operations, and queries. All operations are automatically instrumented.

For setup instructions, see ../../README.rst

Tested with weaviate-client>=4.0.0
Code adapted from: https://weaviate.io/developers/weaviate/client-libraries/python
"""

import os

import weaviate
import weaviate.classes as wvc

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.semconv.resource import ResourceAttributes

CLASS_NAME = "Article"
RAW_QUERY = """
 {
   Get {
     Article(limit: 2) {
        author
        text
     }
   }
 }
 """

# Set up the tracer provider with service name
resource = Resource(
    attributes={
        ResourceAttributes.SERVICE_NAME: "weaviate-example",
    }
)
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Add OTLP exporter (reads from OTEL_EXPORTER_OTLP_ENDPOINT env var)
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    headers=(),
)
otlp_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(otlp_processor)

# Add console exporter to see traces in terminal as well
console_exporter = ConsoleSpanExporter()
console_processor = BatchSpanProcessor(console_exporter)
tracer_provider.add_span_processor(console_processor)

# Now instrument Weaviate
WeaviateInstrumentor().instrument()


def create_schema(client):
    client.collections.create(
        name=CLASS_NAME,
        description="An Article class to store a text",
        properties=[
            wvc.config.Property(
                name="author",
                data_type=wvc.config.DataType.TEXT,
                description="The name of the author",
            ),
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
                description="The text content",
            ),
        ],
    )


def get_collection(client):
    """Get the collection to test connection"""
    return client.collections.get(CLASS_NAME)


def delete_collection(client):
    client.collections.delete(CLASS_NAME)


def create_object(collection):
    return collection.data.insert(
        {
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        }
    )


def create_batch(collection):
    objs = [
        {
            "author": "Robert",
            "text": "Once upon a time, R. wrote a book...",
        },
        {
            "author": "Johnson",
            "text": "Once upon a time, J. wrote some news...",
        },
        {
            "author": "Maverick",
            "text": "Never again, M. will write a book...",
        },
        {
            "author": "Wilson",
            "text": "Lost in the island, W. did not write anything...",
        },
        {
            "author": "Ludwig",
            "text": "As king, he ruled...",
        },
    ]
    with collection.batch.dynamic() as batch:
        for obj in objs:
            batch.add_object(properties=obj)


def query_get(collection):
    return collection.query.fetch_objects(
        limit=5,
        return_properties=[
            "author",
            "text",
        ],
    )


def query_aggregate(collection):
    return collection.aggregate.over_all(total_count=True)


def query_raw(client):
    return client.graphql_raw_query(RAW_QUERY)


def validate(collection, uuid=None):
    """Validate by attempting to fetch an object by ID."""
    if uuid:
        return collection.query.fetch_object_by_id(uuid)
    return None


def create_schemas(client):
    client.collections.create_from_dict(
        {
            "class": "Author",
            "description": "An author that writes an article",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["string"],
                    "description": "The name of the author",
                },
            ],
        },
    )
    client.collections.create_from_dict(
        {
            "class": CLASS_NAME,
            "description": "An Article class to store a text",
            "properties": [
                {
                    "name": "author",
                    "dataType": ["Author"],
                    "description": "The author",
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The text content",
                },
            ],
        },
    )


def delete_all(client):
    client.collections.delete_all()


def example_schema_workflow(client):
    delete_all(client)

    create_schema(client)
    print("Created schema")
    collection = get_collection(client)
    print("Retrieved collection: ", collection.name)

    uuid = create_object(collection)
    print("Created object of UUID: ", uuid)
    obj = collection.query.fetch_object_by_id(uuid)
    print("Retrieved obj: ", obj)

    create_batch(collection)
    result = query_get(collection)
    print("Query result:", result)
    aggregate_result = query_aggregate(collection)
    print("Aggregate result:", aggregate_result)
    raw_result = query_raw(client)
    print("Raw result: ", raw_result)

    delete_collection(client)
    print("Deleted schema")


def example_schema_workflow2(client):
    delete_all(client)
    create_schemas(client)


if __name__ == "__main__":
    print("OpenTelemetry Weaviate instrumentation initialized")

    # Connect to local Weaviate instance (default: http://localhost:8080)
    # Make sure Weaviate is running locally, e.g., via Docker:
    # docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest

    client = weaviate.connect_to_local()
    print("Connected to local Weaviate instance")

    try:
        example_schema_workflow2(client)
        example_schema_workflow(client)
        delete_all(client)
    finally:
        # Ensure all spans are exported before exiting
        tracer_provider.force_flush(timeout_millis=5000)
        client.close()
