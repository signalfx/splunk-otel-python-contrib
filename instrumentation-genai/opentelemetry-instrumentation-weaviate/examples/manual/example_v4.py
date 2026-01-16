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

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
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
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317"),
    headers=(),
)
otlp_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(otlp_processor)

# Add console exporter to see traces in terminal as well
console_exporter = ConsoleSpanExporter()
console_processor = BatchSpanProcessor(console_exporter)
tracer_provider.add_span_processor(console_processor)

# Set up metric exporters and readers
otlp_metric_exporter = OTLPMetricExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317"),
    headers=(),
)
otlp_metric_reader = PeriodicExportingMetricReader(
    otlp_metric_exporter,
    export_interval_millis=5000,  # Export every 5 seconds
)

console_metric_exporter = ConsoleMetricExporter()
console_metric_reader = PeriodicExportingMetricReader(
    console_metric_exporter,
    export_interval_millis=5000,
)

# Create meter provider with metric readers
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[otlp_metric_reader, console_metric_reader],
)
metrics.set_meter_provider(meter_provider)

# Now instrument Weaviate with both trace and metric providers
WeaviateInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)


def create_schema(client):
    """Create a collection.

    Note: Vectorizer configuration is optional. If text2vec-ollama module is available,
    you can enable it by setting vectorizer_config parameter.
    """
    # Try to create with vectorizer first, fall back to none if module not available
    try:
        client.collections.create(
            name=CLASS_NAME,
            description="An Article class to store a text",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_ollama(
                api_endpoint="http://ollama:11434",
                model="nomic-embed-text:latest",
            ),
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
        print("Collection created with text2vec-ollama vectorizer")
    except Exception as e:
        print(f"Could not create with vectorizer ({e}), creating without vectorizer...")
        # Create without vectorizer
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
        print("Collection created without vectorizer (near_text queries will not work)")


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


def query_fetch_objects(collection):
    """Fetch objects using fetch_objects method.

    Note: This internally calls _QueryGRPC.get() which will generate
    both 'collections.query.fetch_objects' and 'collections.query.get' spans.
    """
    return collection.query.fetch_objects(
        limit=5,
        return_properties=[
            "author",
            "text",
        ],
    )


def query_near_text(collection):
    """Query using near_text to find similar articles with distance/certainty/score.

    Note: This requires a vectorizer (text2vec-ollama) to be configured in the schema,
    which is done in create_schema(). Requires Ollama to be running.
    """
    return collection.query.near_text(
        query="lost while writing",
        limit=3,
        return_metadata=["distance", "certainty", "score"],
        return_properties=["author", "text"],
    )


# TODO: Not instrumented
def query_aggregate(collection):
    """Query aggregate statistics over all objects in the collection."""
    return collection.aggregate.over_all(total_count=True)


def query_raw(client):
    """Execute a raw GraphQL query."""
    return client.graphql_raw_query(RAW_QUERY)


def validate(collection, uuid=None):
    """Validate by attempting to fetch an object by ID."""
    if uuid:
        return collection.query.fetch_object_by_id(uuid)
    return None


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

    # Query objects (internally calls both fetch_objects and get)
    result = query_fetch_objects(collection)
    print("Query result:", result)

    # Try near_text query (requires vectorizer and Ollama)
    try:
        near_text_result = query_near_text(collection)
        print("Near text result:", near_text_result)
    except Exception as e:
        print(
            f"Near text query skipped (requires Ollama running at http://ollama:11434): {e}"
        )

    aggregate_result = query_aggregate(collection)
    print("Aggregate result:", aggregate_result)
    raw_result = query_raw(client)
    print("Raw result: ", raw_result)

    delete_collection(client)
    print("Deleted schema")


if __name__ == "__main__":
    print("OpenTelemetry Weaviate instrumentation initialized")

    # Connect to local Weaviate instance (default: http://localhost:8080)
    # Make sure Weaviate is running locally, e.g., via Docker:
    # docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest

    client = weaviate.connect_to_local()
    print("Connected to local Weaviate instance")

    try:
        example_schema_workflow(client)
    finally:
        # Ensure all spans are exported before exiting
        tracer_provider.force_flush(timeout_millis=5000)
        client.close()
