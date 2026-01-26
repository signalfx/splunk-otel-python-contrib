# Tested with weaviate-client==3.26.7
# Code is adapted from official documentation.
# V3 documentation: https://weaviate.io/developers/weaviate/client-libraries/python/python_v3
# Some parts were also adapted from:
# https://towardsdatascience.com/getting-started-with-weaviate-python-client-e85d14f19e4f
import json
import os

import weaviate

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor
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

# Set up the tracer provider
tracer_provider = TracerProvider()
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

# Set up the meter provider for metrics
meter_provider = MeterProvider()
metrics.set_meter_provider(meter_provider)

# Add OTLP metric exporter
otlp_metric_exporter = OTLPMetricExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    headers=(),
)
otlp_metric_reader = PeriodicExportingMetricReader(
    otlp_metric_exporter,
    export_interval_millis=5000,  # Export every 5 seconds
)
meter_provider.add_metric_reader(otlp_metric_reader)

# Add console metric exporter to see metrics in terminal
console_metric_exporter = ConsoleMetricExporter()
console_metric_reader = PeriodicExportingMetricReader(
    console_metric_exporter,
    export_interval_millis=5000,
)
meter_provider.add_metric_reader(console_metric_reader)

# Now instrument Weaviate with both trace and metric providers
WeaviateInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)


def create_schema(client):
    client.schema.create_class(
        {
            "class": CLASS_NAME,
            "description": "An Article class to store a text",
            "vectorizer": "text2vec-ollama",
            "moduleConfig": {
                "text2vec-ollama": {
                    "apiEndpoint": "http://ollama:11434",
                    "model": "nomic-embed-text:latest",
                }
            },
            "properties": [
                {
                    "name": "author",
                    "dataType": ["string"],
                    "description": "The name of the author",
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The text content",
                },
            ],
        }
    )


def get_schema(client):
    """Get the schema to test connection"""
    return client.schema.get(CLASS_NAME)


def delete_schema(client):
    client.schema.delete_class(CLASS_NAME)


def create_object(client):
    return client.data_object.create(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name=CLASS_NAME,
    )


def create_batch(client):
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
    with client.batch as batch:
        for obj in objs:
            batch.add_data_object(obj, class_name=CLASS_NAME)


def query_get(client):
    return client.query.get(class_name=CLASS_NAME, properties=["author"]).do()


def query_aggregate(client):
    return client.query.aggregate(class_name=CLASS_NAME).with_meta_count().do()


def query_raw(client):
    return client.query.raw(RAW_QUERY)


def query_near_text(client, text):
    """Query using nearText to find similar articles with distance/certainty/score.

    Note: This requires a vectorizer (text2vec-ollama) to be configured in the schema,
    which is done in create_schema(). Requires Ollama to be running.
    """
    near_text_filter = {
        "concepts": ["lost while writing"],
    }

    query_result = (
        client.query.get(CLASS_NAME, ["text", "author"])
        .with_additional(["id", "distance", "certainty", "score"])
        .with_near_text(near_text_filter)
        .do()
    )

    return query_result


def validate(client, uuid=None):
    return client.data_object.validate(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        uuid=uuid,
        class_name=CLASS_NAME,
    )


def delete_all(client):
    client.schema.delete_all()


def example_schema_workflow(client):
    delete_all(client)

    create_schema(client)
    print("Created schema")
    schema = get_schema(client)
    print("Retrieved schema: ", json.dumps(schema, indent=2))
    result = validate(client)
    print(f"Object found: {result.get('valid')}")

    uuid = create_object(client)
    print("Created object of UUID: ", uuid)
    client.data_object.exists(uuid, class_name=CLASS_NAME)
    obj = client.data_object.get(uuid, class_name=CLASS_NAME)
    print("Retrieved obj: ", json.dumps(obj, indent=2))
    result = validate(client, uuid=uuid)
    print(f"Object found: {result.get('valid')}")

    create_batch(client)
    result = query_get(client)
    print("Query result:", json.dumps(result, indent=2))
    aggregate_result = query_aggregate(client)
    print("Aggregate result:", json.dumps(aggregate_result, indent=2))
    raw_result = query_raw(client)
    print("Raw result: ", json.dumps(raw_result, indent=2))

    # Try near_text query (requires vectorizer and Ollama)
    try:
        near_text_result = query_near_text(client, "book")
        print("Near text result: ", json.dumps(near_text_result, indent=2))
    except Exception as e:
        print(
            f"Near text query skipped (requires Ollama running at http://ollama:11434): {e}"
        )

    delete_schema(client)
    print("Deleted schema")


if __name__ == "__main__":
    print("OpenTelemetry Weaviate instrumentation initialized")

    additional_headers = {}
    if (key := os.getenv("COHERE_API_KEY")) is not None:
        additional_headers.update({"X-Cohere-Api-Key": key})
    elif (key := os.getenv("OPENAI_API_KEY")) is not None:
        additional_headers.update({"X-OpenAI-Api-Key": key})
    else:
        print("Warning: No API key found for Cohere or OpenAI")

    if (cluster_name := os.getenv("WEAVIATE_CLUSTER_URL")) is not None:
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_client_secret=auth_config,
            timeout_config=(5, 15),
            additional_headers=additional_headers,
        )
    else:
        client = weaviate.Client(
            url="http://localhost:8080",
            additional_headers=additional_headers,
        )
    print("Client connected")

    try:
        example_schema_workflow(client)
    finally:
        # Ensure all spans are exported before exiting
        tracer_provider.force_flush(timeout_millis=5000)
