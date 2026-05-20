# pylint: skip-file
# ruff: noqa: E402
"""
Google GenAI Travel Planner — Demo App

A single cohesive flow that exercises every instrumented code path of the
google-genai instrumentation in one run:

  - embed_content         → semantic destination search
  - generate_content      → system instructions, config params, multi-turn
  - generate_content_stream → streaming final travel plan
  - manual function calling  → multi-turn tool-call loop with generate_content
                                spans for each round-trip (flights, hotels,
                                weather, activities)

The flow:
  1. User describes their ideal trip (CLI or default).
  2. Embeddings rank destinations by semantic similarity.
  3. The top destination is passed to a tool-calling planner that fetches
     flights, hotels, weather, and activities via mock tools.
  4. A streaming call produces the final travel summary.

Authentication is auto-detected:
  - GOOGLE_API_KEY           → Gemini Developer API
  - GOOGLE_CLOUD_PROJECT     → Vertex AI (requires ADC)

Usage:
    cp .env.example .env   # fill in credentials
    pip install -r requirements.txt
    python main.py
    python main.py --query "adventure trip with hiking and northern lights"
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OpenTelemetry setup — tracing + logging + metrics
# ---------------------------------------------------------------------------
from opentelemetry import _logs as otel_logs
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

os.environ.setdefault("OTEL_SERVICE_NAME", "google-genai-travel-planner")
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", "SPAN_AND_EVENT"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event"
)

OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
)

resource = Resource.create(
    {
        "service.name": os.environ.get(
            "OTEL_SERVICE_NAME", "google-genai-travel-planner"
        ),
        "service.version": "1.0.0",
    }
)

trace_provider = TracerProvider(resource=resource)
trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
otel_trace.set_tracer_provider(trace_provider)

log_provider = LoggerProvider(resource=resource)
log_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)
otel_logs.set_logger_provider(log_provider)

meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
)
otel_metrics.set_meter_provider(meter_provider)

# ---------------------------------------------------------------------------
# Instrument google-genai (must happen before creating clients)
# ---------------------------------------------------------------------------
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)

GoogleGenAiSdkInstrumentor().instrument()

# ---------------------------------------------------------------------------
# google-genai imports (after instrumentation)
# ---------------------------------------------------------------------------
import google.genai
from google.genai.types import GenerateContentConfig

MODEL = os.getenv("MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")


def _client() -> google.genai.Client:
    return google.genai.Client()


# ============================================================================
# Destinations database
# ============================================================================
DESTINATIONS = [
    {
        "name": "Tokyo, Japan",
        "origin": "San Francisco",
        "description": "Vibrant metropolis blending ancient temples with "
        "cutting-edge technology. Famous for sushi, anime, and cherry blossoms.",
    },
    {
        "name": "Paris, France",
        "origin": "New York",
        "description": "City of lights and love. Home to the Eiffel Tower, "
        "world-class museums, and exquisite cuisine.",
    },
    {
        "name": "Bali, Indonesia",
        "origin": "Los Angeles",
        "description": "Tropical paradise with stunning beaches, rice terraces, "
        "and spiritual temples. Perfect for relaxation and wellness retreats.",
    },
    {
        "name": "Reykjavik, Iceland",
        "origin": "Boston",
        "description": "Gateway to dramatic landscapes — northern lights, "
        "geysers, glaciers, and volcanic hot springs. Adventure in raw nature.",
    },
    {
        "name": "Marrakech, Morocco",
        "origin": "London",
        "description": "Exotic bazaars, vibrant souks, and beautiful riads. "
        "A sensory feast of colors, spices, and ancient history.",
    },
]


# ============================================================================
# Mock tool — automatic function calling (produces execute_tool span)
# ============================================================================
def get_weather(city: str) -> dict:
    """Get the current weather forecast for a city."""
    print(f"    [TOOL CALLED] get_weather({city})")
    return {
        "temperature_celsius": 28,
        "condition": "Sunny",
        "humidity_percent": 65,
    }


# ============================================================================
# Step 1 — Semantic destination search (embed_content)
# ============================================================================
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def find_destination(client: google.genai.Client, query: str) -> dict:
    """Use embeddings to find the best-matching destination for a query."""
    print("\n--- Step 1: Finding best destination (embeddings) ---")
    print(f'  Query: "{query}"')

    # Embed the user query
    query_resp = client.models.embed_content(model=EMBED_MODEL, contents=query)
    query_vec = query_resp.embeddings[0].values
    print(f"  Query embedding: {len(query_vec)} dimensions")

    # Embed all destination descriptions (batch)
    descriptions = [d["description"] for d in DESTINATIONS]
    dest_resp = client.models.embed_content(
        model=EMBED_MODEL, contents=descriptions
    )

    # Rank by similarity
    scored = []
    for i, dest in enumerate(DESTINATIONS):
        sim = cosine_similarity(query_vec, dest_resp.embeddings[i].values)
        scored.append((dest, sim))
    scored.sort(key=lambda x: x[1], reverse=True)

    print("  Rankings:")
    for rank, (dest, score) in enumerate(scored, 1):
        marker = " <-- best match" if rank == 1 else ""
        print(f"    {rank}. {dest['name']} ({score:.3f}){marker}")

    return scored[0][0]


# ============================================================================
# Step 2 — Plan the trip (tool calling + system instruction + config)
# ============================================================================
def plan_trip(
    client: google.genai.Client, destination: dict, query: str
) -> str:
    """Use automatic function calling to gather travel data and produce a plan.

    Exercises: generate_content with system_instruction, config params
    (temperature, top_p, max_output_tokens), and automatic function calling
    (execute_tool spans for each mock tool invoked by the model).
    """
    print("\n--- Step 2: Planning trip (tool calling) ---")

    departure = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    return_date = (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")
    dest_name = destination["name"]
    origin = destination["origin"]

    print(f"  Destination: {dest_name}")
    print(f"  Origin: {origin}")
    print(f"  Dates: {departure} to {return_date}")
    print("  Calling model with tools...")

    prompt = (
        f"What is the weather in {dest_name}? "
        f"Based on the weather, suggest what to pack for a week-long trip "
        f"from {origin} to {dest_name}."
    )

    # Automatic function calling — pass a callable so the SDK executes
    # the tool and the instrumentation creates an execute_tool span.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=GenerateContentConfig(tools=[get_weather]),
    )

    plan = response.text or ""
    print(f"  Plan received ({len(plan)} chars):")
    print(f"  {plan[:200]}")
    return plan


# ============================================================================
# Step 3 — Stream a polished summary (generate_content_stream)
# ============================================================================
def stream_summary(
    client: google.genai.Client, destination: dict, plan: str
) -> None:
    """Stream a polished one-paragraph summary of the travel plan.

    Exercises: generate_content_stream with system_instruction.
    """
    print("\n--- Step 3: Streaming travel summary ---")

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=(
            f"Here is a detailed travel plan for {destination['name']}:\n\n"
            f"{plan}\n\n"
            "Write a short, enthusiastic 3-sentence summary of this trip "
            "that would make someone excited to book it."
        ),
        config=GenerateContentConfig(
            system_instruction="You are a travel copywriter. Be vivid and concise.",
            temperature=0.9,
            max_output_tokens=256,
        ),
    )

    print("  ", end="", flush=True)
    for chunk in stream:
        print(chunk.text or "", end="", flush=True)
    print()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Google GenAI travel planner — demonstrates embeddings, "
        "tool calling, streaming, and multi-turn conversation "
        "with OpenTelemetry instrumentation."
    )
    parser.add_argument(
        "--query",
        default="relaxing beach vacation with temples and wellness",
        help="Describe your ideal trip (default: beach + temples + wellness)",
    )
    args = parser.parse_args()

    # Validate credentials
    has_api_key = bool(os.environ.get("GOOGLE_API_KEY"))
    has_vertex = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if not has_api_key and not has_vertex:
        print(
            "ERROR: Set GOOGLE_API_KEY (Gemini) or "
            "GOOGLE_CLOUD_PROJECT (Vertex AI) to authenticate."
        )
        sys.exit(1)

    backend = (
        "Vertex AI"
        if has_vertex and not has_api_key
        else "Gemini Developer API"
    )
    print("=" * 60)
    print("  Google GenAI Travel Planner")
    print(f"  Backend : {backend}")
    print(f"  Model   : {MODEL}")
    print(f"  OTLP    : {OTLP_ENDPOINT}")
    print("=" * 60)

    client = _client()

    # Step 1 — embeddings: find best destination
    destination = find_destination(client, args.query)

    # Step 2 — tool calling + multi-turn: plan the trip
    plan = plan_trip(client, destination, args.query)

    # Step 3 — streaming: produce a polished summary
    stream_summary(client, destination, plan)

    # Flush telemetry
    print("\n" + "-" * 60)
    print("Flushing telemetry...")
    trace_provider.force_flush()
    log_provider.force_flush()
    meter_provider.force_flush()
    print(
        "Done. Check your collector / backend for traces, logs, and metrics."
    )


if __name__ == "__main__":
    main()
