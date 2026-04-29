# pylint: skip-file
"""VertexAI Single-Agent Example.

Demonstrates a ReAct-style agentic loop using the native VertexAI SDK
(GenerativeModel + tool calling) with automatic OpenTelemetry instrumentation
via VertexAIInstrumentor.

Run modes:
1. Default (no CLI args): queries "What is the weather in San Francisco?" and exits.
2. CLI mode: python main.py --city "Paris"

Required environment variables:
    GOOGLE_CLOUD_PROJECT   - GCP project ID
    GOOGLE_CLOUD_LOCATION  - GCP region (default: us-central1)
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP collector endpoint (default: http://localhost:4317)

Optional:
    GOOGLE_APPLICATION_CREDENTIALS - path to service-account JSON for ADC
"""

import argparse
import os

import requests
import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

# NOTE: OpenTelemetry Python Logs and Events APIs are in beta
from opentelemetry import _logs, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()

# --- OTEL environment defaults ---
os.environ.setdefault("OTEL_SERVICE_NAME", "vertexai-single-agent")
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
)

# --- Configure tracing ---
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# --- Configure logging / events ---
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# --- Instrument VertexAI ---
VertexAIInstrumentor().instrument()


# ---------------------------------------------------------------------------
# Tool implementation — uses the free Open-Meteo API (no key required)
# ---------------------------------------------------------------------------


def get_weather(location: str) -> dict:
    """Get current weather for a city using the Open-Meteo geocoding and weather APIs."""
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(
        geocode_url,
        params={
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    results = geo_data.get("results")
    if not results:
        return {"error": f"Location '{location}' not found"}

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    name = results[0].get("name", location)

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(
        weather_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=10,
    )
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()

    current = weather_data.get("current_weather", {})
    return {
        "location": name,
        "temperature_celsius": current.get("temperature"),
        "wind_speed_kmh": current.get("windspeed"),
        "weather_code": current.get("weathercode"),
    }


# Map tool name → callable for dispatch
_TOOLS = {"get_weather": get_weather}

# FunctionDeclaration for the model
_get_weather_decl = FunctionDeclaration(
    name="get_weather",
    description="Get the current weather for a given city or location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for, e.g. 'San Francisco'",
            }
        },
        "required": ["location"],
    },
)

_weather_tool = Tool(function_declarations=[_get_weather_decl])


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(query: str) -> str:
    """Run a single-turn agentic query against Gemini with tool calling."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    vertexai.init(project=project, location=location)

    model = GenerativeModel(
        "gemini-2.5-flash-lite",
        tools=[_weather_tool],
        system_instruction=(
            "You are a helpful weather assistant. "
            "Use the get_weather tool to fetch real weather data when asked about weather. "
            "After presenting the data, add a short commentary about the conditions."
        ),
    )

    chat = model.start_chat()
    response = chat.send_message(query)

    # ReAct loop: keep dispatching tool calls until the model produces a text reply
    while response.candidates and response.candidates[0].function_calls:
        fc = response.candidates[0].function_calls[0]
        tool_fn = _TOOLS.get(fc.name)
        if tool_fn is None:
            tool_result = {"error": f"Unknown tool: {fc.name}"}
        else:
            try:
                tool_result = tool_fn(**{k: v for k, v in fc.args.items()})
            except Exception as exc:  # noqa: BLE001
                tool_result = {"error": str(exc)}

        response = chat.send_message(
            Part.from_function_response(
                name=fc.name, response={"output": tool_result}
            )
        )

    return response.text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VertexAI single-agent weather demo with OpenTelemetry instrumentation"
    )
    parser.add_argument(
        "--city",
        default="San Francisco",
        help="City to query weather for (default: San Francisco)",
    )
    args = parser.parse_args()

    query = f"What is the weather in {args.city}?"
    print(f"Query: {query}\n")
    result = run_agent(query)
    print(result)


if __name__ == "__main__":
    main()
