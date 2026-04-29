#!/usr/bin/env python3
"""
Weather MCP Server

Provides weather and travel packing tools via the Model Context Protocol.
Supports both stdio (subprocess) and HTTP (streamable-http) transports.

Usage:
    # Load env vars
    source .env

    # --- stdio mode (spawned as subprocess by weather_agent.py) ---
    source .env
    OTEL_SERVICE_NAME=weather-mcp-server python weather_server.py

    # --- HTTP mode (standalone server, client connects over HTTP) ---
    source .env
    OTEL_SERVICE_NAME=weather-mcp-server python weather_server.py --transport http
    OTEL_SERVICE_NAME=weather-mcp-server python weather_server.py --transport http --port 8001

    # --- Zero-code instrumentation (explicit) ---
    OTEL_SERVICE_NAME=weather-mcp-server opentelemetry-instrument python weather_server.py --transport http

Telemetry is set up automatically when OTEL_EXPORTER_OTLP_ENDPOINT is in the environment
(loaded from .env). Pass --manual to force in-process setup regardless.
OTEL_SERVICE_NAME defaults to 'weather-mcp-server'.
"""

import os
import sys

from _otel_helpers import load_dotenv as _load_dotenv
from _otel_helpers import providers_already_configured as _providers_already_configured

_load_dotenv()


def setup_server_telemetry():
    """Configure OTel providers and FastMCP instrumentation for the server."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    service_name = os.environ.get("OTEL_SERVICE_NAME", "weather-mcp-server")
    resource = Resource.create({"service.name": service_name})

    trace_provider = TracerProvider(resource=resource)
    metric_readers = []

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            print(f"🔭 OTLP exporters enabled → {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print("⚠️  OTLP exporters not available in server", file=sys.stderr)
    else:
        print(
            "⚠️  OTEL_EXPORTER_OTLP_ENDPOINT not set — server traces will NOT reach Splunk.",
            file=sys.stderr,
        )

    if os.environ.get("OTEL_SERVER_CONSOLE_EXPORT"):
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    print(
        f"🔭 service.name={service_name}  emitters={emitters}",
        file=sys.stderr,
    )

    trace.set_tracer_provider(trace_provider)
    if metric_readers:
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=metric_readers)
        )

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()


# Set up telemetry before FastMCP is imported/instantiated so that
# instrumentation is applied to the server object.
#
# Priority order:
#   1. --manual flag  → explicit in-process setup
#   2. OTEL_EXPORTER_OTLP_ENDPOINT in env (from .env) + no existing providers
#      → auto in-process setup (same as --manual, no flag required)
#   3. opentelemetry-instrument wrapping this process
#      → providers already configured; just instrument FastMCP
#   4. Nothing configured → server runs without telemetry (warn)
_manual = "--manual" in sys.argv
if _manual:
    sys.argv.remove("--manual")

if _providers_already_configured():
    # opentelemetry-instrument set up providers; only need to instrument FastMCP.
    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor  # noqa: E402

    FastMCPInstrumentor().instrument()
elif _manual or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    setup_server_telemetry()
else:
    print(
        "⚠️  No OTel providers configured for server. "
        "Set OTEL_EXPORTER_OTLP_ENDPOINT in .env or use opentelemetry-instrument.",
        file=sys.stderr,
    )


from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("Weather Server")

MOCK_WEATHER = {
    "tokyo": {
        "temp_c": 22,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_kph": 12,
    },
    "london": {"temp_c": 14, "condition": "Rainy", "humidity": 82, "wind_kph": 20},
    "new york": {"temp_c": 18, "condition": "Sunny", "humidity": 55, "wind_kph": 8},
    "paris": {"temp_c": 16, "condition": "Overcast", "humidity": 70, "wind_kph": 15},
    "sydney": {"temp_c": 26, "condition": "Sunny", "humidity": 45, "wind_kph": 10},
}


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: Name of the city to get weather for.

    Returns:
        Weather data including temperature, condition, humidity, and wind speed.
    """
    key = city.lower().strip()
    if key in MOCK_WEATHER:
        data = MOCK_WEATHER[key]
        return {
            "city": city,
            "temperature_celsius": data["temp_c"],
            "condition": data["condition"],
            "humidity_percent": data["humidity"],
            "wind_kph": data["wind_kph"],
        }
    return {
        "city": city,
        "temperature_celsius": 20,
        "condition": "Clear",
        "humidity_percent": 50,
        "wind_kph": 10,
    }


@mcp.tool()
def get_packing_suggestions(temperature: float, condition: str, days: int = 3) -> dict:
    """Get packing suggestions based on weather conditions.

    Args:
        temperature: Expected temperature in Celsius at the destination.
        condition: Weather condition (e.g. Sunny, Rainy, Overcast, Partly cloudy).
        days: Number of days for the trip.

    Returns:
        Categorized packing suggestions.
    """
    essentials = ["passport", "phone charger", "toiletries", "medications"]
    clothing = []
    accessories = []

    if temperature >= 25:
        clothing = ["t-shirts", "shorts", "light dresses", "sandals", "swimwear"]
        accessories = ["sunglasses", "sunscreen SPF 50", "hat", "reusable water bottle"]
    elif temperature >= 15:
        clothing = [
            "light layers",
            "jeans",
            "long-sleeve shirts",
            "sneakers",
            "light jacket",
        ]
        accessories = ["sunglasses", "sunscreen SPF 30", "daypack"]
    else:
        clothing = [
            "warm layers",
            "sweaters",
            "thermal underwear",
            "boots",
            "warm coat",
        ]
        accessories = ["scarf", "gloves", "beanie", "hand warmers"]

    if "rain" in condition.lower():
        accessories.extend(["umbrella", "waterproof jacket", "waterproof bag cover"])
    if "wind" in condition.lower() or "storm" in condition.lower():
        accessories.append("windbreaker")

    return {
        "essentials": essentials,
        "clothing": [
            f"{days}x {item}" if "shirt" in item or "t-shirt" in item else item
            for item in clothing
        ],
        "accessories": accessories,
        "tip": f"Pack for {temperature}°C and {condition.lower()} conditions. Layers are always a good idea!",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    args = parser.parse_args()

    emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    otlp = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "not set")
    print(f"🔭 emitters={emitters}  otlp={otlp}", file=sys.stderr)

    if args.transport in ("http", "streamable-http"):
        print(f"🌐 HTTP server → http://{args.host}:{args.port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport=args.transport)
