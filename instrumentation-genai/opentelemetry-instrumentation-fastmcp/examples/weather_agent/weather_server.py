#!/usr/bin/env python3
"""
Weather MCP Server

Provides weather and travel packing tools via the Model Context Protocol.
Designed to be spawned as a subprocess by the weather agent.

Usage:
    # --- Manual instrumentation (agent passes --manual) ---
    python weather_server.py --manual

    # --- Zero-code instrumentation ---
    opentelemetry-instrument python weather_server.py

Telemetry setup reads OTEL env vars (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
propagated by the client process so both sides export to the same collector.
"""

import os
import sys


def setup_server_telemetry():
    """Configure OTel providers and FastMCP instrumentation for the server."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", "weather-mcp-server")}
    )

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
        except ImportError:
            print("OTLP exporters not available in server", file=sys.stderr)

    if os.environ.get("OTEL_SERVER_CONSOLE_EXPORT"):
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(trace_provider)
    if metric_readers:
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=metric_readers)
        )

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()


# Apply manual telemetry only when explicitly requested.
# Zero-code mode (opentelemetry-instrument) sets up providers automatically.
if "--manual" in sys.argv:
    sys.argv.remove("--manual")
    setup_server_telemetry()


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
        choices=["stdio", "http", "streamable-http", "sse"],
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

    if args.transport in ("http", "streamable-http"):
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport=args.transport)
