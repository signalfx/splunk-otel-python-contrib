#!/usr/bin/env python3
"""Weather MCP Server — tools, resources, and prompts over Streamable HTTP.

Demonstrates all three MCP primitives in a single server:

  Tools:
    - get_current_weather(city)    — current conditions for a city
    - get_forecast(city, days)     — multi-day forecast
    - get_travel_packing(destination, days) — packing list based on weather
    - book_flight(origin, dest, dates)     — simulated flight booking

  Resources:
    - climate://cities             — list of supported cities
    - climate://{city}/annual      — annual climate summary

  Prompts:
    - weather_briefing(city)       — templated briefing prompt
    - travel_packing_advice(destination, days) — packing advice prompt

Usage (zero-code instrumentation via opentelemetry-instrument):

    # Streamable HTTP with console exporter
    OTEL_SERVICE_NAME=weather-mcp-server \\
    OTEL_TRACES_EXPORTER=console \\
    OTEL_METRICS_EXPORTER=console \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
      opentelemetry-instrument python server.py

    # With OTLP export to a collector
    OTEL_SERVICE_NAME=weather-mcp-server \\
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
      opentelemetry-instrument python server.py

    # stdio (for subprocess spawning by client)
    opentelemetry-instrument python server.py --stdio
"""

import sys
import urllib.request

from fastmcp import FastMCP

mcp = FastMCP("Weather Server")

CITY_DATA = {
    "london": {
        "temp_c": 14,
        "condition": "Partly cloudy",
        "humidity": 72,
        "wind_kph": 19,
        "annual": "Mild oceanic; avg 5-23 C, rain year-round.",
    },
    "tokyo": {
        "temp_c": 22,
        "condition": "Clear",
        "humidity": 55,
        "wind_kph": 12,
        "annual": "Humid subtropical; avg 2-31 C, rainy season Jun-Jul.",
    },
    "new york": {
        "temp_c": 18,
        "condition": "Sunny",
        "humidity": 45,
        "wind_kph": 15,
        "annual": "Humid continental; avg -3 to 30 C, snow in winter.",
    },
    "sydney": {
        "temp_c": 20,
        "condition": "Mostly sunny",
        "humidity": 60,
        "wind_kph": 22,
        "annual": "Temperate oceanic; avg 8-26 C, mild winters.",
    },
    "mumbai": {
        "temp_c": 32,
        "condition": "Hazy",
        "humidity": 78,
        "wind_kph": 14,
        "annual": "Tropical monsoon; avg 19-33 C, heavy rain Jun-Sep.",
    },
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather conditions for a city.

    Args:
        city: City name (e.g. "London", "Tokyo", "New York")
    """
    data = CITY_DATA.get(city.lower())
    if data is None:
        raise ValueError(f"Unknown city: {city}. Supported: {', '.join(CITY_DATA)}")
    return (
        f"City: {city.title()}\n"
        f"Temperature: {data['temp_c']} C\n"
        f"Condition: {data['condition']}\n"
        f"Humidity: {data['humidity']}%\n"
        f"Wind: {data['wind_kph']} km/h"
    )


@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get a multi-day weather forecast for a city.

    Args:
        city: City name
        days: Number of forecast days (1-7, default 3)
    """
    data = CITY_DATA.get(city.lower())
    if data is None:
        raise ValueError(f"Unknown city: {city}. Supported: {', '.join(CITY_DATA)}")
    lines = [f"Forecast for {city.title()} ({days} days):"]
    for d in range(1, min(days, 7) + 1):
        offset = d - 1
        temp = data["temp_c"] + offset * (-1 if d % 2 else 1)
        lines.append(f"  Day {d}: {temp} C, {data['condition']}")
    return "\n".join(lines)


@mcp.tool()
def get_travel_packing(destination: str, days: int = 5) -> str:
    """Generate a packing list based on weather at a destination.

    Args:
        destination: Travel destination city
        days: Trip length in days (default 5)
    """
    data = CITY_DATA.get(destination.lower())
    if data is None:
        raise ValueError(
            f"Unknown city: {destination}. Supported: {', '.join(CITY_DATA)}"
        )

    items = ["passport / ID", "phone charger", "toiletries"]
    temp = data["temp_c"]
    if temp < 10:
        items += ["warm jacket", "gloves", "scarf", "thermal layers"]
    elif temp < 20:
        items += ["light jacket", "sweater", "long pants"]
    else:
        items += ["sunscreen", "sunglasses", "light clothing", "hat"]
    if data["humidity"] > 70:
        items += ["umbrella", "rain jacket"]
    if days > 5:
        items.append("laundry bag")

    lines = [
        f"Packing list for {destination.title()} ({days} days, {temp} C, {data['condition']}):"
    ]
    for item in items:
        lines.append(f"  - {item}")
    return "\n".join(lines)


@mcp.tool()
def book_flight(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    passengers: int = 1,
) -> str:
    """Book a flight between two cities (simulated).

    Args:
        origin: Departure city
        destination: Arrival city
        departure_date: Departure date (YYYY-MM-DD)
        return_date: Return date (YYYY-MM-DD)
        passengers: Number of passengers (default 1)
    """
    import random
    import string

    confirmation = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    price = random.randint(250, 1200)
    return (
        f"Flight booked!\n"
        f"  Confirmation: {confirmation}\n"
        f"  Route: {origin.title()} -> {destination.title()}\n"
        f"  Depart: {departure_date}  Return: {return_date}\n"
        f"  Passengers: {passengers}\n"
        f"  Total: ${price * passengers}"
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("climate://cities")
def list_cities() -> str:
    """List all cities with available climate data."""
    return "\n".join(f"- {c.title()}" for c in sorted(CITY_DATA))


@mcp.resource("climate://{city}/annual")
def annual_climate(city: str) -> str:
    """Annual climate summary for a city.

    Args:
        city: City name
    """
    data = CITY_DATA.get(city.lower())
    if data is None:
        return f"No climate data for {city}"
    return f"{city.title()}: {data['annual']}"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _fetch_live_weather(city: str) -> str:
    """Best-effort live weather from wttr.in (falls back to static data)."""
    url = f"https://wttr.in/{urllib.request.quote(city)}?format=3"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.read().decode().strip()
    except Exception:
        data = CITY_DATA.get(city.lower())
        if data:
            return f"{city.title()}: {data['condition']}, {data['temp_c']} C"
        return f"{city.title()}: weather unavailable"


@mcp.prompt()
def weather_briefing(city: str) -> str:
    """Create a weather briefing prompt for a city.

    Args:
        city: City name
    """
    weather = _fetch_live_weather(city)
    return (
        f"Here is the current weather for {city}:\n\n"
        f"  {weather}\n\n"
        f"Please provide a concise, friendly weather briefing."
    )


@mcp.prompt()
def travel_packing_advice(destination: str, days: int = 5) -> str:
    """Packing advice prompt based on destination weather.

    Args:
        destination: Travel destination city
        days: Trip length in days
    """
    weather = _fetch_live_weather(destination)
    return (
        f"I'm traveling to {destination} for {days} days.\n\n"
        f"Current weather: {weather}\n\n"
        f"Based on this weather, what should I pack? "
        f"Please provide a concise packing list."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()

    if args.stdio:
        mcp.run()
    else:
        print(
            f"Starting streamable-http server at http://127.0.0.1:{args.port}/mcp",
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
