#!/usr/bin/env python3
"""MCP Weather Prompt Server.

Exposes templatized prompts that fetch weather data from wttr.in
and return them as structured MCP prompt messages.

Usage:
    # stdio mode (for subprocess spawning by client)
    python server.py

    # SSE mode (for external client connections)
    python server.py --sse --port 8001
"""

import urllib.request

from fastmcp import FastMCP

mcp = FastMCP("Weather Prompt Server")


def _fetch_weather(city: str) -> str:
    """Fetch plain-text weather from wttr.in."""
    url = f"https://wttr.in/{urllib.request.quote(city)}?format=3"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception as e:
        return f"Unable to fetch weather for {city}: {e}"


@mcp.prompt()
def weather_forecast(city: str) -> str:
    """Get a weather forecast summary for a city.

    Fetches live weather from wttr.in and returns it as a prompt
    that an LLM can use to provide a natural-language forecast.

    Args:
        city: City name (e.g. "London", "San Francisco", "Tokyo")
    """
    weather = _fetch_weather(city)
    return (
        f"Here is the current weather data for {city}:\n\n"
        f"  {weather}\n\n"
        f"Please provide a brief, friendly forecast summary for {city} "
        f"based on the data above."
    )


@mcp.prompt()
def travel_packing_advice(destination: str, days: int) -> str:
    """Get packing advice based on weather at a travel destination.

    Fetches weather and returns a prompt asking for packing recommendations.

    Args:
        destination: Travel destination city
        days: Number of days for the trip
    """
    weather = _fetch_weather(destination)
    return (
        f"I'm traveling to {destination} for {days} days.\n\n"
        f"Current weather: {weather}\n\n"
        f"Based on this weather, what should I pack? "
        f"Please provide a concise packing list."
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MCP Weather Prompt Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode")
    parser.add_argument("--port", type=int, default=8001, help="SSE port")
    args = parser.parse_args()

    if args.sse:
        print(
            f"Starting SSE server at http://localhost:{args.port}/sse",
            file=sys.stderr,
        )
        mcp.run(transport="sse", host="localhost", port=args.port)
    else:
        mcp.run()
