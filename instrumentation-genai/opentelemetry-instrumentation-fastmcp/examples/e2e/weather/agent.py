#!/usr/bin/env python3
"""Trip-planning agent — chains MCP tool calls into a workflow.

Connects to the Weather Server and orchestrates a multi-step trip plan:

  Step 1: get_current_weather     — check conditions at destination
  Step 2: get_forecast            — look ahead for the trip window
  Step 3: get_travel_packing      — build a packing list from weather
  Step 4: book_flight             — reserve flights
  Step 5: weather_briefing prompt — generate a departure-day briefing

Each step is a separate MCP tool/prompt call, demonstrating how an agent
chains operations where each step's output informs the next.

Usage (zero-code instrumentation via opentelemetry-instrument):

    # Against the default HTTP server (http://127.0.0.1:8000/mcp)
    OTEL_SERVICE_NAME=trip-planning-agent \\
    OTEL_TRACES_EXPORTER=console \\
    OTEL_METRICS_EXPORTER=console \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
      opentelemetry-instrument python agent.py

    # With OTLP export
    OTEL_SERVICE_NAME=trip-planning-agent \\
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
      opentelemetry-instrument python agent.py

    # stdio mode (spawns server as subprocess)
    opentelemetry-instrument python agent.py --stdio
"""

import argparse
import asyncio
from datetime import date, timedelta
from pathlib import Path


def _text(result) -> str:
    """Extract text from an MCP result."""
    items = getattr(result, "content", None)
    if items is None:
        items = result if isinstance(result, list) else [result]
    return "\n".join(c.text if hasattr(c, "text") else str(c) for c in items)


async def plan_trip(target, *, origin: str, destination: str, days: int):
    """Run the trip-planning agent workflow against *target*."""
    from fastmcp import Client

    departure = date.today() + timedelta(days=7)
    return_dt = departure + timedelta(days=days)

    print(f"\n{'=' * 60}")
    print("  Trip Planning Agent")
    print(f"  {origin.title()} -> {destination.title()}, {days} days")
    print(f"  {departure} to {return_dt}")
    print(f"{'=' * 60}\n")

    async with Client(target) as client:
        # Step 1 — Check current weather at destination
        print("  [1/5] Checking current weather...")
        weather = await client.call_tool("get_current_weather", {"city": destination})
        weather_text = _text(weather)
        for line in weather_text.splitlines():
            print(f"        {line}")

        # Step 2 — Get forecast for the trip window
        print(f"\n  [2/5] Getting {days}-day forecast...")
        forecast = await client.call_tool(
            "get_forecast", {"city": destination, "days": days}
        )
        forecast_text = _text(forecast)
        for line in forecast_text.splitlines():
            print(f"        {line}")

        # Step 3 — Build packing list based on weather
        print("\n  [3/5] Building packing list...")
        packing = await client.call_tool(
            "get_travel_packing", {"destination": destination, "days": days}
        )
        packing_text = _text(packing)
        for line in packing_text.splitlines():
            print(f"        {line}")

        # Step 4 — Book flights
        print("\n  [4/5] Booking flights...")
        booking = await client.call_tool(
            "book_flight",
            {
                "origin": origin,
                "destination": destination,
                "departure_date": str(departure),
                "return_date": str(return_dt),
                "passengers": 1,
            },
        )
        booking_text = _text(booking)
        for line in booking_text.splitlines():
            print(f"        {line}")

        # Step 5 — Get a weather briefing prompt for departure day
        print("\n  [5/5] Generating departure-day briefing...")
        briefing = await client.get_prompt(
            "weather_briefing", arguments={"city": destination}
        )
        for msg in briefing.messages:
            text = (
                msg.content.text if hasattr(msg.content, "text") else str(msg.content)
            )
            print(f"        [{msg.role}] {text[:200]}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Trip plan complete!")
    print(f"{'=' * 60}")


async def main(target, *, origin: str, destination: str, days: int, wait: int):
    try:
        await plan_trip(target, origin=origin, destination=destination, days=days)
    except Exception as exc:
        print(f"\n  FAILED: {exc}")
        import traceback

        traceback.print_exc()

    if wait > 0:
        print(f"\n  Waiting {wait}s for telemetry flush...")
        await asyncio.sleep(wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip Planning Agent")
    parser.add_argument("--origin", default="New York", help="Departure city")
    parser.add_argument("--destination", default="Tokyo", help="Destination city")
    parser.add_argument("--days", type=int, default=5, help="Trip length in days")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP server URL",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Spawn server.py as stdio subprocess",
    )
    parser.add_argument("--wait", type=int, default=5, help="Seconds to wait for flush")
    args = parser.parse_args()

    if args.stdio:
        target = Path(__file__).parent / "server.py"
    else:
        target = args.server_url

    asyncio.run(
        main(
            target,
            origin=args.origin,
            destination=args.destination,
            days=args.days,
            wait=args.wait,
        )
    )
