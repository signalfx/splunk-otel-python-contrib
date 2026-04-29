#!/usr/bin/env python3
"""Weather MCP Client — exercises tools, resources, and prompts over HTTP.

Connects to the Weather Server and exercises every MCP primitive:

  1. tools/list  -> tools/call  (get_current_weather, get_forecast, get_travel_packing)
  2. resources/list -> resources/read  (climate://cities, climate://london/annual)
  3. prompts/list -> prompts/get  (weather_briefing, travel_packing_advice)
  4. Error path   -> tools/call with unknown city

Usage (zero-code instrumentation via opentelemetry-instrument):

    # Against the default HTTP server (http://127.0.0.1:8000/mcp)
    OTEL_SERVICE_NAME=weather-mcp-client \\
    OTEL_TRACES_EXPORTER=console \\
    OTEL_METRICS_EXPORTER=console \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
      opentelemetry-instrument python client.py

    # With OTLP export to a collector
    OTEL_SERVICE_NAME=weather-mcp-client \\
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
      opentelemetry-instrument python client.py

    # stdio mode (spawns server.py as subprocess)
    opentelemetry-instrument python client.py --stdio
"""

import argparse
import asyncio
from pathlib import Path


def _text(result) -> str:
    """Extract text from an MCP result (CallToolResult, list, etc.)."""
    items = getattr(result, "content", None)
    if items is None:
        items = result if isinstance(result, list) else [result]
    parts = []
    for c in items:
        parts.append(c.text if hasattr(c, "text") else str(c))
    return "\n".join(parts)


async def run_demo(target):
    """Run the full demo against *target* (URL or Path)."""
    from fastmcp import Client

    banner = "Weather MCP Demo"
    print(f"\n{'=' * 60}\n  {banner}\n{'=' * 60}\n")

    async with Client(target) as client:
        # ---- Tools --------------------------------------------------------
        print("  [1] tools/list")
        tools = await client.list_tools()
        for t in tools:
            print(f"      - {t.name}")

        print("\n  [2] tools/call  get_current_weather('London')")
        r = await client.call_tool("get_current_weather", {"city": "London"})
        for line in _text(r).splitlines():
            print(f"      {line}")

        print("\n  [3] tools/call  get_forecast('Tokyo', 3)")
        r = await client.call_tool("get_forecast", {"city": "Tokyo", "days": 3})
        for line in _text(r).splitlines():
            print(f"      {line}")

        print("\n  [4] tools/call  get_travel_packing('Sydney', 7)")
        r = await client.call_tool(
            "get_travel_packing", {"destination": "Sydney", "days": 7}
        )
        for line in _text(r).splitlines():
            print(f"      {line}")

        # ---- Resources ----------------------------------------------------
        print("\n  [5] resources/list")
        resources = await client.list_resources()
        for res in resources:
            print(f"      {res.uri}")

        print("\n  [6] resources/read  climate://cities")
        r = await client.read_resource("climate://cities")
        for line in _text(r).splitlines():
            print(f"      {line}")

        print("\n  [7] resources/read  climate://london/annual")
        r = await client.read_resource("climate://london/annual")
        for line in _text(r).splitlines():
            print(f"      {line}")

        # ---- Prompts ------------------------------------------------------
        print("\n  [8] prompts/list")
        prompts = await client.list_prompts()
        for p in prompts:
            args = ", ".join(a.name for a in (p.arguments or []))
            print(f"      {p.name}({args})")

        print("\n  [9] prompts/get  weather_briefing(city='New York')")
        r = await client.get_prompt("weather_briefing", arguments={"city": "New York"})
        for m in r.messages:
            text = m.content.text if hasattr(m.content, "text") else str(m.content)
            print(f"      [{m.role}] {text[:120]}")

        print(
            "\n  [10] prompts/get  travel_packing_advice(destination='Mumbai', days=4)"
        )
        r = await client.get_prompt(
            "travel_packing_advice",
            arguments={"destination": "Mumbai", "days": "4"},
        )
        for m in r.messages:
            text = m.content.text if hasattr(m.content, "text") else str(m.content)
            print(f"      [{m.role}] {text[:120]}")

        # ---- Error path ---------------------------------------------------
        print("\n  [11] tools/call  get_current_weather('Atlantis')  [error]")
        try:
            await client.call_tool("get_current_weather", {"city": "Atlantis"})
        except Exception as exc:
            print(f"      caught: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 60}\n  Demo complete\n{'=' * 60}")


async def main(target, wait: int = 0):
    try:
        await run_demo(target)
    except Exception as exc:
        print(f"\n  FAILED: {exc}")
        import traceback

        traceback.print_exc()

    if wait > 0:
        print(f"\n  Waiting {wait}s for telemetry flush...")
        await asyncio.sleep(wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather MCP Client")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP server URL (default: http://127.0.0.1:8000/mcp)",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Spawn server.py as stdio subprocess instead of HTTP",
    )
    parser.add_argument("--wait", type=int, default=5, help="Seconds to wait for flush")
    args = parser.parse_args()

    if args.stdio:
        target = Path(__file__).parent / "server.py"
    else:
        target = args.server_url

    asyncio.run(main(target, wait=args.wait))
