#!/usr/bin/env python3
"""MCP Weather Prompt Client.

Connects to the weather prompt server and demonstrates prompt operations
with OpenTelemetry instrumentation capturing traces and metrics.

Expected spans:
    - "prompts/list"              (SpanKind.CLIENT)
    - "prompts/get weather_forecast"     (SpanKind.CLIENT)
    - "prompts/get travel_packing_advice" (SpanKind.CLIENT)

Usage:
    # Spawn server as subprocess (single terminal)
    OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317" \\
    OTEL_SERVICE_NAME="mcp-prompt-demo" \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" \\
    python client.py --wait 10

    # Connect to external SSE server
    python client.py --server-url http://localhost:8001/sse --wait 10
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def setup_telemetry():
    """Set up OpenTelemetry with OTLP and/or console exporters."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", "mcp-prompt-demo")}
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
            print(f"  OTLP exporter -> {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print("  OTLP not available", file=sys.stderr)

    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    print("  Console exporter enabled", file=sys.stderr)

    trace.set_tracer_provider(trace_provider)
    if metric_readers:
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=metric_readers)
        )

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()
    print("  FastMCP instrumentation applied", file=sys.stderr)


async def run_prompt_demo(server_url: str | None = None):
    """Connect to the weather prompt server and exercise prompt operations."""
    from fastmcp import Client

    print("\n" + "=" * 60)
    print("  MCP Prompt Demo - OpenTelemetry Instrumentation")
    print("=" * 60)

    if server_url:
        target = server_url
        print(f"\n  Connecting to: {server_url}")
    else:
        target = Path(__file__).parent / "server.py"
        print(f"\n  Spawning server: {target.name}")

    async with Client(target) as client:
        print("  Connected!\n")

        # 1. List prompts
        print("  Step 1: List prompts")
        print("  " + "-" * 40)
        prompts = await client.list_prompts()
        for p in prompts:
            args = ", ".join(a.name for a in (p.arguments or []))
            print(f"    {p.name}({args})")
        print()

        # 2. Get weather_forecast prompt
        print("  Step 2: Get weather_forecast for London")
        print("  " + "-" * 40)
        result = await client.get_prompt(
            "weather_forecast", arguments={"city": "London"}
        )
        for msg in result.messages:
            role = msg.role
            text = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
            print(f"    [{role}] {text[:120]}")
        print()

        # 3. Get travel_packing_advice prompt
        print("  Step 3: Get travel_packing_advice for Tokyo, 5 days")
        print("  " + "-" * 40)
        result = await client.get_prompt(
            "travel_packing_advice",
            arguments={"destination": "Tokyo", "days": "5"},
        )
        for msg in result.messages:
            role = msg.role
            text = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
            print(f"    [{role}] {text[:120]}")
        print()

    print("=" * 60)
    print("  Prompt demo completed!")
    print("=" * 60)


async def main(wait_seconds: int = 0, server_url: str | None = None):
    setup_telemetry()
    print()

    try:
        await run_prompt_demo(server_url=server_url)
    except Exception as e:
        print(f"\n  Demo failed: {e}")
        import traceback

        traceback.print_exc()

    if wait_seconds > 0:
        print(f"\n  Waiting {wait_seconds}s for telemetry flush...")
        await asyncio.sleep(wait_seconds)
        print("  Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Prompt Client")
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()

    asyncio.run(main(wait_seconds=args.wait, server_url=args.server_url))
