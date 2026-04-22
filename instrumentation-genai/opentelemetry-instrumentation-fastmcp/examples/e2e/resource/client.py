#!/usr/bin/env python3
"""MCP System Dashboard Resource Client.

Connects to the system dashboard server and demonstrates resource operations
with OpenTelemetry instrumentation.

Expected spans:
    - "resources/list"                          (SpanKind.CLIENT)
    - "resources/read system://info"            (SpanKind.CLIENT)
    - "resources/read system://uptime"          (SpanKind.CLIENT)
    - "resources/read system://env/HOME"        (SpanKind.CLIENT)

Usage:
    OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317" \\
    OTEL_SERVICE_NAME="mcp-resource-demo" \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" \\
    python client.py --wait 10

    # Or connect to external SSE server
    python client.py --server-url http://localhost:8002/sse --wait 10
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
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", "mcp-resource-demo")}
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


async def run_resource_demo(server_url: str | None = None):
    """Connect to the system dashboard server and exercise resource operations."""
    from fastmcp import Client

    print("\n" + "=" * 60)
    print("  MCP Resource Demo - OpenTelemetry Instrumentation")
    print("=" * 60)

    if server_url:
        target = server_url
        print(f"\n  Connecting to: {server_url}")
    else:
        target = Path(__file__).parent / "server.py"
        print(f"\n  Spawning server: {target.name}")

    async with Client(target) as client:
        print("  Connected!\n")

        # 1. List resources
        print("  Step 1: List resources")
        print("  " + "-" * 40)
        resources = await client.list_resources()
        for r in resources:
            print(f"    {r.uri}  ({r.name})")
        print()

        # Also list resource templates
        templates = await client.list_resource_templates()
        if templates:
            print("  Resource templates:")
            for t in templates:
                print(f"    {t.uriTemplate}  ({t.name})")
            print()

        # 2. Read static resource: system://info
        print("  Step 2: Read system://info")
        print("  " + "-" * 40)
        contents = await client.read_resource("system://info")
        for c in contents:
            text = c.text if hasattr(c, "text") else str(c)
            for line in text.strip().splitlines():
                print(f"    {line}")
        print()

        # 3. Read static resource: system://uptime
        print("  Step 3: Read system://uptime")
        print("  " + "-" * 40)
        contents = await client.read_resource("system://uptime")
        for c in contents:
            text = c.text if hasattr(c, "text") else str(c)
            for line in text.strip().splitlines():
                print(f"    {line}")
        print()

        # 4. Read template resource: system://env/HOME
        print("  Step 4: Read system://env/HOME")
        print("  " + "-" * 40)
        contents = await client.read_resource("system://env/HOME")
        for c in contents:
            text = c.text if hasattr(c, "text") else str(c)
            print(f"    {text.strip()}")
        print()

        # 5. Read template resource: system://env/USER
        print("  Step 5: Read system://env/USER")
        print("  " + "-" * 40)
        contents = await client.read_resource("system://env/USER")
        for c in contents:
            text = c.text if hasattr(c, "text") else str(c)
            print(f"    {text.strip()}")
        print()

    print("=" * 60)
    print("  Resource demo completed!")
    print("=" * 60)


async def main(wait_seconds: int = 0, server_url: str | None = None):
    setup_telemetry()
    print()

    try:
        await run_resource_demo(server_url=server_url)
    except Exception as e:
        print(f"\n  Demo failed: {e}")
        import traceback

        traceback.print_exc()

    if wait_seconds > 0:
        print(f"\n  Waiting {wait_seconds}s for telemetry flush...")
        await asyncio.sleep(wait_seconds)
        print("  Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Resource Client")
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()

    asyncio.run(main(wait_seconds=args.wait, server_url=args.server_url))
