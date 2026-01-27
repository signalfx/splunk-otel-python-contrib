#!/usr/bin/env python3
"""
MCP Client for Calculator Server

Connects to the calculator server and demonstrates tool calls
with OpenTelemetry instrumentation capturing traces and metrics.

Usage:
    # Option 1: Spawn server as subprocess (single terminal)
    python client.py --console

    # Option 2: Connect to external server (separate terminals)
    # Terminal 1: python server_instrumented.py --sse --port 8000
    # Terminal 2:
    python client.py --server-url http://localhost:8000/sse --console

    # With metrics enabled
    OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" python client.py --console

    # OTLP export
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
    python client.py --wait 30
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def setup_telemetry(console_output: bool = False):
    """Set up OpenTelemetry with tracing and metrics."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    # Create resource with service info
    resource = Resource.create(
        {
            "service.name": os.environ.get(
                "OTEL_SERVICE_NAME", "mcp-calculator-client"
            ),
        }
    )

    # Set up trace provider
    trace_provider = TracerProvider(resource=resource)

    # Set up metric readers
    metric_readers = []

    if console_output:
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )

        trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=5000,
            )
        )
        print("✅ Console exporters enabled (traces + metrics)", file=sys.stderr)

    # Check for OTLP endpoint
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
            print(f"✅ OTLP exporters enabled: {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print(
                "⚠️  OTLP exporters not available (pip install opentelemetry-exporter-otlp)",
                file=sys.stderr,
            )

    trace.set_tracer_provider(trace_provider)

    if metric_readers:
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers,
        )
        metrics.set_meter_provider(meter_provider)

    # Apply FastMCP instrumentation AFTER setting up providers
    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()
    print("✅ FastMCP instrumentation applied", file=sys.stderr)


async def run_calculator_demo(server_url: str | None = None):
    """Connect to calculator server and demonstrate tool calls.

    Args:
        server_url: Optional URL of external MCP server (e.g., http://localhost:8000/sse).
                   If not provided, spawns server.py as a subprocess.
    """
    from fastmcp import Client

    print("\n" + "=" * 60)
    print("MCP Calculator Client - OpenTelemetry Instrumentation Demo")
    print("=" * 60)

    if server_url:
        # Connect to external server
        print(f"\n🌐 Connecting to external server: {server_url}")
        server_target = server_url
    else:
        # Spawn server as subprocess
        server_script = Path(__file__).parent / "server.py"
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found: {server_script}")
        print(f"\n📡 Spawning server subprocess: {server_script.name}")
        server_target = server_script

    # Connect to the server using FastMCP Client
    async with Client(server_target) as client:
        print("✅ Connected to Calculator Server\n")

        # List available tools
        print("📋 Available Tools:")
        tools = await client.list_tools()
        for tool in tools:
            desc = (
                tool.description.split("\n")[0]
                if tool.description
                else "No description"
            )
            print(f"   - {tool.name}: {desc}")

        print("\n🔧 Testing Calculator Tools:")
        print("-" * 40)

        # Test each operation
        test_cases = [
            ("add", {"a": 5, "b": 3}, "5 + 3"),
            ("subtract", {"a": 10, "b": 4}, "10 - 4"),
            ("multiply", {"a": 6, "b": 7}, "6 × 7"),
            ("divide", {"a": 20, "b": 4}, "20 ÷ 4"),
            ("calculate_expression", {"expression": "2 + 3 * 4"}, "Expression"),
        ]

        for tool_name, args, description in test_cases:
            try:
                result = await client.call_tool(tool_name, args)
                # Extract text content from result
                if hasattr(result, "content") and result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        value = content.text
                    else:
                        value = str(content)
                else:
                    value = str(result)
                print(f"   {description} = {value}")
            except Exception as e:
                print(f"   {description} = Error: {e}")

        # Test error handling (divide by zero)
        print("\n🚨 Testing Error Handling:")
        print("-" * 40)
        try:
            result = await client.call_tool("divide", {"a": 10, "b": 0})
            print(f"   divide(10, 0) = {result}")
        except Exception as e:
            print(f"   divide(10, 0) = ❌ Error handled: {type(e).__name__}")

    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


async def main(
    console_output: bool = False, wait_seconds: int = 0, server_url: str | None = None
):
    """Main entry point."""
    # Set up telemetry before anything else
    setup_telemetry(console_output=console_output)

    try:
        await run_calculator_demo(server_url=server_url)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

    # Wait for telemetry to flush
    if wait_seconds > 0:
        print(f"\n⏳ Waiting {wait_seconds}s for telemetry to flush...")
        await asyncio.sleep(wait_seconds)
        print("✅ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Calculator Client with OpenTelemetry"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Enable console exporters for debugging",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Wait for telemetry to flush before exit",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        metavar="URL",
        help="URL of external MCP server (e.g., http://localhost:8000/sse). "
        "If not provided, spawns server.py as subprocess.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            console_output=args.console,
            wait_seconds=args.wait,
            server_url=args.server_url,
        )
    )
