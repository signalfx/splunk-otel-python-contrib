#!/usr/bin/env python3
"""
End-to-End MCP Demo Runner

This script orchestrates running both the MCP server and client together
to demonstrate the complete instrumentation flow.

Usage:
    # Console output (see traces and metrics)
    OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" python run_demo.py --console

    # OTLP export to backend
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
    python run_demo.py --wait 30

    # With content capture enabled
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
    python run_demo.py --console
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def print_banner():
    """Print demo banner."""
    print()
    print("=" * 70)
    print("  MCP End-to-End Demo with OpenTelemetry Instrumentation")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  • Client connecting to MCP server via stdio")
    print("  • Tool discovery (list_tools)")
    print("  • Tool invocations with full tracing")
    print("  • MCP metrics (duration, output size)")
    print()
    print("Environment:")
    emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    capture = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
    )
    otlp = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "not set")
    print(f"  • Emitters: {emitters}")
    print(f"  • Content Capture: {capture}")
    print(f"  • OTLP Endpoint: {otlp}")
    print()


def setup_telemetry(console_output: bool = False):
    """Set up OpenTelemetry with tracing and metrics."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    # Create resource with service info
    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "mcp-e2e-demo"),
            "service.version": "1.0.0",
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
        print("✅ Console exporters enabled", file=sys.stderr)

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
                "⚠️  OTLP not available (pip install opentelemetry-exporter-otlp)",
                file=sys.stderr,
            )

    trace.set_tracer_provider(trace_provider)

    if metric_readers:
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers,
        )
        metrics.set_meter_provider(meter_provider)

    # Apply FastMCP instrumentation
    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()
    print("✅ FastMCP instrumentation applied", file=sys.stderr)


async def run_demo():
    """Run the end-to-end demo."""
    from fastmcp import Client

    # Get the server script path
    server_script = Path(__file__).parent / "server.py"

    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found: {server_script}")

    print(f"📡 Connecting to Calculator Server ({server_script.name})...")
    print()

    # Connect to the server
    async with Client(server_script) as client:
        print("✅ Connected!\n")

        # Step 1: List available tools
        print("📋 Step 1: Discovering available tools")
        print("-" * 50)
        tools = await client.list_tools()
        print(f"   Found {len(tools)} tools:")
        for tool in tools:
            desc = tool.description.split("\n")[0] if tool.description else ""
            print(f"   • {tool.name}: {desc}")
        print()

        # Step 2: Call each tool
        print("🔧 Step 2: Calling calculator tools")
        print("-" * 50)

        operations = [
            ("add", {"a": 15, "b": 27}),
            ("subtract", {"a": 100, "b": 37}),
            ("multiply", {"a": 8, "b": 9}),
            ("divide", {"a": 144, "b": 12}),
            ("calculate_expression", {"expression": "(10 + 5) * 2"}),
        ]

        for tool_name, args in operations:
            try:
                result = await client.call_tool(tool_name, args)
                # Extract result value
                if hasattr(result, "content") and result.content:
                    value = (
                        result.content[0].text
                        if hasattr(result.content[0], "text")
                        else str(result.content[0])
                    )
                else:
                    value = str(result)

                # Format args for display
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                print(f"   {tool_name}({args_str}) → {value}")
            except Exception as e:
                print(f"   {tool_name}(...) → ❌ {e}")
        print()

        # Step 3: Test error handling
        print("🚨 Step 3: Testing error handling")
        print("-" * 50)
        try:
            result = await client.call_tool("divide", {"a": 42, "b": 0})
            # Check if result indicates an error
            if hasattr(result, "isError") and result.isError:
                print(f"   divide(42, 0) → Error returned: {result}")
            else:
                print(f"   divide(42, 0) → {result}")
        except Exception as e:
            print(f"   divide(42, 0) → ❌ Exception: {type(e).__name__}: {e}")
        print()

    print("=" * 50)
    print("✅ End-to-end demo completed successfully!")
    print("=" * 50)


async def main(console_output: bool = False, wait_seconds: int = 0):
    """Main entry point."""
    print_banner()

    # Set up telemetry
    setup_telemetry(console_output=console_output)
    print()

    try:
        await run_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

    # Wait for telemetry to flush
    if wait_seconds > 0:
        print(f"\n⏳ Waiting {wait_seconds}s for telemetry to flush...")
        await asyncio.sleep(wait_seconds)
        print("✅ Telemetry flushed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-End MCP Demo with OpenTelemetry"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Enable console exporters for traces and metrics",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Wait for telemetry to flush before exit (default: 0)",
    )
    args = parser.parse_args()

    asyncio.run(main(console_output=args.console, wait_seconds=args.wait))
