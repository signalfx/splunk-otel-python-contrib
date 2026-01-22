#!/usr/bin/env python3
"""
MCP Development Assistant Demo Client

Demonstrates the OpenTelemetry instrumentation with Splunk Distro
for FastMCP client-side operations.

Usage:
    # Run with default providers (OTLP if configured):
    export OTEL_SERVICE_NAME="mcp-dev-assistant-client"
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
    python dev_assistant_client.py

    # Run with console output for debugging (traces only by default):
    python dev_assistant_client.py --console

    # Run with console output including metrics:
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
    python dev_assistant_client.py --console --wait 10

    # Run with OTLP exporter:
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
    python dev_assistant_client.py

    # Wait for telemetry and evaluations to complete:
    python dev_assistant_client.py --console --wait 60

Environment Variables:
    OTEL_INSTRUMENTATION_GENAI_EMITTERS: Controls telemetry output
        - "span" (default): Only traces
        - "span_metric": Traces + metrics
        - "span_metric_event": Traces + metrics + content events
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def setup_telemetry(console_output: bool = False):
    """Set up OpenTelemetry with optional console output."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    # Create provider with resource attributes
    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "mcp-client-demo"),
        }
    )
    trace_provider = TracerProvider(resource=resource)

    # Set up metrics provider
    metric_readers = []
    if console_output:
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )

        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=5000,  # Export every 5 seconds
            )
        )
        print("Console metric exporter enabled", file=sys.stderr)

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            print(f"OTLP metric exporter enabled: {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print(
                "OTLP metric exporter not available (install opentelemetry-exporter-otlp)",
                file=sys.stderr,
            )

    if metric_readers:
        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)

    # Add console span exporter if requested via argument
    if console_output:
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        print("Console span exporter enabled", file=sys.stderr)

    # Add OTLP trace exporter if endpoint is configured
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            print(f"OTLP trace exporter enabled: {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print(
                "OTLP trace exporter not available (install opentelemetry-exporter-otlp)",
                file=sys.stderr,
            )

    trace.set_tracer_provider(trace_provider)

    # Import and apply instrumentation AFTER setting up the provider
    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    # Apply FastMCP instrumentation
    FastMCPInstrumentor().instrument()


async def demo_mcp_client():
    """Demonstrate FastMCP client with OpenTelemetry instrumentation."""
    # Import FastMCP client (after instrumentation is applied)
    from fastmcp import Client

    print("MCP Development Assistant - OpenTelemetry Instrumentation Demo")
    print("This demo showcases the Splunk Distro FastMCP instrumentation")
    print()

    # Get the server script path
    server_script = Path(__file__).parent / "dev_assistant_server.py"

    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found: {server_script}")

    print(f"📡 Connecting to MCP server: {server_script}")
    print()

    # Create FastMCP client - this triggers the instrumentation
    # FastMCP Client can connect to a server script directly
    async with Client(server_script) as client:
        print("✅ Connected to Development Assistant MCP Server")

        print("\n" + "=" * 60)
        print("🔧 Demonstrating MCP Tool Calls with OpenTelemetry Tracing")
        print("=" * 60)

        # Demo 1: List tools
        print("\n📋 Listing available tools...")
        tools = await client.list_tools()
        print(f"   Found {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5
            desc = tool.description[:50] if tool.description else "No description"
            print(f"   - {tool.name}: {desc}...")

        # Demo 2: Get system info
        print("\n💻 Getting system information...")
        result = await client.call_tool("get_system_info", {})
        print(f"   System info retrieved: {str(result)[:100]}...")

        # Demo 3: List files
        print("\n📁 Listing current directory...")
        result = await client.call_tool("list_files", {"directory": "."})
        print(f"   Directory listing retrieved: {str(result)[:100]}...")

        # Demo 4: Git status
        print("\n🔀 Checking Git status...")
        try:
            result = await client.call_tool("git_status", {"repo_path": "."})
            print(f"   Git status: {str(result)[:100]}...")
        except Exception as e:
            print(f"   Git status failed (not a git repo?): {e}")

        # Demo 5: Search code
        print("\n🔍 Searching for 'def ' in Python files...")
        result = await client.call_tool(
            "search_code",
            {
                "pattern": "def ",
                "directory": ".",
                "file_extensions": [".py"],
                "max_results": 5,
            },
        )
        print(f"   Search completed: {str(result)[:100]}...")

        # Demo 6: File operations
        print("\n📝 Creating and reading a test file...")
        test_file = "/tmp/mcp_otel_test.txt"
        await client.call_tool(
            "write_file",
            {
                "file_path": test_file,
                "content": "Hello from MCP with OpenTelemetry!\nThis is a test file.",
            },
        )
        print(f"   Created {test_file}")

        result = await client.call_tool(
            "read_file", {"file_path": test_file, "max_lines": 5}
        )
        print(f"   Read {test_file}: {str(result)[:100]}...")

    print("\n" + "=" * 60)
    print("🎉 Demo completed! Check the trace output above to see:")
    print("   • Spans for MCP client session and tool calls")
    print("   • Input/output capture (if content capture is enabled)")
    print("   • Duration metrics for tool executions")
    print("   • Error handling and status codes")
    print("=" * 60)


async def main(console_output: bool = False, wait_seconds: int = 0):
    """Main entry point."""
    # Set up telemetry first
    setup_telemetry(console_output=console_output)

    try:
        await demo_mcp_client()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

    # Wait for telemetry and evaluations to complete if requested
    if wait_seconds > 0:
        print(f"\n⏳ Waiting {wait_seconds} seconds for telemetry and evaluations...")
        await asyncio.sleep(wait_seconds)
        print("✅ Wait complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Development Assistant Demo Client"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Enable console span exporter for debugging",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Wait specified seconds for telemetry and evaluations to complete",
    )
    args = parser.parse_args()

    asyncio.run(main(console_output=args.console, wait_seconds=args.wait))
