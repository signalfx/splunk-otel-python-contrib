#!/usr/bin/env python3
"""
MCP Client for Calculator Server

Connects to the calculator server and demonstrates tool calls
with OpenTelemetry instrumentation capturing traces, metrics,
and session context propagation via OTel Baggage.

Usage:
    # Option 1: Spawn server as subprocess (single terminal)
    python client.py --console

    # Option 2: Connect to external server (separate terminals)
    # Terminal 1: python server_instrumented.py --sse --port 8000
    # Terminal 2:
    python client.py --server-url http://localhost:8000/sse --console

    # With metrics enabled
    OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" python client.py --console

    # With session propagation via baggage
    OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION="baggage" \\
    OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS="user.id" \\
    python client.py --console --session-id "conv-123" --user-id "user-456"

    # OTLP export (default endpoint http://localhost:4317)
    python client.py --otlp --wait 10

    # OTLP export with custom endpoint
    OTEL_EXPORTER_OTLP_ENDPOINT="http://collector:4317" python client.py --wait 10
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def setup_telemetry(console_output: bool = False, otlp_enabled: bool = False):
    """Set up OpenTelemetry with tracing and metrics.

    Args:
        console_output: Enable console span/metric exporters for debugging.
        otlp_enabled: Enable OTLP gRPC exporters. Uses OTEL_EXPORTER_OTLP_ENDPOINT
            env var if set, otherwise defaults to http://localhost:4317.

    Returns:
        Tuple of (TracerProvider, MeterProvider | None) for shutdown.
    """
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
    meter_provider = None

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

    # OTLP export: enabled via --otlp flag or OTEL_EXPORTER_OTLP_ENDPOINT env var
    otlp_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )
    if otlp_enabled or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
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

    return trace_provider, meter_provider


async def run_calculator_demo(
    server_url: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
):
    """Connect to calculator server and demonstrate tool calls.

    Args:
        server_url: Optional URL of external MCP server (e.g., http://localhost:8000/sse).
                   If not provided, spawns server.py as a subprocess.
        session_id: Optional session ID to propagate via OTel Baggage.
        user_id: Optional user ID to propagate via OTel Baggage.
    """
    from fastmcp import Client

    print("\n" + "=" * 60)
    print("MCP Calculator Client - OpenTelemetry Instrumentation Demo")
    print("=" * 60)

    # Set session context if provided — this propagates via OTel Baggage
    # when OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION=baggage
    if session_id or user_id:
        from opentelemetry.util.genai.handler import set_session_context

        set_session_context(
            session_id=session_id,
            user_id=user_id,
        )
        print("\n🔑 Session context set:", file=sys.stderr)
        if session_id:
            print(f"   gen_ai.conversation.id = {session_id}", file=sys.stderr)
        if user_id:
            print(f"   user.id                = {user_id}", file=sys.stderr)

        propagation_mode = os.environ.get(
            "OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION", "contextvar"
        )
        print(f"   propagation = {propagation_mode}", file=sys.stderr)

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

        # Test session propagation (if session context was set)
        if session_id or user_id:
            print("\n🔗 Testing Session Propagation:")
            print("-" * 40)
            try:
                result = await client.call_tool("get_session_info", {})
                if hasattr(result, "content") and result.content:
                    content = result.content[0]
                    value = content.text if hasattr(content, "text") else str(content)
                else:
                    value = str(result)
                print(f"   Server sees: {value}")
            except Exception as e:
                print(f"   get_session_info = ❌ Error: {e}")
                print(
                    "   (Ensure the server has the get_session_info tool)",
                    file=sys.stderr,
                )

    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


async def main(
    console_output: bool = False,
    otlp_enabled: bool = False,
    wait_seconds: int = 0,
    server_url: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
):
    """Main entry point."""
    # Set up telemetry before anything else
    trace_provider, meter_provider = setup_telemetry(
        console_output=console_output, otlp_enabled=otlp_enabled
    )

    try:
        await run_calculator_demo(
            server_url=server_url,
            session_id=session_id,
            user_id=user_id,
        )
        await run_calculator_demo(server_url=server_url)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Shutdown providers to flush pending telemetry (BatchSpanProcessor,
        # PeriodicExportingMetricReader) before the process exits.
        if wait_seconds > 0:
            print(f"\n⏳ Waiting {wait_seconds}s for telemetry to flush...")
            await asyncio.sleep(wait_seconds)
        print("🔄 Shutting down telemetry providers...", file=sys.stderr)
        trace_provider.shutdown()
        if meter_provider:
            meter_provider.shutdown()
        print("✅ Telemetry flushed", file=sys.stderr)


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
        "--otlp",
        action="store_true",
        help="Enable OTLP gRPC exporters (default endpoint: http://localhost:4317, "
        "override with OTEL_EXPORTER_OTLP_ENDPOINT)",
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
    parser.add_argument(
        "--session-id",
        "--session",
        type=str,
        default=None,
        metavar="ID",
        help="Session ID (gen_ai.conversation.id) to propagate via OTel Baggage "
        "to the MCP server. "
        "Requires OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION=baggage.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        metavar="ID",
        help="User ID to propagate via OTel Baggage to the MCP server. "
        "Requires OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION=baggage.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            console_output=args.console,
            otlp_enabled=args.otlp,
            wait_seconds=args.wait,
            server_url=args.server_url,
            session_id=args.session_id,
            user_id=args.user_id,
        )
    )
