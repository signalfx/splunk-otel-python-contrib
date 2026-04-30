#!/usr/bin/env python3
"""
End-to-End MCP Demo Runner

Orchestrates both MCP server and client to demonstrate the complete
instrumentation flow in stdio or HTTP transport mode.

Usage:
    # Load env vars (Splunk OTLP config)
    source .env

    # stdio mode — spawns server as subprocess (default)
    python run_demo.py --console
    python run_demo.py --wait 5

    # HTTP mode — spawns HTTP server subprocess, client connects over HTTP
    python run_demo.py --http --console
    python run_demo.py --http --port 8001 --wait 5

    # OTLP export to Splunk (via local collector configured in .env)
    source .env && python run_demo.py --http --wait 5
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


from _otel_helpers import load_dotenv as _load_dotenv


def print_banner(transport: str = "stdio"):
    """Print demo banner."""
    print()
    print("=" * 70)
    print("  MCP End-to-End Demo with OpenTelemetry Instrumentation")
    print("=" * 70)
    print()
    print("This demo shows:")
    print(f"  • Client connecting to MCP server via {transport}")
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
    service = os.environ.get("OTEL_SERVICE_NAME", "mcp-e2e-client")
    print(f"  • Emitters: {emitters}")
    print(f"  • Content Capture: {capture}")
    print(f"  • OTLP Endpoint: {otlp}")
    print(f"  • Service Name (client): {service}")
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


async def _run_operations(client) -> None:
    """Execute all demo tool calls against an already-connected client."""
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
            if hasattr(result, "content") and result.content:
                value = (
                    result.content[0].text
                    if hasattr(result.content[0], "text")
                    else str(result.content[0])
                )
            else:
                value = str(result)

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
        if hasattr(result, "isError") and result.isError:
            print(f"   divide(42, 0) → Error returned: {result}")
        else:
            print(f"   divide(42, 0) → {result}")
    except Exception as e:
        print(f"   divide(42, 0) → ❌ Exception: {type(e).__name__}: {e}")
    print()


async def run_demo(http_mode: bool = False, http_port: int = 8000):
    """Run the end-to-end demo (stdio or HTTP transport)."""
    from fastmcp import Client

    server_script = Path(__file__).parent / "server_instrumented.py"
    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found: {server_script}")

    server_env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("OTEL_", "VIRTUAL_ENV"))
        or k in ("HOME", "PATH", "SHELL", "TERM", "USER", "LOGNAME")
    }
    server_env["OTEL_SERVICE_NAME"] = "mcp-calculator-server"
    server_env.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")

    if http_mode:
        server_url = f"http://localhost:{http_port}/mcp"
        print(f"📡 HTTP mode: spawning {server_script.name} on port {http_port}...")
        print()

        server_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(server_script),
            "--http",
            "--host",
            "localhost",
            "--port",
            str(http_port),
            env={**server_env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait briefly for the server to become ready
        await asyncio.sleep(1.5)
        if server_proc.returncode is not None:
            stderr = await server_proc.stderr.read()
            raise RuntimeError(
                f"HTTP server exited early (code {server_proc.returncode}):\n"
                + stderr.decode()
            )

        print(f"🌐 HTTP server running at {server_url}")
        print(f"📡 Connecting client → {server_url}")
        print()
        try:
            async with Client(server_url) as client:
                await _run_operations(client)
        finally:
            server_proc.terminate()
            try:
                await asyncio.wait_for(server_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                server_proc.kill()
    else:
        # stdio: spawn server as subprocess
        from fastmcp.client.transports.stdio import PythonStdioTransport

        print(f"📡 stdio mode: Connecting to Calculator Server ({server_script.name})...")
        print()
        server_target = PythonStdioTransport(script_path=server_script, env=server_env)

        async with Client(server_target) as client:
            await _run_operations(client)

    print("=" * 50)
    print("✅ End-to-end demo completed successfully!")
    print("=" * 50)


async def main(
    console_output: bool = False,
    wait_seconds: int = 0,
    http_mode: bool = False,
    http_port: int = 8000,
):
    """Main entry point."""
    print_banner(transport="http (streamable-http)" if http_mode else "stdio")

    setup_telemetry(console_output=console_output)
    print()

    try:
        await run_demo(http_mode=http_mode, http_port=http_port)
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
    _load_dotenv()

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
        default=5,
        metavar="SECONDS",
        help="Wait for telemetry to flush before exit (default: 5)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use Streamable-HTTP transport instead of stdio. "
        "Spawns the server as an HTTP subprocess on --port.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the HTTP server subprocess (default: 8000)",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            console_output=args.console,
            wait_seconds=args.wait,
            http_mode=args.http,
            http_port=args.port,
        )
    )
