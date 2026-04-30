#!/usr/bin/env python3
"""
MCP Development Assistant Client

Demonstrates end-to-end FastMCP client–server instrumentation with the
Splunk Distro for OpenTelemetry.  Supports two transport modes:

  • stdio  (default) — spawns ``dev_assistant_server.py`` as a sub-process
                       and communicates via stdin/stdout pipes.
  • HTTP              — connects to a running HTTP server via Streamable-HTTP.

Usage — stdio (default):
    # Spawn server automatically, traces go to Splunk:
    source .env
    OTEL_SERVICE_NAME=dev-assistant-client \\
        python dev_assistant_client.py

    # With console span output for local debugging:
    python dev_assistant_client.py --console

Usage — HTTP (requires server running first):
    # Terminal 1:
    OTEL_SERVICE_NAME=dev-assistant-server \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
        python dev_assistant_server.py --http --port 8001

    # Terminal 2:
    source .env
    OTEL_SERVICE_NAME=dev-assistant-client \\
        python dev_assistant_client.py --http --server-url http://localhost:8001/mcp

Usage — zero-code instrumentation (HTTP client):
    source .env
    OTEL_SERVICE_NAME=dev-assistant-client \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
        opentelemetry-instrument python dev_assistant_client.py \\
            --http --server-url http://localhost:8001/mcp

Environment Variables:
    OTEL_SERVICE_NAME                       Service name reported in Splunk
    OTEL_EXPORTER_OTLP_ENDPOINT            OTLP gRPC endpoint (e.g. http://localhost:4317)
    OTEL_EXPORTER_OTLP_HEADERS             Auth headers (e.g. X-SF-Token=<token>)
    OTEL_INSTRUMENTATION_GENAI_EMITTERS    span | span_metric | span_metric_event (default: span)
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT  true/false (default: false)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from _otel_helpers import load_dotenv, providers_already_configured

load_dotenv()


def setup_telemetry(console_output: bool = False) -> None:
    """Configure OpenTelemetry SDK and instrument FastMCP."""
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.environ.get("OTEL_SERVICE_NAME", "dev-assistant-client")
    resource = Resource.create({"service.name": service_name})

    trace_provider = TracerProvider(resource=resource)
    meter_readers: list = []

    if console_output:
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        meter_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5000))
        print("[client] Console exporters enabled", file=sys.stderr)

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            meter_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            print(f"[client] OTLP exporter → {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print("[client] OTLP exporter unavailable — install opentelemetry-exporter-otlp-proto-grpc", file=sys.stderr)

    trace.set_tracer_provider(trace_provider)
    if meter_readers:
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=meter_readers))

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor
    FastMCPInstrumentor().instrument()
    print(f"[client] FastMCP instrumentation applied — service: {service_name}", file=sys.stderr)


def _flush_telemetry() -> None:
    """Force-flush and shut down trace/metric providers before exit."""
    from opentelemetry import metrics, trace

    try:
        tp = trace.get_tracer_provider()
        if hasattr(tp, "force_flush"):
            tp.force_flush(timeout_millis=10_000)
        if hasattr(tp, "shutdown"):
            tp.shutdown()
    except Exception:
        pass

    try:
        mp = metrics.get_meter_provider()
        if hasattr(mp, "force_flush"):
            mp.force_flush(timeout_millis=10_000)
        if hasattr(mp, "shutdown"):
            mp.shutdown()
    except Exception:
        pass


async def run_demo(server_url: str | None, use_http: bool) -> None:
    """Exercise the Development Assistant tools via the MCP client."""
    from fastmcp import Client

    print("\n=== MCP Development Assistant — OTel Instrumentation Demo ===\n")

    if use_http:
        print(f"[client] Transport: Streamable-HTTP → {server_url}")
        target = server_url
    else:
        server_script = Path(__file__).parent / "dev_assistant_server.py"
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found: {server_script}")
        print(f"[client] Transport: stdio (spawning {server_script})")
        target = server_script  # FastMCP Client accepts a Path for stdio

    async with Client(target) as client:
        print("[client] Connected\n")

        print("1. Listing available tools...")
        tools = await client.list_tools()
        print(f"   → {len(tools)} tools: {[t.name for t in tools]}\n")

        print("2. Getting system info...")
        result = await client.call_tool("get_system_info", {})
        print(f"   → {str(result)[:120]}...\n")

        print("3. Listing current directory...")
        result = await client.call_tool("list_files", {"directory": "."})
        print(f"   → {str(result)[:120]}...\n")

        print("4. Checking Git status...")
        try:
            result = await client.call_tool("git_status", {"repo_path": "."})
            print(f"   → {str(result)[:120]}...\n")
        except Exception as e:
            print(f"   → skipped ({e})\n")

        print("5. Searching for Python functions...")
        result = await client.call_tool(
            "search_code",
            {"pattern": "def ", "directory": ".", "file_extensions": [".py"], "max_results": 5},
        )
        print(f"   → {str(result)[:120]}...\n")

        print("6. Writing then reading a test file...")
        test_file = "/tmp/mcp_otel_devassist_test.txt"
        await client.call_tool("write_file", {
            "file_path": test_file,
            "content": "Hello from MCP Development Assistant + OpenTelemetry!\n",
        })
        result = await client.call_tool("read_file", {"file_path": test_file, "max_lines": 5})
        print(f"   → {str(result)[:120]}...\n")

    print("=== Demo complete — check Splunk O11y for traces ===")


async def main(use_http: bool, server_url: str | None, console_output: bool, wait_seconds: int) -> None:
    if providers_already_configured():
        from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor
        FastMCPInstrumentor().instrument()
        print("[client] Providers already configured (running under opentelemetry-instrument)", file=sys.stderr)
    elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or console_output:
        setup_telemetry(console_output=console_output)
    else:
        print("[client] No OTel provider configured — skipping telemetry setup.", file=sys.stderr)

    try:
        await run_demo(server_url, use_http)
    except KeyboardInterrupt:
        print("\n[client] Interrupted.")
    except Exception as e:
        print(f"\n[client] Demo failed: {e}")
        import traceback
        traceback.print_exc()

    if wait_seconds > 0:
        print(f"\n[client] Waiting {wait_seconds}s for telemetry flush...", file=sys.stderr)
        await asyncio.sleep(wait_seconds)

    _flush_telemetry()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Development Assistant Client")
    parser.add_argument("--http", action="store_true",
                        help="Connect to an HTTP server instead of spawning a stdio sub-process")
    parser.add_argument("--server-url", default="http://localhost:8001/mcp",
                        help="HTTP server URL (HTTP mode only, default: http://localhost:8001/mcp)")
    parser.add_argument("--console", action="store_true",
                        help="Enable console span/metric exporters for local debugging")
    parser.add_argument("--wait", type=int, default=5, metavar="SECONDS",
                        help="Seconds to wait after demo for telemetry flush (default: 5)")
    args = parser.parse_args()

    asyncio.run(main(
        use_http=args.http,
        server_url=args.server_url,
        console_output=args.console,
        wait_seconds=args.wait,
    ))
