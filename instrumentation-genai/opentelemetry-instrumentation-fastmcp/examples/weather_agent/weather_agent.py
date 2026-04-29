#!/usr/bin/env python3
"""
Weather Agent — MCP Client + LLM Orchestration

Demonstrates the full agentic loop with both stdio and HTTP transports
and both manual and zero-code (opentelemetry-instrument) instrumentation modes.

    ┌───────────┐  stdio  ┌────────────────────┐     ┌─────┐
    │   Agent   │◄───────►│ MCP Server         │     │ LLM │
    │(MCP Client)│         │ (subprocess)       │     │     │
    └───────────┘         └────────────────────┘     └─────┘

    ┌───────────┐  HTTP   ┌────────────────────┐
    │   Agent   │◄───────►│ MCP Server         │
    │(MCP Client)│  /mcp  │ (standalone HTTP)  │
    └───────────┘         └────────────────────┘

Usage — Manual instrumentation (--manual sets up OTel providers in-process):

    source .env

    # stdio (server spawned as subprocess):
    python weather_agent.py --manual
    python weather_agent.py --manual --console --query "Weather in London?"

    # HTTP (server must be running separately):
    # Terminal 1:
    OTEL_SERVICE_NAME=weather-mcp-server python weather_server.py --manual --transport http
    # Terminal 2:
    python weather_agent.py --manual --transport http --wait 5

Usage — Zero-code instrumentation (opentelemetry-instrument auto-configures OTel):

    source .env

    # stdio (server spawned with opentelemetry-instrument):
    opentelemetry-instrument python weather_agent.py

    # HTTP (both processes wrapped independently):
    # Terminal 1:
    OTEL_SERVICE_NAME=weather-mcp-server opentelemetry-instrument python weather_server.py --transport http
    # Terminal 2:
    OTEL_SERVICE_NAME=weather-agent opentelemetry-instrument python weather_agent.py --transport http --wait 5
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


from _otel_helpers import load_dotenv as _load_dotenv
from _otel_helpers import providers_already_configured as _providers_already_configured


def setup_telemetry(console_output: bool = False):
    """Configure OpenTelemetry providers and apply FastMCP instrumentation."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", "weather-agent")}
    )

    trace_provider = TracerProvider(resource=resource)
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
                ConsoleMetricExporter(), export_interval_millis=5000
            )
        )

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
            print(f"🔭 OTLP exporters enabled → {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print(
                "⚠️  OTLP exporters not available (pip install opentelemetry-exporter-otlp)",
                file=sys.stderr,
            )
    else:
        print(
            "⚠️  OTEL_EXPORTER_OTLP_ENDPOINT not set — traces will NOT be exported to Splunk.\n"
            "    Set it in .env or export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317",
            file=sys.stderr,
        )

    service_name = os.environ.get("OTEL_SERVICE_NAME", "weather-agent")
    emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    print(
        f"🔭 service.name={service_name}  emitters={emitters}",
        file=sys.stderr,
    )

    trace.set_tracer_provider(trace_provider)
    if metric_readers:
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=metric_readers)
        )

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    FastMCPInstrumentor().instrument()
    OpenAIInstrumentor().instrument()


def _normalize_packing_args(args: dict) -> dict:
    """Normalize LLM-generated args to match the tool's exact parameter names.

    LLMs frequently rename or nest parameters. This bridges the gap between
    what the LLM sends and what the MCP tool schema expects.
    """
    # Flatten nested weather_data dict if the LLM wrapped everything
    if "weather_data" in args and isinstance(args["weather_data"], dict):
        args = {**args, **args.pop("weather_data")}

    return {
        "temperature": args.get("temperature")
        or args.get("temperature_celsius")
        or args.get("temp", 20),
        "condition": args.get("condition") or args.get("weather_condition", "Clear"),
        "days": args.get("days")
        or args.get("trip_days")
        or args.get("trip_duration", 3),
    }


async def run_agent(
    user_query: str,
    manual: bool = True,
    transport: str = "stdio",
    server_url: str = "http://localhost:8000/mcp",
):
    """
    Run the weather agent loop:
      1. Connect to weather MCP server (subprocess via stdio OR HTTP)
      2. Discover available tools
      3. Send user query + tool definitions to LLM
      4. Execute any tool calls the LLM requests via MCP
      5. Return tool results to LLM for final answer
    """
    from fastmcp import Client
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY (Azure / OpenAI) or NVIDIA_API_KEY before running."
        )
    base_url = (
        os.environ.get("OPENAI_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    )
    model = os.environ.get("OPENAI_MODEL") or "nvidia/llama-3.3-nemotron-super-49b-v1"

    openai = OpenAI(base_url=base_url, api_key=api_key)
    server_script = str(Path(__file__).parent / "weather_server.py")

    print(f"\n{'=' * 60}")
    print("Weather Agent — MCP + LLM Agentic Loop")
    print(f"{'=' * 60}")
    print(f"\nTransport mode : {transport}")
    print(f"User           : {user_query}")

    if transport == "stdio":
        # MCP SDK only inherits a small env allowlist (HOME, PATH, …) when
        # env=None, so OTEL_* vars must be passed explicitly to the server.
        server_env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("OTEL_", "NVIDIA_", "VIRTUAL_ENV", "FASTMCP_"))
            or k in ("HOME", "PATH", "SHELL", "TERM", "USER", "LOGNAME")
        }
        server_env["OTEL_SERVICE_NAME"] = "weather-mcp-server"
        server_env.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")

        from fastmcp.client.transports import StdioTransport

        if manual:
            server_target = StdioTransport(
                command=sys.executable,
                args=[server_script, "--manual"],
                env=server_env,
            )
        else:
            # Zero-code: opentelemetry-instrument wraps the server process and
            # auto-discovers FastMCPInstrumentor via the entry point.
            server_target = StdioTransport(
                command="opentelemetry-instrument",
                args=[sys.executable, server_script],
                env=server_env,
            )
    else:
        # HTTP mode: connect to an already-running server.
        # Manual:    OTEL_SERVICE_NAME=weather-mcp-server python weather_server.py --manual --transport http
        # Zero-code: OTEL_SERVICE_NAME=weather-mcp-server opentelemetry-instrument python weather_server.py --transport http
        print(f"\n🌐 Connecting to HTTP server: {server_url}")
        if manual:
            print(
                "   Server expected: python weather_server.py --manual --transport http"
            )
        else:
            print(
                "   Server expected: opentelemetry-instrument python weather_server.py --transport http"
            )
        server_target = server_url

    async with Client(server_target) as client:
        print("\n✅ Connected to MCP server")

        tools = await client.list_tools()
        print(f"📋 Tools available: {[t.name for t in tools]}")

        openai_tools = []
        for tool in tools:
            schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful travel assistant. Use the available tools to "
                    "get weather information and packing suggestions.\n\n"
                    "WORKFLOW:\n"
                    "1. Call get_weather with the city name.\n"
                    "2. Call get_packing_suggestions using EXACTLY these parameters:\n"
                    "   - temperature: the numeric celsius value from the weather result\n"
                    "   - condition: the condition string from the weather result\n"
                    "   - days: the number of trip days\n\n"
                    "IMPORTANT: Pass individual parameters, NOT a nested object."
                ),
            },
            {"role": "user", "content": user_query},
        ]

        while True:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                print(f"\n🤖 Assistant: {assistant_message.content}")
                break

            messages.append(assistant_message.model_dump())

            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                if fn_name == "get_packing_suggestions":
                    fn_args = _normalize_packing_args(fn_args)

                print(f"\n🔧 Tool call: {fn_name}({fn_args})")

                try:
                    result = await client.call_tool(fn_name, fn_args)

                    if hasattr(result, "content") and result.content:
                        content = result.content[0]
                        result_text = (
                            content.text if hasattr(content, "text") else str(content)
                        )
                    else:
                        result_text = str(result)
                except Exception as e:
                    result_text = f"ERROR: {e}"

                print(f"   ➜ Result: {result_text[:120]}...")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

    print(f"\n{'=' * 60}")
    print("✅ Agent completed")
    print(f"{'=' * 60}\n")


async def _flush_telemetry(wait_seconds: int) -> None:
    """Flush and shut down OTel providers so all spans/metrics reach the backend."""
    from opentelemetry import trace, metrics

    if wait_seconds > 0:
        print(f"⏳ Flushing telemetry ({wait_seconds}s)...", file=sys.stderr)
        await asyncio.sleep(wait_seconds)

    # Force-flush BatchSpanProcessor before the process exits.
    tp = trace.get_tracer_provider()
    if hasattr(tp, "force_flush"):
        tp.force_flush(timeout_millis=15_000)
    if hasattr(tp, "shutdown"):
        tp.shutdown()

    mp = metrics.get_meter_provider()
    if hasattr(mp, "force_flush"):
        mp.force_flush(timeout_millis=15_000)
    if hasattr(mp, "shutdown"):
        mp.shutdown()


async def main(
    manual: bool = False,
    console_output: bool = False,
    wait_seconds: int = 10,
    query: str | None = None,
    transport: str = "stdio",
    server_url: str = "http://localhost:8000/mcp",
):
    # Auto-setup telemetry when:
    #   (a) --manual flag was passed, OR
    #   (b) OTEL_EXPORTER_OTLP_ENDPOINT is set (loaded from .env) and
    #       opentelemetry-instrument has NOT already configured providers.
    should_setup = manual or (
        bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
        and not _providers_already_configured()
    )
    if should_setup:
        setup_telemetry(console_output=console_output)
    elif not _providers_already_configured():
        print(
            "⚠️  No OTel providers configured. Pass --manual or set OTEL_EXPORTER_OTLP_ENDPOINT\n"
            "    (or run with: opentelemetry-instrument python weather_agent.py)",
            file=sys.stderr,
        )

    user_query = (
        query
        or "I'm traveling to Tokyo for 5 days next week. What's the weather like and what should I pack?"
    )

    try:
        await run_agent(
            user_query, manual=should_setup, transport=transport, server_url=server_url
        )
    except KeyboardInterrupt:
        print("\n⏹️  Agent interrupted")
    except Exception as e:
        print(f"\n❌ Agent failed: {e}")
        import traceback

        traceback.print_exc()

    await _flush_telemetry(wait_seconds)


if __name__ == "__main__":
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Weather Agent — MCP + LLM Demo")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Force manual in-process OTel setup (default: auto when OTEL_EXPORTER_OTLP_ENDPOINT is set).",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Also write spans to console (useful for debugging).",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=10,
        help="Seconds to wait for telemetry flush (default: 10).",
    )
    parser.add_argument("--query", type=str, default=None, help="Custom user query")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: 'stdio' spawns server as subprocess (default), "
        "'http' connects to an already-running HTTP server.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000/mcp",
        help="HTTP server URL for --transport http (default: http://localhost:8000/mcp)",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            manual=args.manual,
            console_output=args.console,
            wait_seconds=args.wait,
            query=args.query,
            transport=args.transport,
            server_url=args.server_url,
        )
    )
