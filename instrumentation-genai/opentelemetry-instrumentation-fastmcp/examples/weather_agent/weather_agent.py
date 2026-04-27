#!/usr/bin/env python3
"""
Weather Agent — MCP Client + LLM Orchestration

Demonstrates the full agentic loop:

    ┌───────────┐         ┌────────────┐         ┌─────┐
    │   Agent   │◄───────►│ MCP Server │         │ LLM │
    │(MCP Client)│  stdio  │(subprocess)│         │     │
    └─────┬─────┘         └────────────┘         └──┬──┘
          │                                          │
          │  1. initialize / tools/list              │
          │  2. user query ─────────────────────────►│
          │  3. tool_call ◄──────────────────────────│
          │  4. call_tool (MCP) ────►                │
          │  5. tool result  ◄──────                 │
          │  6. [messages + result] ────────────────►│
          │  7. final answer ◄───────────────────────│
          └──────────────────────────────────────────┘

Usage:
    export NVIDIA_API_KEY="nvapi-..."

    # --- Manual instrumentation (sets up providers in-process) ---
    python weather_agent.py --manual --console
    python weather_agent.py --manual --wait 10

    # --- Zero-code instrumentation (providers configured by the wrapper) ---
    opentelemetry-instrument python weather_agent.py --wait 10

    # Custom query:
    python weather_agent.py --manual --query "What's the weather in London?"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


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
        except ImportError:
            print("OTLP exporters not available", file=sys.stderr)

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


async def run_agent(user_query: str, manual: bool = True):
    """
    Run the weather agent loop:
      1. Connect to weather MCP server (spawned as subprocess via stdio)
      2. Discover available tools
      3. Send user query + tool definitions to LLM
      4. Execute any tool calls the LLM requests via MCP
      5. Return tool results to LLM for final answer
    """
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from openai import OpenAI

    # Prefer generic OPENAI_BASE_URL / OPENAI_API_KEY env vars (Azure or any
    # OpenAI-compatible endpoint), falling back to NVIDIA.
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY (Azure / OpenAI) or NVIDIA_API_KEY before running."
        )
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    model = os.environ.get("OPENAI_MODEL") or "nvidia/llama-3.3-nemotron-super-49b-v1"

    openai = OpenAI(base_url=base_url, api_key=api_key)
    server_script = str(Path(__file__).parent / "weather_server.py")

    print(f"\n{'=' * 60}")
    print("Weather Agent — MCP + LLM Agentic Loop")
    print(f"{'=' * 60}")
    print(f"\nUser: {user_query}")

    # MCP SDK only inherits a small env allowlist (HOME, PATH, …) when
    # env=None, so OTEL_* vars must be passed explicitly to the server.
    server_env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("OTEL_", "NVIDIA_", "VIRTUAL_ENV", "FASTMCP_"))
        or k in ("HOME", "PATH", "SHELL", "TERM", "USER", "LOGNAME")
    }
    server_env["OTEL_SERVICE_NAME"] = "weather-mcp-server"

    if manual:
        # Manual: server runs its own setup_server_telemetry()
        transport = StdioTransport(
            command=sys.executable,
            args=[server_script, "--manual"],
            env=server_env,
        )
    else:
        # Zero-code: opentelemetry-instrument auto-configures providers
        # and discovers FastMCPInstrumentor via entry points.
        transport = StdioTransport(
            command="opentelemetry-instrument",
            args=[sys.executable, server_script],
            env=server_env,
        )

    # --- Step 1: Connect to MCP server (spawns subprocess, stdio transport) ---
    async with Client(transport) as client:
        print(f"\n✅ Connected to MCP server: {Path(server_script).name}")

        # --- Step 2: Discover tools ---
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

        # --- Step 3: Send user query to LLM with tool definitions ---
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

        # --- Agentic loop: LLM calls until it produces a final text response ---
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

            # --- Step 4: Execute tool calls via MCP ---
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

                # --- Step 5: Append tool result for next LLM call ---
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


async def main(
    manual: bool = True,
    console_output: bool = False,
    wait_seconds: int = 0,
    query: str | None = None,
):
    if manual:
        setup_telemetry(console_output=console_output)

    user_query = (
        query
        or "I'm traveling to Tokyo for 5 days next week. What's the weather like and what should I pack?"
    )

    try:
        await run_agent(user_query, manual=manual)
    except KeyboardInterrupt:
        print("\n⏹️  Agent interrupted")
    except Exception as e:
        print(f"\n❌ Agent failed: {e}")
        import traceback

        traceback.print_exc()

    if wait_seconds > 0:
        print(f"⏳ Waiting {wait_seconds}s for telemetry flush...")
        await asyncio.sleep(wait_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Agent — MCP + LLM Demo")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual instrumentation (set up providers in-process). "
        "Omit to use zero-code via opentelemetry-instrument.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Enable console exporters (only with --manual)",
    )
    parser.add_argument(
        "--wait", type=int, default=0, help="Wait seconds for telemetry flush"
    )
    parser.add_argument("--query", type=str, default=None, help="Custom user query")
    args = parser.parse_args()

    asyncio.run(
        main(
            manual=args.manual,
            console_output=args.console,
            wait_seconds=args.wait,
            query=args.query,
        )
    )
