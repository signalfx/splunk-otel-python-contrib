#!/usr/bin/env python3
"""
Instrumented MCP Calculator Server

A standalone MCP server with OpenTelemetry instrumentation enabled.
Run this in one terminal, then use client.py in another terminal.

Usage:
    # Terminal 1: Start the instrumented server (SSE mode for external clients)
    export OTEL_SERVICE_NAME="mcp-calculator-server"
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
    python server_instrumented.py --sse --port 8000

    # Terminal 2: Run the client connecting to the server
    export OTEL_SERVICE_NAME="mcp-calculator-client"
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
    python client.py --server-url http://localhost:8000/sse --console

    # Alternative: Run with Streamable HTTP transport
    python server_instrumented.py --http --port 8000
    python client.py --server-url http://localhost:8000/mcp --console

    # Alternative: Run in stdio mode (for subprocess spawning)
    python server_instrumented.py

For OTLP export to a backend:
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
    export OTEL_SERVICE_NAME="mcp-calculator-server"
    python server_instrumented.py --sse
"""

import argparse
import os
import sys


def setup_telemetry():
    """Set up OpenTelemetry for the server."""
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    service_name = os.environ.get("OTEL_SERVICE_NAME", "mcp-calculator-server")

    # Create resource with service info
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
        }
    )

    # Set up trace provider
    trace_provider = TracerProvider(resource=resource)
    metric_readers = []

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
    else:
        # Console output for debugging
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
                export_interval_millis=10000,
            )
        )
        print(
            "✅ Console exporters enabled (set OTEL_EXPORTER_OTLP_ENDPOINT for OTLP)",
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
    print(
        f"✅ FastMCP instrumentation applied (service: {service_name})", file=sys.stderr
    )


# Set up telemetry BEFORE importing FastMCP
setup_telemetry()

# Now import and create the MCP server
# (must be after setup_telemetry to apply instrumentation)
from fastmcp import FastMCP  # noqa: E402

# Create the MCP server
mcp = FastMCP("Calculator Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract second number from first.

    Args:
        a: First number
        b: Second number

    Returns:
        Difference (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide first number by second.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient (a / b)

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def calculate_expression(expression: str) -> str:
    """Evaluate a simple mathematical expression.

    Supports +, -, *, / operators with numbers.
    Example: "2 + 3 * 4"

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result as a string with the original expression
    """
    # Simple and safe evaluation for basic math
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"

    try:
        # Use eval with restricted builtins for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@mcp.tool()
def get_session_info() -> str:
    """Return the current session context as seen by the server.

    This tool validates that session context (gen_ai.conversation.id, user.id,
    customer.id) is properly propagated from the client via OTel Baggage.

    Returns:
        JSON string with the session context values visible on the server side.
    """
    import json

    result: dict[str, str | None] = {
        "gen_ai.conversation.id": None,
        "user.id": None,
        "customer.id": None,
    }

    # Try reading from OTel Baggage (cross-service propagation)
    try:
        from opentelemetry import baggage

        result["gen_ai.conversation.id"] = baggage.get_baggage("gen_ai.conversation.id")
        result["user.id"] = baggage.get_baggage("user.id")
        result["customer.id"] = baggage.get_baggage("customer.id")
    except ImportError:
        pass

    # Try reading from GenAI session context (contextvar propagation)
    try:
        from opentelemetry.util.genai.handler import get_session_context

        ctx = get_session_context()
        if ctx.session_id and not result["gen_ai.conversation.id"]:
            result["gen_ai.conversation.id"] = ctx.session_id
        if ctx.user_id and not result["user.id"]:
            result["user.id"] = ctx.user_id
        if ctx.customer_id and not result["customer.id"]:
            result["customer.id"] = ctx.customer_id
    except ImportError:
        pass

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Calculator Server with OpenTelemetry Instrumentation"
    )
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--sse",
        action="store_true",
        help="Run in SSE mode for external client connections (default: stdio)",
    )
    transport_group.add_argument(
        "--http",
        action="store_true",
        help="Run in Streamable HTTP mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to in SSE/HTTP mode (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on in SSE/HTTP mode (default: 8000)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60, file=sys.stderr)
    print("MCP Calculator Server with OpenTelemetry Instrumentation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if args.sse:
        print(
            f"\n🌐 Starting SSE server at http://{args.host}:{args.port}/sse",
            file=sys.stderr,
        )
        print(
            "   Connect with: python client.py --server-url "
            + f"http://{args.host}:{args.port}/sse --console",
            file=sys.stderr,
        )
        print("\nPress Ctrl+C to stop.\n", file=sys.stderr)
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.http:
        print(
            f"\n🌐 Starting Streamable HTTP server at "
            f"http://{args.host}:{args.port}/mcp",
            file=sys.stderr,
        )
        print(
            "   Connect with: python client.py --server-url "
            + f"http://{args.host}:{args.port}/mcp --console",
            file=sys.stderr,
        )
        print("\nPress Ctrl+C to stop.\n", file=sys.stderr)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        print("\n📡 Running in stdio mode (for subprocess spawning)", file=sys.stderr)
        print(
            "   Use --sse or --http flag for external client connections",
            file=sys.stderr,
        )
        print("\nServer is ready. Waiting for client connections...", file=sys.stderr)
        print("Press Ctrl+C to stop.\n", file=sys.stderr)
        mcp.run()
