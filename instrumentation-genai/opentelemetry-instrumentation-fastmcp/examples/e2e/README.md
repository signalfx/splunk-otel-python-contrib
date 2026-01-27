# End-to-End MCP Instrumentation Example

This example demonstrates a complete end-to-end deployment of an MCP server and client
with OpenTelemetry instrumentation enabled.

## Overview

- **`server.py`**: A simple MCP server with calculator tools (no instrumentation)
- **`server_instrumented.py`**: Server with OpenTelemetry instrumentation built-in
- **`client.py`**: A client that connects to the server and calls tools
- **`run_demo.py`**: Orchestrates running both server and client together

## Prerequisites

1. Install the required packages:

```bash
# From the repository root
pip install -e ./instrumentation-genai/opentelemetry-instrumentation-fastmcp
pip install -e ./util/opentelemetry-util-genai

# Optional: For OTLP export to a backend
pip install opentelemetry-exporter-otlp
```

2. Verify FastMCP is installed:

```bash
pip install fastmcp
```

## Quick Start

### Option 1: Run the Demo Script (Single Terminal)

The easiest way to see everything working:

```bash
cd instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/e2e

# Run with console output (traces + metrics)
OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" python run_demo.py --console

# Run with just traces
python run_demo.py --console
```

### Option 2: Run Server and Client in Separate Terminals

This approach lets you see server-side and client-side telemetry separately,
each with its own service name for proper attribution in your observability backend.

The server runs in **SSE (Server-Sent Events) mode** to accept external connections.

#### Terminal 1 - Start the Instrumented Server

```bash
cd instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/e2e

# Set service name for the server
export OTEL_SERVICE_NAME="mcp-calculator-server"

# Enable metrics (optional)
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"

# For OTLP export (optional):
# export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Start the instrumented server in SSE mode
python server_instrumented.py --sse --port 8000
```

You should see:
```
✅ Console exporters enabled
✅ FastMCP instrumentation applied (service: mcp-calculator-server)
============================================================
MCP Calculator Server with OpenTelemetry Instrumentation
============================================================

🌐 Starting SSE server at http://localhost:8000/sse
   Connect with: python client.py --server-url http://localhost:8000/sse --console

Press Ctrl+C to stop.
```

#### Terminal 2 - Run the Client

```bash
cd instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/e2e

# Set a DIFFERENT service name for the client
export OTEL_SERVICE_NAME="mcp-calculator-client"

# Enable metrics
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"

# For OTLP export (optional - same endpoint as server):
# export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Connect to the SSE server
python client.py --server-url http://localhost:8000/sse --console --wait 10
```

#### What You'll See

**In Terminal 1 (Server):** Server-side spans for tool executions with `service.name: mcp-calculator-server`

**In Terminal 2 (Client):** Client-side spans and metrics with `service.name: mcp-calculator-client`

**Both share the same `trace_id`** - The instrumentation automatically propagates trace context
from client to server via the MCP protocol's `_meta` field, enabling distributed tracing.

This separation is useful for:
- Debugging client vs server issues
- Monitoring each component independently
- Understanding the full request flow across services
- Validating that both client and server telemetry are working correctly

### Trace Context Propagation

The FastMCP instrumentation automatically propagates W3C TraceContext (traceparent, tracestate)
between client and server processes. This means:

1. **Client spans and server spans share the same `trace_id`**
2. Server tool execution spans are **children** of client tool call spans
3. Works for all MCP transports: stdio, SSE, streamable-http

The propagation is transparent - no code changes needed in your MCP server or client.

## Sending Telemetry to an OTLP Backend

To send telemetry to Splunk Observability Cloud, Jaeger, or any OTLP-compatible backend:

```bash
# Set OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Set service name
export OTEL_SERVICE_NAME="mcp-e2e-demo"

# Enable content capture (optional - captures tool arguments/results)
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"

# Enable metrics
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"

# Run the demo
python run_demo.py --wait 30
```

## What You'll See

### Traces

The instrumentation creates spans for:

1. **Client Session** (`invoke_agent mcp.client`)
   - Parent span covering the entire client session

2. **Tool Calls** (`tool_call <tool_name>`)
   - One span per tool invocation
   - Attributes include:
     - `mcp.method.name`: "tools/call"
     - `gen_ai.tool.name`: Name of the tool
     - `gen_ai.operation.name`: "execute_tool"
     - `network.transport`: "pipe" (for stdio)

3. **Admin Operations** (`step list_tools`)
   - Spans for listing available tools

### Metrics

When `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"`:

- **`mcp.client.operation.duration`**: Duration of client-side MCP operations
- **`mcp.tool.output.size`**: Size of tool output in bytes (useful for tracking LLM context growth)

Metric attributes follow [OTel MCP Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/):
- `mcp.method.name`
- `gen_ai.tool.name`
- `gen_ai.operation.name`
- `network.transport`
- `error.type` (when operation fails)

## Example Output

```
MCP End-to-End Demo with OpenTelemetry Instrumentation
======================================================

📡 Starting MCP Calculator Server...
✅ Server ready

📋 Available Tools:
   - add: Add two numbers together
   - subtract: Subtract second number from first
   - multiply: Multiply two numbers
   - divide: Divide first number by second

🔧 Testing Tools:
   add(5, 3) = 8.0
   subtract(10, 4) = 6.0
   multiply(6, 7) = 42.0
   divide(20, 4) = 5.0

✅ Demo completed!
```

## Architecture

```
┌─────────────────┐     stdio      ┌─────────────────────────┐
│   MCP Client    │ ─────────────► │      MCP Server         │
│  (client.py)    │ ◄───────────── │  (server_instrumented.py│
│                 │                │   or server.py)         │
│ Instrumented:   │                │                         │
│ • Session spans │                │ Instrumented:           │
│ • Tool calls    │                │ • Tool spans            │
│ • Metrics       │                │ • Duration metrics      │
└─────────────────┘                └─────────────────────────┘
         │                                  │
         ▼                                  ▼
┌──────────────────────────────────────────────────────┐
│              OpenTelemetry Collector                 │
│    (or ConsoleExporter for local debugging)         │
└──────────────────────────────────────────────────────┘
```

## File Reference

| File | Description |
|------|-------------|
| `server.py` | Simple calculator server (no built-in telemetry) |
| `server_instrumented.py` | Server with OpenTelemetry setup included |
| `client.py` | Instrumented client with telemetry setup |
| `run_demo.py` | All-in-one demo orchestrator |

## Troubleshooting

### No metrics appearing?

Make sure you've set the emitters environment variable:

```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
```

### Traces not showing in backend?

1. Check the OTLP endpoint is correct
2. Verify the exporter packages are installed:
   ```bash
   pip install opentelemetry-exporter-otlp
   ```
3. Use `--wait 30` to ensure telemetry is flushed before exit

### Import errors?

Install packages in editable mode from repo root:

```bash
pip install -e ./instrumentation-genai/opentelemetry-instrumentation-fastmcp
pip install -e ./util/opentelemetry-util-genai
```
