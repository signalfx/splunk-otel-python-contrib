# End-to-End MCP Instrumentation Example

This example demonstrates a complete end-to-end deployment of an MCP server and client
with OpenTelemetry instrumentation enabled.

## Overview

- **`server.py`**: A simple MCP server with calculator tools (no instrumentation)
- **`server_instrumented.py`**: Server with OpenTelemetry instrumentation built-in
- **`client.py`**: A client that connects to the server and calls tools
- **`run_demo.py`**: Orchestrates running both server and client together

## Prerequisites

### Compatibility Matrix

| Component | Supported version range |
|-----------|-------------------------|
| `fastmcp` | `>=2.0.0, <=2.14.7` |
| `splunk-otel-util-genai` | `<=0.1.8` |

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

### Configure telemetry (.env)

A `.env` file is provided with Splunk OTLP settings. Load it before running:

```bash
source .env
```

### Option 1: stdio mode — Demo Script (Single Terminal)

The easiest way to see everything working (server spawned as subprocess):

```bash
source .env

# Traces + metrics, wait 5s for telemetry flush
python run_demo.py --console --wait 5

# Send to Splunk (OTLP from .env)
python run_demo.py --wait 5
```

### Option 2: HTTP mode — Demo Script (Single Terminal)

Uses Streamable-HTTP transport. `run_demo.py` spawns the server subprocess automatically:

```bash
source .env

# HTTP mode with console output
python run_demo.py --http --console --wait 5

# Custom port
python run_demo.py --http --port 8001 --wait 5
```

You will see `network.transport: tcp`, `network.protocol.name: http`, `server.address` and
`server.port` on each span.

### Option 3: Separate Terminals (HTTP — Streamable-HTTP)

Run server and client in separate processes for independent service names.

#### Terminal 1 — Start the Instrumented Server

```bash
source .env
OTEL_SERVICE_NAME=mcp-calculator-server python server_instrumented.py --http --port 8000
```

#### Terminal 2 — Run the Client

```bash
source .env
python client.py --server-url http://localhost:8000/mcp --wait 5
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

Set `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric` (already in `.env`).

When enabled:

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

### stdio transport
```
┌─────────────────┐     stdio      ┌─────────────────────────┐
│   MCP Client    │ ─────────────► │      MCP Server         │
│  (client.py)    │ ◄───────────── │  (server_instrumented.py│
│ Instrumented:   │                │ Instrumented:           │
│ • Session spans │                │ • Tool spans            │
│ • Tool calls    │                │ • Duration metrics      │
└─────────────────┘                └─────────────────────────┘
```

### HTTP transport (Streamable-HTTP)
```
┌─────────────────┐  HTTP/mcp   ┌────────────────────────────┐
│   MCP Client    │ ──────────► │  MCP Server (HTTP mode)    │
│  (client.py)    │ ◄────────── │  server_instrumented.py    │
│                 │             │  --http --port 8000        │
│ Span attrs:     │             │                            │
│ • network.transport: tcp      │ Span attrs:                │
│ • network.protocol.name: http │ • network.transport: tcp   │
│ • server.address / port       │ • client.address / port    │
│ • mcp.protocol.version        │ • mcp.session.id           │
└─────────────────┘             └────────────────────────────┘
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
