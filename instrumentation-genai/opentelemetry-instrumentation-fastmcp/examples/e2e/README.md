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

Three transports are supported:

| Transport | Server flag | Client URL | Notes |
|-----------|-------------|------------|-------|
| **stdio** | *(default)* | *(default)* | Single process, subprocess spawning |
| **SSE** | `--sse` | `http://host:port/sse` | Legacy HTTP, Server-Sent Events |
| **Streamable HTTP** | `--http` | `http://host:port/mcp` | Modern HTTP transport (recommended) |

#### Terminal 1 - Start the Instrumented Server

```bash
cd instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/e2e

export OTEL_SERVICE_NAME="mcp-calculator-server"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"

# SSE transport:
python server_instrumented.py --sse --port 8000

# Or Streamable HTTP transport:
# python server_instrumented.py --http --port 8000
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

export OTEL_SERVICE_NAME="mcp-calculator-client"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"

# Connect to SSE server:
python client.py --server-url http://localhost:8000/sse --console --wait 10

# Or connect to Streamable HTTP server:
# python client.py --server-url http://localhost:8000/mcp --console --wait 10
```

#### What You'll See

**In Terminal 1 (Server):** Server-side spans for tool executions with `service.name: mcp-calculator-server`

**In Terminal 2 (Client):** Client-side spans and metrics with `service.name: mcp-calculator-client`

**Both share the same `trace_id`** - The instrumentation automatically propagates trace context
from client to server via standard OTel Baggage, enabling distributed tracing.

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

## Session Propagation via OTel Baggage

The instrumentation supports propagating session context (`gen_ai.conversation.id`, `user.id`, `customer.id`)
across MCP client→server boundaries using [W3C Baggage](https://www.w3.org/TR/baggage/).

### Enable Baggage Propagation

```bash
# Enable baggage-based session propagation
export OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION="baggage"

# Optionally include session attributes in metrics (⚠️ cardinality)
export OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS="user.id"
# Or include all: "all" or "gen_ai.conversation.id,user.id,customer.id"
```

### Run with Session Context

```bash
# Terminal 1: Start the server (SSE or Streamable HTTP)
export OTEL_SERVICE_NAME="mcp-calculator-server"
export OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION="baggage"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
python server_instrumented.py --sse --port 8000
# or: python server_instrumented.py --http --port 8000

# Terminal 2: Run client with session
export OTEL_SERVICE_NAME="mcp-calculator-client"
export OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION="baggage"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
python client.py --server-url http://localhost:8000/sse --console \
    --session-id "conv-123" --user-id "user-456"
# or for HTTP: --server-url http://localhost:8000/mcp
```

### How It Works

1. **Client side**: `set_session_context()` stores session in both a `ContextVar` and
   OTel Baggage. When making MCP calls, `propagate.inject()` writes the `baggage`
   header into the carrier alongside `traceparent`/`tracestate`.

2. **Server side**: `propagate.extract()` restores both trace context and baggage.
   The transport instrumentor then calls `restore_session_from_context()` to populate
   the local session `ContextVar`, making `gen_ai.conversation.id`/`user.id` available to GenAI spans.

3. **Span attributes**: Session fields (`gen_ai.conversation.id`, `user.id`, `customer.id`) appear
   automatically on all GenAI spans via the `GenAI` base type's `semantic_convention_attributes()`.

4. **Metric attributes**: Optionally controlled via
   `OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS`.

---

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
     - `network.transport`: Detected automatically (`"pipe"` for stdio, `"tcp"` for SSE/HTTP)

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
┌─────────────────┐  stdio / SSE / ┌─────────────────────────┐
│   MCP Client    │  streamable-   │      MCP Server         │
│  (client.py)    │     http       │  (server_instrumented.py│
│                 │ ─────────────► │   or server.py)         │
│ Instrumented:   │ ◄───────────── │                         │
│ • Session spans │                │ Instrumented:           │
│ • Tool calls    │   OTel Baggage: │ • Tool spans            │
│ • Metrics       │   traceparent  │ • Duration metrics      │
│                 │   baggage      │                         │
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

## Running Tests

### Unit Tests

Run from the repository root:

```bash
# Install packages in editable mode
pip install -e ./util/opentelemetry-util-genai
pip install -e "./instrumentation-genai/opentelemetry-instrumentation-fastmcp[instruments,test]"

# Run all FastMCP instrumentation tests (111 tests)
pytest ./instrumentation-genai/opentelemetry-instrumentation-fastmcp/tests/ -v

# Run only transport detection tests
pytest ./instrumentation-genai/opentelemetry-instrumentation-fastmcp/tests/test_utils.py -v -k "Transport"

# Run propagation tests (trace context + baggage)
pytest ./instrumentation-genai/opentelemetry-instrumentation-fastmcp/tests/test_transport_propagation.py -v

# Run with coverage
pytest ./instrumentation-genai/opentelemetry-instrumentation-fastmcp/tests/ -v --cov=opentelemetry.instrumentation.fastmcp
```

### Lint

```bash
make lint
```

## Sending Telemetry to an OTel Collector

### 1. Start a Local Collector + Jaeger

Create a `docker-compose.yml`:

```yaml
version: '3'
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml

  jaeger:
    image: jaegertracing/jaeger:latest
    ports:
      - "16686:16686" # Jaeger UI
      - "4317"        # OTLP gRPC (internal)
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

Collector config (`otel-collector-config.yaml`):

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger, debug]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```

Start:

```bash
docker compose up -d
```

### 2. Set Service Names and Run

Each process should have a **distinct `OTEL_SERVICE_NAME`** so spans are attributed
correctly in the backend. The service name appears as `service.name` on every span.

**Terminal 1 — MCP Server:**

```bash
export OTEL_SERVICE_NAME="mcp-calculator-server"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"

# Streamable HTTP transport (recommended)
python server_instrumented.py --http --port 8000
```

**Terminal 2 — MCP Client:**

```bash
export OTEL_SERVICE_NAME="mcp-calculator-client"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"

python client.py --server-url http://localhost:8000/mcp --wait 10
```

### 3. View Traces

Open Jaeger UI at **http://localhost:16686** → search for service `mcp-calculator-client`.

You'll see a distributed trace spanning both services:

```
mcp-calculator-client: invoke_agent mcp.client
  └── mcp-calculator-client: execute_tool add         (client span)
        └── mcp-calculator-server: execute_tool add    (server span, same trace_id)
```

### Service Name Best Practices

| Env Var | Purpose | Example |
|---------|---------|-------|
| `OTEL_SERVICE_NAME` | Sets `service.name` resource attribute | `mcp-calculator-server` |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes | `deployment.environment=staging,service.version=1.0` |

Use distinct service names per process:
- `mcp-<app>-server` for the MCP server
- `mcp-<app>-client` for the MCP client
- `mcp-<app>-agent` for an AI agent that orchestrates MCP calls

This ensures correct service topology in your observability backend.

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
