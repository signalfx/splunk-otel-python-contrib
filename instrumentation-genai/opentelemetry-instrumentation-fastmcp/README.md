# OpenTelemetry FastMCP Instrumentation

[![PyPI](https://badge.fury.io/py/splunk-otel-instrumentation-fastmcp.svg)](https://pypi.org/project/splunk-otel-instrumentation-fastmcp/)

Automatic OpenTelemetry instrumentation for [FastMCP](https://github.com/gofastmcp/fastmcp) — the Python framework for building [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers and clients.

Produces spans, metrics, and optional events that follow the [OpenTelemetry GenAI MCP Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/).

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Transport Modes](#transport-modes)
  - [stdio (pipe)](#stdio-pipe)
  - [HTTP (Streamable-HTTP)](#http-streamable-http)
- [Configuration Reference](#configuration-reference)
  - [Service Name](#service-name)
  - [Environment Variables](#environment-variables)
- [What Is Instrumented](#what-is-instrumented)
- [Telemetry Reference](#telemetry-reference)
- [Examples](#examples)
  - [Dev Assistant as a Cursor / Claude Desktop MCP server](#dev-assistant-as-a-cursor--claude-desktop-mcp-server)
  - [Dev Assistant (stdio + HTTP)](#dev-assistant-stdio--http)
  - [Weather Agent (stdio + HTTP)](#weather-agent-stdio--http)
  - [End-to-End (e2e)](#end-to-end-e2e)
- [Trace Context Propagation](#trace-context-propagation)
- [Compatibility Matrix](#compatibility-matrix)

---

## Installation

```bash
pip install splunk-otel-instrumentation-fastmcp
```

With FastMCP pinned automatically:

```bash
pip install 'splunk-otel-instrumentation-fastmcp[instruments]'
```

---

## Quick Start

### Programmatic instrumentation

```python
from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

FastMCPInstrumentor().instrument()
```

Call this **before** creating your `FastMCP` server or `Client`.

### Zero-code instrumentation

```bash
opentelemetry-instrument python your_mcp_server.py
```

No code changes required. `opentelemetry-instrument` discovers the FastMCP entry point and applies instrumentation automatically at startup.

---

## Transport Modes

FastMCP supports two transports.  The instrumentation works identically for both, but the setup differs.

### stdio (pipe)

In stdio mode the **client spawns the server as a child process** and communicates over stdin/stdout pipes.  This is the default transport for local development and tools like Claude Desktop.

```
┌─────────────────────────────────────┐
│  Client process                     │
│                                     │
│  FastMCP Client  ←──────────────►  │
│  (instrumented)   stdin/stdout pipe │
│                        │            │
└────────────────────────┼────────────┘
                         │  (spawned)
┌────────────────────────▼────────────┐
│  Server process                     │
│                                     │
│  FastMCP Server                     │
│  (instrumented)                     │
│                                     │
│  ⚠ Do NOT write to stdout —         │
│    it is reserved for the MCP wire  │
│    protocol.  Use stderr for logs.  │
└─────────────────────────────────────┘
```

**Server** — run it by passing a `Path` or module to the FastMCP `Client`:

```python
# server.py
from fastmcp import FastMCP

server = FastMCP("my-server")

@server.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    server.run(transport="stdio")  # default
```

**Client** — spawn the server as a sub-process:

```python
# client.py
import asyncio
from pathlib import Path
from fastmcp import Client

async def main():
    async with Client(Path("server.py")) as client:
        result = await client.call_tool("add", {"a": 1, "b": 2})
        print(result)

asyncio.run(main())
```

**Telemetry setup for stdio**:

```bash
# .env
OTEL_SERVICE_NAME=my-mcp-server
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric
```

```bash
# Terminal 1 — set env, then run the client (it spawns the server)
source .env
OTEL_SERVICE_NAME=my-mcp-client python client.py
```

> **Important:** In stdio mode the server's `OTEL_SERVICE_NAME` must be set in the **server** process environment — either via `source .env` before running the client, or by explicitly passing it through `subprocess.Popen`/`env` when spawning.  The client and server get separate service names because they are separate processes.

### HTTP (Streamable-HTTP)

In HTTP mode the **server runs as a standalone process** and clients connect over the network using the MCP Streamable-HTTP transport.  This is the recommended mode for production deployments.

```
┌──────────────────────┐         ┌──────────────────────┐
│  Client process      │  HTTP   │  Server process       │
│                      │ ──────► │                       │
│  FastMCP Client      │  POST   │  FastMCP Server       │
│  (instrumented)      │ /mcp    │  (instrumented)       │
│                      │ ◄────── │                       │
│  network.transport:  │         │  network.transport:   │
│    tcp               │         │    tcp                │
│  network.protocol:   │         │  network.protocol:    │
│    http              │         │    http               │
└──────────────────────┘         └──────────────────────┘
```

**Server**:

```python
# server.py
from fastmcp import FastMCP

server = FastMCP("my-server")

@server.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    server.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

**Client**:

```python
# client.py
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as client:
        result = await client.call_tool("add", {"a": 1, "b": 2})
        print(result)

asyncio.run(main())
```

**Telemetry setup for HTTP**:

```bash
# Terminal 1 — start the server
source .env
OTEL_SERVICE_NAME=my-mcp-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    python server.py

# Terminal 2 — run the client
source .env
OTEL_SERVICE_NAME=my-mcp-client \
    python client.py
```

**Zero-code instrumentation for HTTP server**:

```bash
source .env
OTEL_SERVICE_NAME=my-mcp-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    opentelemetry-instrument python server.py
```

---

## Configuration Reference

### Service Name

Set `OTEL_SERVICE_NAME` to identify the service in Splunk Observability Cloud.

| Process | Recommended value | Example |
|---------|------------------|---------|
| MCP server | `<app>-mcp-server` | `weather-mcp-server` |
| MCP client | `<app>-mcp-client` | `weather-agent` |

In **stdio mode**, the client spawns the server as a subprocess.  Each process has its own `OTEL_SERVICE_NAME`.  Export the env var **before** starting the client so the server inherits it:

```bash
export OTEL_SERVICE_NAME=my-mcp-server   # inherited by the server sub-process
python client.py                         # client uses its own value if set separately
```

Or set both explicitly:

```bash
OTEL_SERVICE_NAME=my-mcp-server \
    python -c "import subprocess; subprocess.run(['python', 'client.py'], env={**os.environ, 'OTEL_SERVICE_NAME': 'my-mcp-client'})"
```

In **HTTP mode** each process sets its own `OTEL_SERVICE_NAME` independently.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name in Splunk O11y | `unknown_service` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC endpoint | *(not set)* |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers, e.g. `X-SF-Token=<token>` | *(not set)* |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` | `grpc` |
| `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE` | `DELTA` recommended for Splunk | *(not set)* |
| `OTEL_LOGS_EXPORTER` | `otlp` to export log-based events | *(not set)* |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Telemetry flavors (see below) | `span` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture tool args/results | `false` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | `SPAN`, `EVENT`, `SPAN_AND_EVENT` | `SPAN` |

#### `OTEL_INSTRUMENTATION_GENAI_EMITTERS`

Controls what telemetry is produced:

| Value | Spans | Metrics | Events |
|-------|-------|---------|--------|
| `span` | ✓ | — | — |
| `span_metric` | ✓ | ✓ | — |
| `span_metric_event` | ✓ | ✓ | ✓ |

For Splunk Observability Cloud use `span_metric` to get both APM traces and Infrastructure metrics.

#### Minimal `.env` for Splunk

```bash
# .env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA
OTEL_LOGS_EXPORTER=otlp
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

---

## What Is Instrumented

### Server-side (FastMCP 3.x — v0.2.0+)

| Hook | MCP method | Span name |
|------|-----------|-----------|
| `FastMCP.__init__` | — | captures server name |
| `Server.run` | `initialize` | session root span |
| `FastMCP.call_tool` | `tools/call` | `{tool_name}.tool` |
| `FastMCP.read_resource` | `resources/read` | resource span |
| `FastMCP.render_prompt` | `prompts/get` | prompt span |

### Client-side

| Hook | MCP method | Span name |
|------|-----------|-----------|
| `Client.__aenter__` | `initialize` | session root span |
| `Client.__aexit__` | — | closes session span |
| `Client.call_tool` | `tools/call` | `{tool_name}.tool` |
| `Client.list_tools` | `tools/list` | span |
| `Client.read_resource` | `resources/read` | span |
| `Client.get_prompt` | `prompts/get` | span |

### Transport-level

- **Trace context propagation** via W3C `traceparent`/`tracestate` injected into `params._meta` — connects client and server spans into a single distributed trace.
- **HTTP metadata** (when using Streamable-HTTP): `network.transport=tcp`, `network.protocol.name=http`, `network.protocol.version`, `client.address`, `mcp.session.id`.
- **stdio metadata**: `network.transport=pipe`.

---

## Telemetry Reference

### Spans

| Span | Attributes |
|------|-----------|
| `initialize` (client) | `mcp.session.id`, `server.address`, `server.port`, `mcp.protocol.version` |
| `initialize` (server) | `sdot.mcp.server_name`, `network.transport`, `network.protocol.name` |
| `{tool}.tool` | `mcp.tool.name`, `mcp.tool.output.size`, `error.type` (on failure) |
| `resources/read` | `mcp.resource.uri`, `network.transport`, `client.address` |
| `prompts/get` | `gen_ai.prompt.name`, `network.transport` |

### Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `mcp.client.operation.duration` | `s` | Histogram — client-side MCP operation latency (tools/call, tools/list, etc.) |
| `mcp.server.operation.duration` | `s` | Histogram — server-side MCP operation latency |
| `mcp.client.session.duration` | `s` | Histogram — full MCP session duration as seen by the client |
| `mcp.server.session.duration` | `s` | Histogram — full MCP session duration as seen by the server |
| `mcp.tool.output.size` | `{byte}` | Histogram — size of tool call output (impacts LLM token usage when used as context) |

All histograms carry `mcp.method.name`, `network.transport`, and `gen_ai.tool.name` (for tool metrics) as dimensions.
Histogram data points include **exemplars** (Trace ID + Span ID) for trace-metric correlation in Splunk APM.

### Events (when content capture is enabled)

| Event | Description |
|-------|-------------|
| `mcp.tool.input` | Tool call arguments |
| `mcp.tool.output` | Tool call result |

---

## Examples

All examples look for a `.env` file in their directory.  Copy `.env.example` to `.env` and fill in your Splunk token / endpoint.

---

### Dev Assistant as a Cursor / Claude Desktop MCP server

The `dev_assistant_server.py` ships as a fully observable **stdio MCP server** that you can wire directly into Cursor or Claude Desktop.  Because it runs in stdio mode, the host application (Cursor/Claude Desktop) spawns it as a sub-process — no separate terminal needed.

#### What tools it exposes

| Tool | Description |
|------|-------------|
| `list_files` | List files in a directory |
| `read_file` | Read a file's contents |
| `write_file` | Write or overwrite a file |
| `run_command` | Execute a shell command |
| `git_status` | Get `git status` for a repo |
| `search_code` | Search for a pattern in files |
| `get_system_info` | Return OS/Python/memory info |

#### Prerequisites

```bash
# 1. Install the package and its dependencies
pip install 'splunk-otel-instrumentation-fastmcp[instruments]'
pip install 'opentelemetry-sdk' 'opentelemetry-exporter-otlp'

# 2. (optional) Install zero-code bootstrap
pip install 'opentelemetry-distro'
opentelemetry-bootstrap -a install
```

#### Cursor IDE setup (`.cursor/mcp.json`)

Create or edit `.cursor/mcp.json` at the root of your workspace:

```json
{
  "mcpServers": {
    "dev-assistant": {
      "command": "/path/to/.venv/bin/python",
      "args": [
        "/path/to/splunk-otel-python-contrib/instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/dev_assistant_server.py"
      ],
      "env": {
        "OTEL_SERVICE_NAME": "dev-assistant-mcp",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "DELTA",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS": "span_metric",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"
      }
    }
  }
}
```

After saving, **reload the Cursor window** (`Cmd+Shift+P` → *Reload Window*).  Cursor will start the server automatically.

> **Tip:** Use `opentelemetry-instrument` as the command for zero-code instrumentation:
> ```json
> {
>   "command": "/path/to/.venv/bin/opentelemetry-instrument",
>   "args": [
>     "python",
>     "/path/to/.../dev_assistant_server.py"
>   ],
>   "env": { ... }
> }
> ```

#### Claude Desktop setup (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "dev-assistant": {
      "command": "/path/to/.venv/bin/python",
      "args": [
        "/path/to/.../dev_assistant_server.py"
      ],
      "env": {
        "OTEL_SERVICE_NAME": "dev-assistant-mcp",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "DELTA",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS": "span_metric"
      }
    }
  }
}
```

#### Expected telemetry in Splunk Observability Cloud

Each tool call from the AI assistant appears as an MCP span:

```
Cursor / Claude Desktop (host process)
  └── initialize                [CLIENT]  ← session root, network.transport=pipe
        ├── tools/list          [CLIENT]
        ├── tools/call list_files  [CLIENT]
        │     └── tools/call list_files  [SERVER]  ← dev-assistant-mcp service
        ├── tools/call read_file   [CLIENT]
        │     └── tools/call read_file   [SERVER]
        └── ...
```

Key span attributes:

| Attribute | Value |
|-----------|-------|
| `gen_ai.system` | `mcp` |
| `network.transport` | `pipe` (stdio) |
| `sdot.mcp.server_name` | `dev-assistant` |
| `mcp.protocol.version` | `2025-11-25` |

> **Note:** Cursor and Claude Desktop host processes do not yet emit their own client `initialize` span — the server-side root span carries the full context.  Client-side spans for hosts using the raw MCP SDK (not `fastmcp.Client`) are tracked as a follow-up.

---

### Dev Assistant (stdio + HTTP)

A multi-tool assistant with `list_files`, `read_file`, `write_file`, `run_command`, `git_status`, `search_code`, and `get_system_info`.

```
examples/
├── dev_assistant_server.py   # server (stdio or HTTP)
├── dev_assistant_client.py   # client (stdio or HTTP)
└── .env                      # OTLP config (copy from .env.example)
```

**stdio** — client spawns server automatically:

```bash
cd examples/
source .env
OTEL_SERVICE_NAME=dev-assistant-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    python dev_assistant_client.py
```

**HTTP** — two separate processes:

```bash
# Terminal 1
cd examples/
source .env
OTEL_SERVICE_NAME=dev-assistant-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    python dev_assistant_server.py --http --port 8001

# Terminal 2
cd examples/
source .env
OTEL_SERVICE_NAME=dev-assistant-client \
    python dev_assistant_client.py --http --server-url http://localhost:8001/mcp
```

**Zero-code (HTTP)**:

```bash
# Terminal 1 — server
source .env
OTEL_SERVICE_NAME=dev-assistant-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    opentelemetry-instrument python dev_assistant_server.py --http

# Terminal 2 — client
source .env
OTEL_SERVICE_NAME=dev-assistant-client \
    opentelemetry-instrument python dev_assistant_client.py \
        --http --server-url http://localhost:8001/mcp
```

**Local debugging** (console spans, no Splunk):

```bash
python dev_assistant_client.py --console
```

### Weather Agent (stdio + HTTP)

An OpenAI-powered agent that calls a weather MCP server.

```
examples/weather_agent/
├── weather_server.py   # FastMCP weather server
├── weather_agent.py    # OpenAI agent client
└── .env                # OTLP config + OPENAI_API_KEY
```

**stdio**:

```bash
cd examples/weather_agent/
source .env
OTEL_SERVICE_NAME=weather-mcp-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    python weather_agent.py
```

**HTTP** — two terminals:

```bash
# Terminal 1
cd examples/weather_agent/
source .env
OTEL_SERVICE_NAME=weather-mcp-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    python weather_server.py --http --port 8000

# Terminal 2
cd examples/weather_agent/
source .env
OTEL_SERVICE_NAME=weather-agent \
    python weather_agent.py --http --server-url http://localhost:8000/mcp
```

**Zero-code HTTP server + manual client**:

```bash
# Terminal 1
source .env
OTEL_SERVICE_NAME=weather-mcp-server \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
    opentelemetry-instrument python weather_server.py --http

# Terminal 2
source .env
OTEL_SERVICE_NAME=weather-agent \
    python weather_agent.py --http --server-url http://localhost:8000/mcp
```

### End-to-End (e2e)

A self-contained demo that starts a calculator server and runs a client against it.

```
examples/e2e/
├── run_demo.py          # orchestrator: spawns server, runs client
├── server.py            # bare server (no built-in telemetry)
├── server_instrumented.py  # server with manual OTel setup
├── client.py            # demo client
└── .env
```

**stdio** (run_demo.py spawns the server):

```bash
cd examples/e2e/
source .env
OTEL_SERVICE_NAME=mcp-calculator python run_demo.py
```

**HTTP**:

```bash
cd examples/e2e/
source .env
OTEL_SERVICE_NAME=mcp-calculator python run_demo.py --http --port 8000
```

---

## Trace Context Propagation

The MCP Python SDK v1.x does not natively propagate W3C trace context.  This instrumentation includes a **transport-layer bridge** (`transport_instrumentor.py`) that handles it automatically:

- **Client side**: injects `traceparent`, `tracestate`, and `baggage` into `params._meta` before every request.
- **Server side**: extracts the context from `request_meta` and attaches it via a `ContextVar` (`MCPRequestContext`) so the server instrumentor can link spans to the same trace.

This means client spans and server spans share the same `trace_id` — server tool execution spans appear as children of client tool call spans in Splunk APM.

```
Client trace
└── initialize (client)
    └── tools/call: add_numbers (client)
        └── tools/call: add_numbers (server)  ← same trace_id
```

**Upstream note**: Native OTel support was merged to the MCP Python SDK `main` branch (targeting v2.x — not yet released as of Apr 2026).  Once `mcp >= 2.x` is the minimum requirement, the transport bridge can be simplified.

---

## Compatibility Matrix

| Instrumentation version | fastmcp | util-genai | Notes |
|------------------------|---------|-----------|-------|
| 0.1.x | 2.x (jlowin/fastmcp) | ≤ 0.1.9 | Wraps `ToolManager.call_tool` |
| 0.2.x | ≥ 3.0.0, < 4 | ≥ 0.1.12 | Wraps `FastMCP.call_tool`, `read_resource`, `render_prompt`; HTTP transport metadata |

---

## References

- [FastMCP 3.x](https://github.com/gofastmcp/fastmcp)
- [FastMCP 2.x](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [OTel GenAI MCP Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/)
- [Splunk Distro for OTel Python](https://github.com/signalfx/splunk-otel-python)
