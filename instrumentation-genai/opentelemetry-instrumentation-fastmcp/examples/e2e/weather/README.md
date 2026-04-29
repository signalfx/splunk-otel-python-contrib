# Weather MCP Server — End-to-End Example

A single MCP server that exposes **tools**, **resources**, and **prompts** over
**Streamable HTTP** transport, with two clients:

- **`client.py`** — exercises every MCP primitive individually (tools, resources, prompts, error path)
- **`agent.py`** — trip-planning agent that chains tool calls into a multi-step workflow

All three scripts use **zero-code instrumentation** via `opentelemetry-instrument`.

## Quick start

Both server and client are plain application code — no manual OTel SDK setup.
Use `opentelemetry-instrument` (zero-code) to attach tracing and metrics:

```bash
# Terminal 1 — start the server (HTTP on port 8000)
OTEL_SERVICE_NAME=weather-mcp-server \
OTEL_TRACES_EXPORTER=console \
OTEL_METRICS_EXPORTER=console \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
  opentelemetry-instrument python server.py

# Terminal 2a — run the client (exercises all primitives)
OTEL_SERVICE_NAME=weather-mcp-client \
OTEL_TRACES_EXPORTER=console \
OTEL_METRICS_EXPORTER=console \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
  opentelemetry-instrument python client.py

# Terminal 2b — or run the trip-planning agent (chained tool calls)
OTEL_SERVICE_NAME=trip-planning-agent \
OTEL_TRACES_EXPORTER=console \
OTEL_METRICS_EXPORTER=console \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
  opentelemetry-instrument python agent.py --origin "New York" --destination Tokyo --days 5
```

To export to an OTel collector instead of console:

```bash
OTEL_SERVICE_NAME=weather-mcp-server \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
  opentelemetry-instrument python server.py
```

### Prerequisites

```bash
pip install fastmcp opentelemetry-distro opentelemetry-exporter-otlp-proto-grpc
pip install -e ./path/to/opentelemetry-instrumentation-fastmcp
```

## What the server exposes

| Primitive | Name | Description |
|-----------|------|-------------|
| **Tool** | `get_current_weather(city)` | Current conditions for a city |
| **Tool** | `get_forecast(city, days)` | Multi-day forecast |
| **Tool** | `get_travel_packing(destination, days)` | Packing list based on weather |
| **Tool** | `book_flight(origin, dest, dates)` | Simulated flight booking |
| **Resource** | `climate://cities` | List of supported cities |
| **Resource** | `climate://{city}/annual` | Annual climate summary |
| **Prompt** | `weather_briefing(city)` | Weather briefing template |
| **Prompt** | `travel_packing_advice(destination, days)` | Packing advice template |

## What the client exercises

| Step | MCP method | Target |
|------|-----------|--------|
| 1 | `tools/list` | — |
| 2 | `tools/call` | `get_current_weather("London")` |
| 3 | `tools/call` | `get_forecast("Tokyo", 3)` |
| 4 | `tools/call` | `get_travel_packing("Sydney", 7)` |
| 5 | `resources/list` | — |
| 6 | `resources/read` | `climate://cities` |
| 7 | `resources/read` | `climate://london/annual` |
| 8 | `prompts/list` | — |
| 9 | `prompts/get` | `weather_briefing(city="New York")` |
| 10 | `prompts/get` | `travel_packing_advice(destination="Mumbai", days=4)` |
| 11 | `tools/call` | `get_current_weather("Atlantis")` — **error path** |

## Trip-planning agent workflow (`agent.py`)

Chains 5 MCP operations where each step's output informs the next:

| Step | MCP method | Tool / Prompt | Purpose |
|------|-----------|--------------|---------|
| 1 | `tools/call` | `get_current_weather` | Check conditions at destination |
| 2 | `tools/call` | `get_forecast` | Look ahead for the trip window |
| 3 | `tools/call` | `get_travel_packing` | Build packing list from weather |
| 4 | `tools/call` | `book_flight` | Reserve flights |
| 5 | `prompts/get` | `weather_briefing` | Generate departure-day briefing |

All 5 steps produce `SpanKind.CLIENT` spans under a single `invoke_agent mcp.client`
parent span, giving end-to-end visibility of the agent workflow.

---

## Telemetry wireframe

### CLIENT spans (console exporter output)

#### tools/list

```
name:       tools/list
kind:       SpanKind.CLIENT
attributes:
  mcp.method.name:            tools/list
  gen_ai.system:              mcp
  gen_ai.agent.name:          mcp.client
  network.transport:          tcp
  network.protocol.name:      http
  network.protocol.version:   1.1
  server.address:             127.0.0.1
  server.port:                8000
```

#### tools/call (execute_tool)

```
name:       tools/call get_current_weather
kind:       SpanKind.CLIENT
attributes:
  gen_ai.operation.name:      execute_tool
  gen_ai.provider.name:       mcp
  gen_ai.agent.name:          mcp.client
  gen_ai.tool.name:           get_current_weather
  gen_ai.tool.call.id:        <uuid>
  gen_ai.tool.type:           extension
  mcp.method.name:            tools/call
  mcp.session.id:             <session-hex>
  network.transport:          tcp
  network.protocol.name:      http
  network.protocol.version:   1.1
  server.address:             127.0.0.1
  server.port:                8000
```

#### resources/read

```
name:       resources/read climate://cities
kind:       SpanKind.CLIENT
attributes:
  mcp.method.name:            resources/read
  mcp.resource.uri:           climate://cities
  mcp.session.id:             <session-hex>
  network.transport:          tcp
  network.protocol.name:      http
  network.protocol.version:   1.1
  server.address:             127.0.0.1
  server.port:                8000
```

#### prompts/get

```
name:       prompts/get weather_briefing
kind:       SpanKind.CLIENT
attributes:
  mcp.method.name:            prompts/get
  mcp.session.id:             <session-hex>
  network.transport:          tcp
  network.protocol.name:      http
  network.protocol.version:   1.1
  server.address:             127.0.0.1
  server.port:                8000
```

#### Error path (tools/call with unknown city)

```
name:       tools/call get_current_weather
kind:       SpanKind.CLIENT
status:     ERROR
attributes:
  mcp.method.name:            tools/call
  error.type:                 ToolError
  mcp.session.id:             <session-hex>
  network.transport:          tcp
  server.address:             127.0.0.1
  server.port:                8000
```

---

### Metrics (mcp.client.operation.duration)

Histogram buckets emitted per-operation with HTTP transport attributes:

| mcp.method.name | gen_ai.tool.name | gen_ai.prompt.name | error.type | count | sample sum |
|-----------------|-----------------|-------------------|------------|------:|-----------:|
| `tools/list` | — | — | — | 1 | 0.011 s |
| `tools/call` | `get_current_weather` | — | — | 1 | 0.005 s |
| `tools/call` | `get_forecast` | — | — | 1 | 0.004 s |
| `tools/call` | `get_travel_packing` | — | — | 1 | 0.004 s |
| `resources/read` | — | — | — | 2 | 0.020 s |
| `prompts/get` | — | `weather_briefing` | — | 1 | 0.711 s |
| `prompts/get` | — | `travel_packing_advice` | — | 1 | 0.600 s |
| `tools/call` | `get_current_weather` | — | `ToolError` | 1 | 0.040 s |

All metric data points carry `network.transport=tcp`, `network.protocol.name=http`,
`network.protocol.version=1.1`, `server.address=127.0.0.1`, `server.port=8000`.

---

### Attribute reference (HTTP transport)

| Attribute | Source | Example |
|-----------|--------|---------|
| `network.transport` | Transport detection | `tcp` |
| `network.protocol.name` | Inferred from transport | `http` |
| `network.protocol.version` | Starlette scope / httpx | `1.1` |
| `server.address` | Client transport URL | `127.0.0.1` |
| `server.port` | Client transport URL | `8000` |
| `client.address` | Starlette request (server-side) | `127.0.0.1` |
| `client.port` | Starlette request (server-side) | `54321` |
| `mcp.session.id` | MCP session header | `214b1fcd...` |
| `mcp.protocol.version` | Initialize handshake | `2025-03-26` |
| `error.type` | Exception class / tool error | `ToolError` |
