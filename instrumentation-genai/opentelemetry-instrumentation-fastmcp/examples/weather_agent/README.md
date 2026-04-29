# Weather Agent — MCP + LLM Agentic Loop

This example demonstrates a full agentic loop where an **Agent** (MCP client) orchestrates
between an **MCP Server** (spawned as subprocess) and an **LLM** (OpenAI) to answer weather
and packing questions.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Agent as Agent (MCP Client)
    participant MCP as MCP Server (subprocess)
    participant LLM as LLM (OpenAI)

    Note over Agent,MCP: Spawns MCP Server subprocess and connects via stdio
    activate MCP
    MCP ->> MCP: Initialize and ready
    deactivate MCP

    Agent ->> MCP: initialize: {clientInfo, protocolVersion}
    activate MCP
    MCP -->> Agent: response: {capabilities, serverInfo}
    deactivate MCP

    Agent ->> MCP: tools/list
    activate MCP
    MCP -->> Agent: response: {tools: [{name: "get_weather", ...}, {name: "get_packing_suggestions", ...}]}
    deactivate MCP

    Agent ->> LLM: user: "I'm traveling to Tokyo for 5 days. What's the weather and what should I pack?"
    activate LLM
    Note over LLM: Determines it needs weather data first

    LLM ->> Agent: tool_call: get_weather(city="Tokyo")
    deactivate LLM
    activate Agent

    Agent ->> MCP: tools/call: get_weather {"city": "Tokyo"}
    activate MCP
    MCP -->> Agent: result: {"temperature_celsius": 22, "condition": "Partly cloudy", ...}
    deactivate MCP

    Agent ->> LLM: [messages + tool result: "22°C, Partly cloudy"]
    deactivate Agent
    activate LLM
    Note over LLM: Now has weather data, calls packing tool

    LLM ->> Agent: tool_call: get_packing_suggestions(temperature_celsius=22, condition="Partly cloudy", trip_days=5)
    deactivate LLM
    activate Agent

    Agent ->> MCP: tools/call: get_packing_suggestions {"temperature_celsius": 22, "condition": "Partly cloudy", "trip_days": 5}
    activate MCP
    MCP -->> Agent: result: {"clothing": [...], "accessories": [...], "tip": "..."}
    deactivate MCP

    Agent ->> LLM: [messages + tool result: packing list]
    deactivate Agent
    activate LLM
    Note over LLM: Has all information, produces final answer

    LLM ->> Agent: "Tokyo will be 22°C and partly cloudy. Pack light layers, jeans, sneakers..."
    deactivate LLM

    Note over Agent: Displays final answer to user

    Agent ->> MCP: Close stdin
    activate MCP
    MCP ->> MCP: Exits
    deactivate MCP
```

## Running the Example

### Prerequisites

```bash
pip install openai fastmcp
pip install -e ../../  # Install FastMCP instrumentation
```

Configure credentials and OTLP settings in `.env`:

```bash
source .env
```

---

### stdio mode — server spawned as subprocess

**Manual instrumentation** (`--manual` sets up OTel providers in-process):

```bash
source .env

python weather_agent.py --manual --console          # console output
python weather_agent.py --manual --wait 5           # send to Splunk
python weather_agent.py --manual --query "Weather in London?" --wait 5
```

**Zero-code instrumentation** (`opentelemetry-instrument` auto-configures OTel):

```bash
source .env

opentelemetry-instrument python weather_agent.py
opentelemetry-instrument python weather_agent.py --wait 5
```

---

### HTTP mode — server and agent run as separate processes

**Manual instrumentation:**

```bash
source .env

# Terminal 1: start the server
OTEL_SERVICE_NAME=weather-mcp-server python weather_server.py --manual --transport http

# Terminal 2: run the agent
python weather_agent.py --manual --transport http --wait 5
python weather_agent.py --manual --transport http --query "Weather in Sydney?" --wait 5
```

**Zero-code instrumentation:**

```bash
source .env

# Terminal 1: start the server with opentelemetry-instrument
OTEL_SERVICE_NAME=weather-mcp-server opentelemetry-instrument python weather_server.py --transport http

# Terminal 2: run the agent with opentelemetry-instrument
OTEL_SERVICE_NAME=weather-agent opentelemetry-instrument python weather_agent.py --transport http --wait 5
```

> **Note**: In zero-code mode the `opentelemetry-instrument` wrapper automatically discovers `FastMCPInstrumentor` via the `opentelemetry_instrumentor` entry point. No code changes needed.

---

The agent uses the model configured in `.env` (`OPENAI_MODEL`). Defaults to Azure OpenAI `gpt-4o-mini`.

### What Gets Instrumented

The OpenTelemetry instrumentation captures:

| Span | Description |
|------|-------------|
| `mcp.session` | Full lifecycle of the MCP client session |
| `tools/list` | Tool discovery call |
| `tools/call get_weather` | Individual tool invocation with args/result |
| `tools/call get_packing_suggestions` | Second tool invocation |
| `mcp.server.session.duration` | Server-side session metric |

With `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"`, you also get:
- `mcp.client.tool.duration` histogram
- `mcp.server.tool.duration` histogram
