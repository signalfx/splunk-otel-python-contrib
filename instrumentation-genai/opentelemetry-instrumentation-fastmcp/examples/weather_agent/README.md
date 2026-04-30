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
export NVIDIA_API_KEY="nvapi-..."
```

### Quick Start

```bash
# With console trace output
python weather_agent.py --console

# Custom query
python weather_agent.py --query "What's the weather in London and what should I pack for a rainy day?" --console

# With OTLP export (e.g., to Splunk O11y)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
python weather_agent.py --wait 10
```

The agent uses **NVIDIA Nemotron** (`nvidia/llama-3.3-nemotron-super-49b-v1`) via the
OpenAI-compatible API at `https://integrate.api.nvidia.com/v1`.

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
