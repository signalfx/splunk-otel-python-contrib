# OpenTelemetry Cisco AI Defense Instrumentation

This package provides OpenTelemetry instrumentation for the [Cisco AI Defense Python SDK](https://github.com/cisco-ai-defense/ai-defense-python-sdk), enabling automatic telemetry capture for security inspection operations.

## Overview

Cisco AI Defense is a security guardrail for GenAI applications at runtime. This instrumentation captures security inspection events from the AI Defense SDK, adding the critical `gen_ai.security.event_id` span attribute for security event correlation in Splunk APM and other observability platforms.

### Primary Attribute

The key attribute captured is `gen_ai.security.event_id`, which is essential for:
- Correlating security events across distributed traces
- Filtering AI-specific telemetry in GDI pipelines
- Security incident investigation and analysis

## Architecture & Approach

### Design Philosophy

We treat AI Defense security inspections as **LLM invocations** because:
1. AI Defense internally uses LLM-based analysis to detect security violations
2. Each `inspect_prompt()` or `inspect_response()` call is semantically similar to an LLM call
3. This allows security spans to integrate naturally with existing GenAI telemetry

### Instrumentation Pattern

We use **monkey-patching via `wrapt`** to wrap AI Defense SDK methods:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│                                                              │
│   security.check_request("Find activities in Tokyo...")      │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │        AIDefenseInstrumentor (this package)          │   │
│   │                                                      │   │
│   │   1. Create LLMInvocation with security context     │   │
│   │   2. handler.start_llm(invocation)  ← Start span    │   │
│   │   3. Call original inspect_prompt()                 │   │
│   │   4. Extract event_id from result                   │   │
│   │   5. invocation.security_event_id = event_id        │   │
│   │   6. handler.stop_llm(invocation)   ← End span      │   │
│   └─────────────────────────────────────────────────────┘   │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │         TelemetryHandler (splunk-otel-util-genai)    │   │
│   │                                                      │   │
│   │   • Creates span: "chat cisco-ai-defense"           │   │
│   │   • Emits semantic convention attributes            │   │
│   │   • Records metrics (duration, tokens)              │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Why `LLMInvocation`?

We map AI Defense inspections to `LLMInvocation` (not `Step` or custom types) because:

| Aspect | Rationale |
|--------|-----------|
| **Semantic fit** | AI Defense uses LLM-based analysis internally |
| **Span naming** | Produces `chat cisco-ai-defense` spans (consistent with other LLMs) |
| **Attribute support** | Leverages existing `gen_ai.*` semantic conventions |
| **Trace integration** | Automatically nests under parent workflow spans |

### Attribute Emission Mechanism

The `gen_ai.security.event_id` attribute uses the **semconv metadata pattern**:

```python
# In opentelemetry-util-genai/src/opentelemetry/util/genai/types.py
@dataclass
class LLMInvocation(GenAI):
    # ... other fields ...
    
    security_event_id: Optional[str] = field(
        default=None,
        metadata={"semconv": GEN_AI_SECURITY_EVENT_ID},  # ← Key!
    )
```

The `semantic_convention_attributes()` method in the `GenAI` base class automatically:
1. Iterates over dataclass fields
2. Finds fields with `metadata={"semconv": ...}`
3. Emits them as span attributes

This is different from `gen_ai.input.messages` / `gen_ai.output.messages` which require explicit JSON serialization in the span emitter (controlled by `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`).

### Constant Definition Location

The attribute constant lives in the centralized attributes module:

```python
# In opentelemetry-util-genai/src/opentelemetry/util/genai/attributes.py
GEN_AI_SECURITY_EVENT_ID = "gen_ai.security.event_id"
```

This follows the pattern of other GenAI attributes and allows consistent reuse across instrumentations.

## Installation

```bash
pip install splunk-otel-instrumentation-aidefense
```

## Usage

### Programmatic Instrumentation

```python
from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

# Instrument AI Defense SDK
AIDefenseInstrumentor().instrument()

# Your AI Defense code
from aidefense.runtime import ChatInspectionClient

client = ChatInspectionClient(api_key="your-api-key")

# Spans are automatically created with gen_ai.security.event_id
result = client.inspect_prompt("How to hack a system?")
print(f"Safe: {result.is_safe}, Event ID: {result.event_id}")
```

### Auto-Instrumentation

Using OpenTelemetry auto-instrumentation:

```bash
opentelemetry-instrument --traces_exporter otlp python your_app.py
```

## Instrumented Methods

### ChatInspectionClient

| Method | Description |
|--------|-------------|
| `inspect_prompt` | Inspect user prompts for security violations |
| `inspect_response` | Inspect AI responses for security violations |
| `inspect_conversation` | Inspect full conversations |

### HttpInspectionClient

| Method | Description |
|--------|-------------|
| `inspect_request` | Inspect HTTP requests |
| `inspect_response` | Inspect HTTP responses |
| `inspect_request_from_http_library` | Inspect requests from `requests` library |
| `inspect_response_from_http_library` | Inspect responses from `requests` library |

## Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.security.event_id` | `string` | Unique event ID from AI Defense (only present when content is blocked) |
| `gen_ai.request.model` | `string` | Always `cisco-ai-defense` |
| `gen_ai.system` | `string` | Always `aidefense` |
| `server.address` | `string` | AI Defense API endpoint |

### How Attributes Are Emitted

The `gen_ai.security.event_id` attribute uses the semconv metadata pattern in `LLMInvocation`:

```python
# In opentelemetry-util-genai/types.py
security_event_id: Optional[str] = field(
    default=None,
    metadata={"semconv": GEN_AI_SECURITY_EVENT_ID},  # Auto-emitted to span
)
```

This is different from `gen_ai.input.messages` / `gen_ai.output.messages` which are handled explicitly in the span emitter with JSON serialization (controlled by `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`).

## Trace Integration

When used alongside other GenAI instrumentations (LangChain, CrewAI, etc.), AI Defense inspection spans automatically become children of the active trace:

```
POST /travel/plan
└── workflow LangGraph
    ├── step flight_specialist
    │   ├── chat cisco-ai-defense      ← AI Defense check (passed)
    │   ├── invoke_agent flight_specialist
    │   │   ├── step model → chat gpt-4o-mini
    │   │   └── step tools → tool mock_search_flights
    │   └── step should_continue
    ├── step hotel_specialist
    │   ├── chat cisco-ai-defense      ← AI Defense check (passed)
    │   └── invoke_agent hotel_specialist
    └── step activity_specialist
        └── chat cisco-ai-defense      ← AI Defense check (BLOCKED)
            └── gen_ai.security.event_id: "203d272b-d6b0-4c39-..."
```

## Example: Multi-Agent Travel Planner with Security

See the full example at `examples/langchain_with_aidefense/` in the LangChain instrumentation package.

```python
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

# Instrument LangChain first, then AI Defense
LangchainInstrumentor().instrument()
AIDefenseInstrumentor().instrument()

from aidefense.runtime import ChatInspectionClient

class SecurityGuard:
    def __init__(self, api_key: str):
        self.client = ChatInspectionClient(api_key=api_key)
    
    def check_request(self, agent_name: str, request: str) -> tuple[bool, str | None]:
        """Check if request is safe. Returns (is_safe, event_id)."""
        result = self.client.inspect_prompt(request)
        
        if not result.is_safe:
            return False, result.event_id  # event_id captured in span
        
        return True, None

# Usage in agent workflow
def activity_specialist_node(state, security: SecurityGuard):
    request = f"Find activities. User wants: {state['activities_request']}"
    
    is_safe, event_id = security.check_request("activity_specialist", request)
    if not is_safe:
        print(f"🚫 BLOCKED! Event ID: {event_id}")
        return state
    
    # Safe to proceed with LLM call...
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Set to `true` to capture full message content in spans |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint (e.g., `http://localhost:4317`) |

## Proxy Mode (Not Yet Supported)

This instrumentation currently supports **API mode only**. Proxy mode, where AI Defense adds the `x-cisco-ai-defense-tenant-api-key` header to responses, is not yet implemented.

## Requirements

- Python >= 3.9
- cisco-aidefense-sdk >= 2.0.0
- opentelemetry-api >= 1.38.0
- splunk-otel-util-genai >= 0.1.4

## References

- [Cisco AI Defense Python SDK](https://github.com/cisco-ai-defense/ai-defense-python-sdk)
- [AI Defense API Documentation](https://developer.cisco.com/docs/ai-defense/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Splunk OTel Python Contrib](https://github.com/signalfx/splunk-otel-python-contrib)

## Author

Aditya Mehra (admehra@cisco.com)

## License

Apache License 2.0
