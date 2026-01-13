# OpenTelemetry Cisco AI Defense Instrumentation

This package provides OpenTelemetry instrumentation for [Cisco AI Defense](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html), enabling automatic telemetry capture for security inspection operations.

## Overview

Cisco AI Defense is a security guardrail for GenAI applications at runtime. This instrumentation captures security events, adding the critical `gen_ai.security.event_id` span attribute for security event correlation in Splunk APM and other observability platforms.

### Supported Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **SDK Mode** | Wraps `cisco-aidefense-sdk` methods | Explicit security checks via `inspect_prompt()` |
| **Gateway Mode** | Captures `X-Cisco-AI-Defense-Event-Id` from HTTP headers | LLM calls proxied through AI Defense Gateway |

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

We use **monkey-patching via `wrapt`** to wrap AI Defense SDK methods and HTTP clients:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│                                                              │
│   client.chat.completions.create(...)                       │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │        AIDefenseInstrumentor (this package)          │   │
│   │                                                      │   │
│   │   SDK Mode:                                          │   │
│   │   1. Create LLMInvocation with security context     │   │
│   │   2. handler.start_llm(invocation)  ← Start span    │   │
│   │   3. Call original inspect_prompt()                 │   │
│   │   4. Extract event_id from result                   │   │
│   │   5. handler.stop_llm(invocation)   ← End span      │   │
│   │                                                      │   │
│   │   Gateway Mode:                                      │   │
│   │   1. Wrap httpx/botocore HTTP client                │   │
│   │   2. Call original method → goes to Gateway         │   │
│   │   3. Extract X-Cisco-AI-Defense-Event-Id header     │   │
│   │   4. Add to current span (LangChain/OpenAI span)    │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Why `LLMInvocation`? (SDK Mode)

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
        metadata={"semconv": GEN_AI_SECURITY_EVENT_ID},  # ← Auto-emitted to span
    )
```

The `semantic_convention_attributes()` method in the `GenAI` base class automatically:
1. Iterates over dataclass fields
2. Finds fields with `metadata={"semconv": ...}`
3. Emits them as span attributes

## Installation

```bash
pip install splunk-otel-instrumentation-aidefense
```

## Gateway Mode (NEW in v0.2.0)

Gateway Mode automatically captures security event IDs when LLM calls are proxied through [AI Defense Gateway](https://securitydocs.cisco.com/docs/ai-def/user/105487.dita).

### Supported LLM Providers

| Provider | SDK | URL Pattern |
|----------|-----|-------------|
| **OpenAI** | `openai` | `api.openai.com` |
| **Azure OpenAI** | `openai` | `*.openai.azure.com` |
| **AWS Bedrock** | `boto3` | `bedrock-runtime.*.amazonaws.com` |
| **Google Vertex AI** | `google-cloud-aiplatform` | `*aiplatform.googleapis.com` |
| **Cohere** | `cohere` | `api.cohere.com` |
| **Mistral** | `mistralai` | `api.mistral.ai` |

### How It Works

1. Configure your LLM SDK to use AI Defense Gateway URL as the base URL
2. Gateway inspects requests/responses and adds `X-Cisco-AI-Defense-Event-Id` header
3. This instrumentation extracts the header and adds it to the current span

```python
from openai import OpenAI
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

# Instrument (LangChain first, then AI Defense)
LangchainInstrumentor().instrument()
AIDefenseInstrumentor().instrument()

# Configure OpenAI to use AI Defense Gateway
client = OpenAI(
    base_url="https://gateway.aidefense.security.cisco.com/{tenant}/connections/{conn}/v1",
    api_key="your-llm-api-key",
)

# LLM calls automatically get gen_ai.security.event_id in spans
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Gateway Mode Trace Structure

In Gateway Mode, the `gen_ai.security.event_id` is added to **existing LLM spans** (no separate spans):

```
POST /api/chat
└── ChatOpenAI                              ← LangChain span
    ├── gen_ai.request.model: gpt-4o-mini
    ├── gen_ai.response.id: chatcmpl-...
    └── gen_ai.security.event_id: e91a8f7a-...  ← Added by Gateway Mode
```

### Custom Gateway URLs

For custom AI Defense Gateway deployments:

```bash
export OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS="custom-gateway.internal,my-proxy.corp"
```

## SDK Mode

SDK Mode wraps the `cisco-aidefense-sdk` methods to create dedicated spans for security inspections.

### Usage

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

### SDK Mode Trace Structure

SDK Mode creates separate spans for each inspection:

```
POST /travel/plan
└── workflow LangGraph
    ├── step flight_specialist
    │   ├── chat cisco-ai-defense      ← AI Defense check (passed)
    │   └── invoke_agent flight_specialist
    ├── step hotel_specialist
    │   ├── chat cisco-ai-defense      ← AI Defense check (passed)
    │   └── invoke_agent hotel_specialist
    └── step activity_specialist
        └── chat cisco-ai-defense      ← AI Defense check (BLOCKED)
            └── gen_ai.security.event_id: "203d272b-d6b0-4c39-..."
```

### Instrumented Methods

#### ChatInspectionClient

| Method | Description |
|--------|-------------|
| `inspect_prompt` | Inspect user prompts for security violations |
| `inspect_response` | Inspect AI responses for security violations |
| `inspect_conversation` | Inspect full conversations |

#### HttpInspectionClient

| Method | Description |
|--------|-------------|
| `inspect_request` | Inspect HTTP requests |
| `inspect_response` | Inspect HTTP responses |
| `inspect_request_from_http_library` | Inspect requests from `requests` library |
| `inspect_response_from_http_library` | Inspect responses from `requests` library |

## Trace Integration

When used alongside other GenAI instrumentations (LangChain, CrewAI, etc.), AI Defense spans automatically integrate with the active trace:

### SDK Mode Trace Example

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

### Gateway Mode Trace Example

```
POST /travel/plan
└── workflow LangGraph
    ├── step flight_specialist
    │   └── ChatOpenAI                 ← LLM call through Gateway
    │       ├── gen_ai.request.model: gpt-4o-mini
    │       └── gen_ai.security.event_id: e91a8f7a-...  ← From Gateway
    ├── step hotel_specialist
    │   └── ChatOpenAI
    │       └── gen_ai.security.event_id: f82b9c6d-...
    └── step activity_specialist
        └── ChatOpenAI                 ← BLOCKED by AI Defense
            └── gen_ai.security.event_id: 203d272b-...
```

## Example: Multi-Agent Travel Planner with Security

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

## Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.security.event_id` | `string` | Unique event ID from AI Defense |
| `gen_ai.request.model` | `string` | `cisco-ai-defense` (SDK Mode only) |
| `gen_ai.system` | `string` | `aidefense` (SDK Mode only) |
| `server.address` | `string` | AI Defense API endpoint (SDK Mode only) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AI_DEFENSE_GATEWAY_URL` | AI Defense Gateway endpoint URL (e.g., `https://gateway.aidefense.security.cisco.com/{tenant}/connections/{conn}/v1`) |
| `OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS` | Custom AI Defense Gateway URL patterns for auto-detection (comma-separated) |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Set to `true` to capture full message content in spans |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint (e.g., `http://localhost:4317`) |

## Auto-Instrumentation

Using OpenTelemetry auto-instrumentation:

```bash
opentelemetry-instrument --traces_exporter otlp python your_app.py
```

## Examples

- **SDK Mode**: See `examples/multi_agent_travel_planner/` - Explicit security checks with `inspect_prompt()`
- **Gateway Mode**: See `examples/gateway/multi_agent_travel_planner/` - LLM calls through AI Defense Gateway

## Code Structure

The instrumentation follows the DRY (Don't Repeat Yourself) principle:

```
src/opentelemetry/instrumentation/aidefense/
├── __init__.py           # Package exports
├── instrumentation.py    # Main instrumentor and wrappers
├── util/
│   ├── __init__.py       # Util package exports
│   └── helper.py         # Reusable helpers (DRY)
└── version.py            # Package version
```

**`util/helper.py`** provides common utilities to reduce code repetition:
- `create_ai_defense_invocation()` - Creates standardized LLMInvocation for AI Defense
- `create_input_message()` - Creates InputMessage with automatic content truncation
- `execute_with_telemetry()` - Handles common wrapper pattern (start, execute, stop/fail)
- `get_server_address()` - Extracts server address from client instance

## Requirements

- Python >= 3.9
- opentelemetry-api >= 1.38.0
- splunk-otel-util-genai >= 0.1.5
- For SDK Mode: cisco-aidefense-sdk >= 2.0.0
- For Gateway Mode: httpx (for OpenAI, Cohere, Mistral) or boto3 (for AWS Bedrock)

## References

- [AI Defense Gateway Documentation](https://securitydocs.cisco.com/docs/ai-def/user/105487.dita)
- [Cisco AI Defense Python SDK](https://github.com/cisco-ai-defense/ai-defense-python-sdk)
- [AI Defense API Documentation](https://developer.cisco.com/docs/ai-defense/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Splunk OTel Python Contrib](https://github.com/signalfx/splunk-otel-python-contrib)

## License

Apache License 2.0
