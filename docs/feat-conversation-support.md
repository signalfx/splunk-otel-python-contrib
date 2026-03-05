# Conversation Context & Association Properties — Design

## What This Feature Does

- Adds `gen_ai.conversation.id` context attribute for multi-turn conversation correlation
- Adds custom **association properties** (`gen_ai.association.properties.<key>`) for arbitrary context propagation
- Modeled after [Traceloop association properties](https://github.com/traceloop/openllmetry)

## Public API

```python
from opentelemetry.util.genai import genai_context, set_genai_context

# Context manager (recommended)
with genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice", "customer.id": "acme"},
):
    result = chain.invoke({"input": "Hello"})
    # All spans get:
    #   gen_ai.conversation.id = "conv-123"
    #   gen_ai.association.properties.user.id = "alice"
    #   gen_ai.association.properties.customer.id = "acme"

# Imperative API
set_genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice"},
)
```

## Architecture

### Context Storage

Uses Python `contextvars.ContextVar` for thread-safe and async-safe context propagation.
The `GenAIContext` dataclass holds `conversation_id` and a `properties` dict.

### Priority Order

Context attributes are resolved with three priority levels (highest to lowest):

1. **Explicit value on invocation** — set directly on the GenAI type object
2. **ContextVars** — set via `set_genai_context()` or `genai_context()`
3. **Environment variable** — `OTEL_INSTRUMENTATION_GENAI_CONVERSATION_ID` (conversation_id only)

### Property Merge

Association properties from context and invocation are merged:
- Context-level properties are applied first
- Invocation-level properties override same-key context properties

## Comparison with Traceloop

| Aspect | Traceloop | SDOT |
|--------|-----------|------|
| Storage | OTel Context (`attach`/`set_value`) | Python `contextvars.ContextVar` |
| Application | `SpanProcessor.on_start` stamps every span | `_apply_genai_context()` called in `TelemetryHandler.start_*()` |
| Attribute prefix | `traceloop.association.properties.<key>` | `gen_ai.association.properties.<key>` |
| LangChain integration | Callback handler merges `config.metadata` | Roadmap |
| Cross-service | `traceparent` via MCP `_meta` | Standard OTel `propagate.inject/extract` |

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_INSTRUMENTATION_GENAI_CONVERSATION_ID` | Static conversation ID | — |
| `OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS` | Context attrs as metric dimensions (`all` or comma-separated keys) | empty |

## Capabilities

| Capability | Status |
|------------|--------|
| `GenAIContext` dataclass (`conversation_id`, `properties` dict) | Done |
| `genai_context()` context manager with auto-restore | Done |
| `set_genai_context()` / `get_genai_context()` / `clear_genai_context()` | Done |
| 3-level priority: Invocation > ContextVars > Env var | Done |
| Auto-apply to all GenAI invocation types | Done |
| Association properties as `gen_ai.association.properties.<key>` | Done |
| Opt-in metrics dimensions | Done |
| Cross-service propagation via standard OTel Baggage | Done |
| Dynamic `network.transport` detection for MCP | Done |

## Roadmap

### LangChain Config Metadata Auto-propagation

Traceloop automatically bridges LangChain `config.metadata` into association
properties. SDOT should do the same in the LangChain instrumentation:

```python
chain.invoke(
    {"input": "Hello"},
    config={"metadata": {"tenant": "acme", "request_id": "req-123"}},
)
# -> span attributes: gen_ai.association.properties.tenant="acme"
```

### Config to Disable Same-Process Propagation

Add an env var to disable automatic context inheritance:

```
OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION=true   (default: true)
```

## References

- [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Traceloop association properties](https://github.com/traceloop/openllmetry)
- [API reference](../util/opentelemetry-util-genai/docs/genai-context.md)
