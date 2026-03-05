# GenAI Conversation Context & Association Properties

## Overview

This feature adds end-to-end conversation context tracking and association
properties to the SDOT GenAI instrumentation stack.

| Capability | Description |
|------------|-------------|
| `gen_ai.conversation.id` | First-class conversation identifier on all GenAI spans |
| Association properties | Arbitrary `key=value` pairs as `gen_ai.association.properties.<key>` |
| 3-level priority | Invocation > ContextVars > Env var |
| Cross-service propagation | Standard OTel `propagate.inject/extract` (W3C TraceContext + Baggage) |
| Opt-in metrics | Context attributes as metric dimensions |

## Usage Patterns

### 1. Explicit Context (Context Manager)

```python
from opentelemetry.util.genai import genai_context

with genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice", "customer.id": "acme"},
):
    result = chain.invoke({"input": "Hello"})
```

### 2. Imperative API

```python
from opentelemetry.util.genai import set_genai_context

set_genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice"},
)
```

### 3. Environment Variable (Static Fallback)

```bash
export OTEL_INSTRUMENTATION_GENAI_CONVERSATION_ID="static-conv-id"
```

### 4. Auto-propagation from LangChain Config Metadata (Roadmap)

```python
chain.invoke(
    {"input": "Hello"},
    config={"metadata": {"tenant": "acme"}},
)
# -> span: gen_ai.association.properties.tenant="acme"
```

## Files Changed

### `opentelemetry-util-genai`

| File | Change |
|------|--------|
| `handler.py` | `GenAIContext(conversation_id, properties)`, context API, `_apply_genai_context()` |
| `types.py` | `association_properties` field, `semantic_convention_attributes()` with prefix |
| `attributes.py` | `GEN_AI_ASSOCIATION_PROPERTIES_PREFIX` constant |
| `environment_variables.py` | `CONVERSATION_ID`, `CONTEXT_INCLUDE_IN_METRICS` |
| `emitters/utils.py` | `get_context_metric_attributes()` with properties dict |
| `tests/test_genai_context.py` | Full test coverage for context API, priority, semconv |

### `opentelemetry-instrumentation-fastmcp`

| File | Change |
|------|--------|
| `transport_instrumentor.py` | Standard `propagate.inject/extract` |
| `tests/test_transport_propagation.py` | Round-trip propagation tests |

## References

- [Design doc](feat-conversation-support.md)
- [PR description](pr-conversation-support.md)
- [API reference](../util/opentelemetry-util-genai/docs/genai-context.md)
