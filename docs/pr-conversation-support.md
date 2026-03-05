# PR: Conversation Context & Association Properties

**Branch**: `feature/conversation-support-main` -> `main`

## What

Adds `gen_ai.conversation.id` and custom **association properties** that
auto-propagate to all GenAI spans.

## API

```python
from opentelemetry.util.genai import genai_context

with genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice", "customer.id": "acme"},
):
    result = chain.invoke({"input": "Hello"})
    # All spans get:
    #   gen_ai.conversation.id = "conv-123"
    #   gen_ai.association.properties.user.id = "alice"
    #   gen_ai.association.properties.customer.id = "acme"

# Imperative
set_genai_context(conversation_id="conv-123", properties={"user.id": "alice"})
```

Priority: **Invocation value > ContextVars > Env var** (highest to lowest).

## Key Changes

### `opentelemetry-util-genai`

- **`handler.py`** — `GenAIContext(conversation_id, properties)` dataclass;
  `set_genai_context()`, `genai_context()` context manager;
  `_apply_genai_context()` with property merge (context first, invocation overrides)
- **`types.py`** — `association_properties: Dict[str, Any]` on `GenAI` base;
  `semantic_convention_attributes()` emits `gen_ai.association.properties.<key>`
- **`attributes.py`** — `GEN_AI_ASSOCIATION_PROPERTIES_PREFIX` constant
- **`environment_variables.py`** — `CONVERSATION_ID` and `CONTEXT_INCLUDE_IN_METRICS` env vars
- **`emitters/utils.py`** — `get_context_metric_attributes()` works with properties dict
  (`all` or comma-separated property keys)

### `opentelemetry-instrumentation-fastmcp`

- **`transport_instrumentor.py`** — Standard OTel `propagate.inject/extract` + Baggage

## How to Review

1. **Start with** `handler.py` — `GenAIContext`, `genai_context()`, `_apply_genai_context()`
2. **Type fields** — `types.py` — `association_properties` on `GenAI` base
3. **Tests** — `test_genai_context.py`, `test_transport_propagation.py`
4. **API docs** — `genai-context.md`

## Test Coverage

```bash
pytest ./util/opentelemetry-util-genai/tests/ -v
pytest ./instrumentation-genai/opentelemetry-instrumentation-fastmcp/tests/ -v
```
