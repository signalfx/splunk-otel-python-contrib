# GenAI Context — API Reference

GenAI context enables tracking multi-turn conversations and arbitrary
association properties across GenAI operations. All spans created within a
GenAI context automatically include `gen_ai.conversation.id` and
`gen_ai.association.properties.<key>` attributes.

## Quick Start

### Context Manager (Recommended)

```python
from opentelemetry.util.genai import genai_context

with genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice", "customer.id": "acme"},
):
    result = chain.invoke({"input": "Hello"})
    # Spans will have:
    #   gen_ai.conversation.id = "conv-123"
    #   gen_ai.association.properties.user.id = "alice"
    #   gen_ai.association.properties.customer.id = "acme"
```

### Imperative API

```python
from opentelemetry.util.genai import (
    set_genai_context,
    get_genai_context,
    clear_genai_context,
)

set_genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice"},
)

ctx = get_genai_context()
print(f"Conversation: {ctx.conversation_id}, Props: {ctx.properties}")

clear_genai_context()
```

### Environment Variable

```bash
export OTEL_INSTRUMENTATION_GENAI_CONVERSATION_ID="static-conversation-id"
```

## Priority Order

Context attributes are applied with the following priority (highest to lowest):

1. **Explicit value on invocation** — set directly on the GenAI type
2. **ContextVars** — set via `set_genai_context()` or `genai_context()`
3. **Environment variable** — static fallback (conversation_id only)

Association properties from context and invocation are **merged**: context
properties are applied first, then invocation-level properties override for
the same key.

## Span Attributes

| Attribute | Source | Example |
|-----------|--------|---------|
| `gen_ai.conversation.id` | `conversation_id` param | `"conv-123"` |
| `gen_ai.association.properties.<key>` | `properties` dict | `"alice"` |

## Including Context Attributes in Metrics

By default, **no** context attributes are added to metric dimensions (they are
high-cardinality). To opt in, set `OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS`:

```bash
# Include all context attributes (conversation_id + all association properties)
export OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS=all

# Include only specific attributes (comma-separated keys)
export OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS=user.id,customer.id

# Include only conversation_id in metrics
export OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS=gen_ai.conversation.id
```

Either the short property key (`user.id`) or the full prefixed attribute
(`gen_ai.association.properties.user.id`) can be used when specifying specific keys.

## API Reference

### `genai_context(conversation_id=None, properties=None)`

Context manager that sets GenAI context and auto-restores on exit.

**Parameters:**
- `conversation_id` (`str`, optional): Conversation identifier
    (emitted as `gen_ai.conversation.id`)
- `properties` (`dict[str, Any]`, optional): Association properties
    (emitted as `gen_ai.association.properties.<key>`)

**Yields:** `GenAIContext` object

### `set_genai_context(conversation_id=None, properties=None)`

Sets GenAI context for the current execution context.

### `get_genai_context() -> GenAIContext`

Returns the current `GenAIContext` object.

### `clear_genai_context()`

Clears the current GenAI context.

### `GenAIContext`

Dataclass holding context state:
- `conversation_id: Optional[str]` — conversation identifier
- `properties: dict[str, Any]` — association properties (default: `{}`)
- `is_empty() -> bool` — returns True if no values are set

## Examples

### LangGraph Workflow

```python
from uuid import uuid4
from opentelemetry.util.genai import genai_context

def run_workflow(user_input: str, user_id: str):
    conversation_id = str(uuid4())

    with genai_context(
        conversation_id=conversation_id,
        properties={"user.id": user_id},
    ):
        result = workflow.invoke({"input": user_input})

    return result, conversation_id
```

### Multi-Tenant Application

```python
from opentelemetry.util.genai import genai_context

@app.route("/chat", methods=["POST"])
def chat():
    tenant_id = request.headers.get("X-Tenant-ID")
    user_id = request.headers.get("X-User-ID")
    conversation_id = request.json.get("conversation_id")

    with genai_context(
        conversation_id=conversation_id,
        properties={
            "user.id": user_id,
            "customer.id": tenant_id,
        },
    ):
        response = chain.invoke({"input": request.json["message"]})

    return jsonify({"response": response})
```

### Async Support

The GenAI context is async-safe using Python `contextvars`:

```python
import asyncio
from opentelemetry.util.genai import genai_context

async def process_request(conversation_id: str):
    with genai_context(conversation_id=conversation_id):
        result = await async_chain.ainvoke({"input": "Hello"})
    return result

# Multiple concurrent conversations are properly isolated
await asyncio.gather(
    process_request("conv-1"),
    process_request("conv-2"),
)
```
