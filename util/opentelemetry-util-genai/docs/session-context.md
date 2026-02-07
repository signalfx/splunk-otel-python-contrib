# Session Context Support

Session context enables tracking multi-turn conversations, user sessions, and tenant identification across GenAI operations. All spans created within a session context automatically include `session.id`, `user.id`, and `customer.id` attributes.

## Quick Start

### Using the Context Manager (Recommended)

```python
from opentelemetry.util.genai.handler import session_context

# All GenAI operations within this block will include session attributes
with session_context(session_id="conv-123", user_id="user-456"):
    result = chain.invoke({"input": "Hello"})
    # Spans will have: session.id="conv-123", user.id="user-456"
```

### Manual Context Management

```python
from opentelemetry.util.genai.handler import (
    set_session_context,
    get_session_context,
    clear_session_context,
)

# Set session context
set_session_context(
    session_id="conv-123",
    user_id="user-456",
    customer_id="tenant-789",
)

# ... GenAI operations ...

# Get current context
ctx = get_session_context()
print(f"Session: {ctx.session_id}")

# Clear when done
clear_session_context()
```

### Environment Variables

For simple single-session deployments, you can set static values via environment variables:

```bash
export OTEL_INSTRUMENTATION_GENAI_SESSION_ID="static-session-id"
export OTEL_INSTRUMENTATION_GENAI_USER_ID="static-user-id"
export OTEL_INSTRUMENTATION_GENAI_CUSTOMER_ID="static-customer-id"
```

## Priority Order

Session attributes are applied with the following priority (highest to lowest):

1. **Explicit value on invocation** - Set directly on the GenAI type
2. **Contextvars** - Set via `set_session_context()` or `session_context()`
3. **Environment variables** - Static fallback values

```python
# Explicit value takes priority
inv = LLMInvocation(
    request_model="gpt-4",
    session_id="explicit-session",  # This wins
)

# With context set
with session_context(session_id="context-session"):
    inv2 = LLMInvocation(request_model="gpt-4")
    # inv2.session_id will be "context-session"
```

## Span Attributes

The following attributes are emitted on GenAI spans:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `session.id` | Session/conversation identifier | `"conv-123"` |
| `user.id` | User identifier | `"user-456"` |
| `customer.id` | Customer/tenant identifier | `"tenant-789"` |

## Examples

### LangGraph Workflow

```python
from uuid import uuid4
from opentelemetry.util.genai.handler import session_context

def run_workflow(user_input: str, user_id: str):
    session_id = str(uuid4())
    
    with session_context(session_id=session_id, user_id=user_id):
        # All agent, LLM, and tool spans will include session attributes
        result = workflow.invoke({"input": user_input})
    
    return result, session_id
```

### Multi-Tenant Application

```python
from opentelemetry.util.genai.handler import session_context

@app.route("/chat", methods=["POST"])
def chat():
    tenant_id = request.headers.get("X-Tenant-ID")
    user_id = request.headers.get("X-User-ID")
    session_id = request.json.get("session_id")
    
    with session_context(
        session_id=session_id,
        user_id=user_id,
        customer_id=tenant_id,
    ):
        response = chain.invoke({"input": request.json["message"]})
    
    return jsonify({"response": response})
```

### Async Support

The session context is async-safe using Python's `contextvars`:

```python
import asyncio
from opentelemetry.util.genai.handler import session_context

async def process_request(session_id: str):
    with session_context(session_id=session_id):
        # Works correctly in async context
        result = await async_chain.ainvoke({"input": "Hello"})
    return result

# Multiple concurrent sessions are properly isolated
await asyncio.gather(
    process_request("session-1"),
    process_request("session-2"),
)
```

## API Reference

### `session_context(session_id, user_id, customer_id)`

Context manager that sets session context and automatically clears it on exit.

**Parameters:**
- `session_id` (str, optional): Session/conversation identifier
- `user_id` (str, optional): User identifier
- `customer_id` (str, optional): Customer/tenant identifier

**Yields:** `SessionContext` object

### `set_session_context(session_id, user_id, customer_id)`

Sets the session context for the current execution context.

### `get_session_context()`

Returns the current `SessionContext` object.

### `clear_session_context()`

Clears the current session context.

### `SessionContext`

Dataclass holding session state:
- `session_id: Optional[str]`
- `user_id: Optional[str]`
- `customer_id: Optional[str]`
- `is_empty() -> bool`: Returns True if no values are set

## Migration from Manual Session Tracking

If you were previously tracking sessions manually:

**Before:**
```python
session_id = str(uuid4())
state["session_id"] = session_id
# Session not propagated to telemetry
```

**After:**
```python
from opentelemetry.util.genai.handler import session_context

session_id = str(uuid4())
with session_context(session_id=session_id):
    # Session automatically propagated to all spans
    result = workflow.invoke(state)
```
