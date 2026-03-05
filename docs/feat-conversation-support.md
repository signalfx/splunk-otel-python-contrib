# Conversation Context Support — Design & Roadmap

## What This Feature Does

- Adds `gen_ai.conversation.id` context attribute that automatically propagates
  to all GenAI spans within scope. This enables multi-turn conversation correlation.
- Adds custom association properties via a `properties` dict, which allows
  arbitrary key-value pairs to be propagated as
  `gen_ai.association.properties.<key>` span attributes.

### Context and Attribute Semantics

Canonical conversation field is `conversation_id` mapped to `gen_ai.conversation.id`.

Association properties are represented as:
`gen_ai.association.properties.<key> = <value>`

---

## Current State (Implemented)

### Public API

```python
from opentelemetry.util.genai import genai_context, set_genai_context

# Context manager (recommended)
with genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice", "customer.id": "acme"},
):
    result = chain.invoke({"input": "Hello"})

# Imperative API
set_genai_context(
    conversation_id="conv-123",
    properties={"user.id": "alice"},
)
```

### What works today

| Capability | Status |
|------------|--------|
| `GenAIContext` dataclass (`conversation_id`, `properties`) | Done |
| `genai_context()` context manager with auto-restore | Done |
| `set_genai_context()` / `get_genai_context()` / `clear_genai_context()` | Done |
| Association properties via `properties` dict | Done |
| Attribute prefix: `gen_ai.association.properties.<key>` | Done |
| 3-level priority: Invocation > ContextVars > Env var | Done |
| Auto-apply to all GenAI invocation types (LLM, Agent, Workflow, Tool, etc.) | Done |
| Opt-in metrics dimensions (`CONTEXT_INCLUDE_IN_METRICS`) | Done |
| Cross-service propagation via standard OTel Baggage (`propagate.inject/extract`) | Done |
| Dynamic `network.transport` detection for MCP (stdio/SSE/HTTP/inproc) | Done |

### Architecture

Uses Python `contextvars.ContextVar` for thread-safe and async-safe context
propagation. The `GenAIContext` dataclass holds `conversation_id` and a
`properties` dict.

**Priority order** (highest to lowest):

1. **Explicit value on invocation** — set directly on the GenAI type object
2. **ContextVars** — set via `set_genai_context()` or `genai_context()`
3. **Environment variable** — `OTEL_INSTRUMENTATION_GENAI_CONVERSATION_ID` (conversation_id only)

**Property merge**: Context-level properties are applied first, then
invocation-level properties override for the same key.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_INSTRUMENTATION_GENAI_CONVERSATION_ID` | Static conversation ID fallback | — |
| `OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS` | Context attrs as metric dimensions (`all` or comma-separated keys) | empty |

---

## Remaining Work

### 1. Update Example Files (Critical)

Several example files still import deleted APIs (`session_context`,
`set_session_context`, `get_session_context`, `SessionContext`). These will
fail at runtime.

| File | Issue |
|------|-------|
| `util/.../examples/invocation_example.py` | Imports `session_context`, uses `session_id=` |
| `fastmcp/.../examples/e2e/client.py` | Imports `set_session_context`, uses `session_id=` |
| `fastmcp/.../examples/e2e/server.py` | Imports `get_session_context`, accesses `ctx.session_id` |
| `fastmcp/.../examples/e2e/server_instrumented.py` | Same as server.py |
| `langchain/.../examples/sre_incident_copilot/main.py` | Imports `session_context`, uses `session_id=` |
| `fastmcp/.../examples/e2e/README.md` | References `SESSION_PROPAGATION` env var |
| `util/opentelemetry-util-genai/CHANGELOG.md` | Entire v0.1.10 section uses old API names |

### 2. LangChain Config Metadata Autopropagation

Bridge LangChain's `config.metadata` dict into association properties on spans
automatically in the LangChain instrumentation:

```python
# User code — LangChain standard pattern
chain.invoke(
    {"input": "Hello"},
    config={"metadata": {"tenant": "acme", "request_id": "req-123"}},
)
# -> span attributes:
#   gen_ai.association.properties.tenant = "acme"
#   gen_ai.association.properties.request_id = "req-123"
```

**Implementation approach**: In the LangChain callback handler, read
`config.metadata` at chain/agent start and either:
- Set as association properties on the current `GenAIContext`, or
- Directly add as span attributes via the invocation object

This makes existing LangChain apps observable without code changes.

### 3. Config to Disable Same-Process Propagation

Add an env var to disable automatic context inheritance for child spans:

```
OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION=true   (default: true)
```

When `false`, `_apply_genai_context()` becomes a no-op — users must set
`conversation_id` explicitly on each invocation object.

**Use case**: Large monoliths where different GenAI operations shouldn't
inherit conversation context from a shared scope.

### 4. `session_id` Cleanup in Example Apps (Low priority)

~130 occurrences of `session_id` as a LangGraph state dict key across example
apps (`multi_agent_travel_planner/`, `sre_incident_copilot/`, `aidefense/`).
These are application-level state, not SDOT API calls, but should align with
`conversation_id` naming for consistency.

---

## Design Decisions

### Why `gen_ai.conversation.id` (not `session.id`)

`gen_ai.conversation.id` is defined in the
[OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
and is specific to AI conversation tracking. `session.id` is a more general
concept (proposed but not yet stable in OTel) that could mean browser session,
HTTP session, etc. Using the specific attribute avoids ambiguity.

### Why no `session_id` field on GenAI types

The `GenAI` base dataclass has `conversation_id` mapped to
`GenAIAttributes.GEN_AI_CONVERSATION_ID`. There is no `session_id` alias — this
keeps the API surface clean and aligned with semconv.

### Why `gen_ai.association.properties.<key>` prefix

Association properties use the `gen_ai.association.properties.` prefix to provide
a clear namespace for arbitrary user-supplied attributes. This avoids collisions
with standard OpenTelemetry semantic convention attributes.

### Why standard OTel propagation (not custom `_meta`)

The original implementation used MCP-specific `_meta`-based propagation with
`restore_session_from_context()` and server-side baggage allowlists. This was
replaced with standard `propagate.inject()`/`propagate.extract()` because:
- Works across any transport, not just MCP
- No custom code to maintain
- W3C Baggage is the standard mechanism for cross-service context

### Priority order: Invocation > ContextVars > Env var

Three levels, not four. The original design had OTel Baggage as a fourth level
between ContextVars and env vars. This was removed because baggage values
are already restored into ContextVars by standard OTel `propagate.extract()`.

---

## References

- [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OTel semconv #2883](https://github.com/open-telemetry/semantic-conventions/issues/2883) — `session.id` proposal
- [API reference](../util/opentelemetry-util-genai/docs/genai-context.md)
- [PR summary](pr-conversation-support.md)
- [Feature summary](feat-genai-conversation-id.md)
