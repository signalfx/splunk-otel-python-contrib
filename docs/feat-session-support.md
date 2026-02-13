# Session Support Feature Plan

## Overview

This document outlines options for adding `session.id` support to the Splunk OpenTelemetry Python Contrib GenAI instrumentation. Session tracking enables grouping related traces/spans together for multi-turn conversations, user interactions, and workflow executions.

## Current State Analysis

### Existing Session Usage in the Codebase

1. **SRE Incident Copilot App** ([main.py](../instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot/main.py)):
   - Already generates a `session_id` using `uuid4()`
   - Stores it in `IncidentState` and uses it for LangGraph's `thread_id`
   - Saves it in artifacts metadata (`run_meta.json`)
   - **Not propagated** to OpenTelemetry spans/traces

2. **MCP Tool Calls** ([types.py](../util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py)):
   - `MCPToolCall` dataclass has `mcp_session_id` field mapped to `mcp.session.id` semantic convention
   - This is MCP-protocol-specific, not general GenAI session tracking

3. **Examples** ([retrievals_example.py](../util/opentelemetry-util-genai/examples/retrievals_example.py), [langgraph-multi-agent-rag](../util/opentelemetry-util-genai/examples/langgraph-multi-agent-rag/main.py)):
   - Use custom `session_id` in attributes but not standardized

### Traceloop's Approach (OpenLLMetry)

Traceloop implements session tracking via **Association Properties** - a context-propagation mechanism using Python's `contextvars`:

#### Key Components

1. **`AssociationProperty` Enum** ([associations.py](https://github.com/traceloop/openllmetry/blob/main/packages/traceloop-sdk/traceloop/sdk/associations/associations.py)):
   ```python
   class AssociationProperty(str, Enum):
       CUSTOMER_ID = "customer_id"
       USER_ID = "user_id"
       SESSION_ID = "session_id"
   ```

2. **`set_association_properties()` Function** ([tracing.py](https://github.com/traceloop/openllmetry/blob/main/packages/traceloop-sdk/traceloop/sdk/tracing/tracing.py)):
   ```python
   def set_association_properties(properties: dict) -> None:
       attach(set_value("association_properties", properties))
       span = trace.get_current_span()
       if get_value("workflow_name") is not None or get_value("entity_name") is not None:
           _set_association_properties_attributes(span, properties)
   ```

3. **Span Processor Hook** - Propagates association properties to all child spans via `on_start`:
   ```python
   def default_span_processor_on_start(span: Span, parent_context: Context | None = None):
       association_properties = get_value("association_properties")
       if association_properties is not None and isinstance(association_properties, dict):
           _set_association_properties_attributes(span, association_properties)
   ```

4. **Attribute Format**: `traceloop.association.properties.{key}` (e.g., `traceloop.association.properties.session_id`)

5. **Usage Pattern**:
   ```python
   # Via SDK
   Traceloop.set_association_properties({"session_id": "conv-123", "user_id": "user-456"})
   
   # Via associations API
   client.associations.set([
       (AssociationProperty.SESSION_ID, "conv-123"),
       (AssociationProperty.USER_ID, "user-456"),
   ])
   
   # Via LangChain metadata
   chain.invoke({"input": "..."}, {"metadata": {"session_id": "conv-123"}})
   ```

---

## OpenTelemetry Semantic Conventions

### Relevant Attributes

| Attribute | Status | Description |
|-----------|--------|-------------|
| `gen_ai.conversation.id` | Stable | Unique identifier for the conversation/session |
| `session.id` | Proposed | General session identifier (not GenAI-specific) |
| `user.id` | Stable | User identifier |

The `gen_ai.conversation.id` is already defined in our [types.py](../util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py):
```python
conversation_id: Optional[str] = field(
    default=None,
    metadata={"semconv": GenAIAttributes.GEN_AI_CONVERSATION_ID},
)
```

---

## Implementation Options

### Option 1: Extend Existing GenAI Types (Recommended)

**Approach**: Add session/user context fields to the `GenAI` base type and propagate via existing TelemetryHandler.

#### Changes to `types.py`:
```python
@dataclass(kw_only=True)
class GenAI:
    """Base type for all GenAI telemetry entities."""
    # ... existing fields ...
    
    # Session/User Context (association properties pattern)
    session_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "session.id"},  # or gen_ai.session.id
    )
    user_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "user.id"},
    )
    customer_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "customer.id"},
    )
```

#### Changes to `handler.py`:
```python
class TelemetryHandler:
    # Context variable for session/user propagation
    _session_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
        "session_context", default={}
    )
    
    @classmethod
    def set_session_context(cls, session_id: str | None = None, 
                           user_id: str | None = None,
                           customer_id: str | None = None) -> None:
        """Set session context that propagates to all nested GenAI operations."""
        context = {}
        if session_id:
            context["session_id"] = session_id
        if user_id:
            context["user_id"] = user_id
        if customer_id:
            context["customer_id"] = customer_id
        cls._session_context.set(context)
    
    def start_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        # Apply session context to invocation if not already set
        context = self._session_context.get()
        if context:
            if not invocation.session_id and "session_id" in context:
                invocation.session_id = context["session_id"]
            if not invocation.user_id and "user_id" in context:
                invocation.user_id = context["user_id"]
            # ... similar for other fields
        # ... rest of start_llm
```

#### Pros:
- Minimal new code, leverages existing infrastructure
- Consistent with current dataclass patterns
- Automatic emission via `semantic_convention_attributes()`
- Works with all emitters (span, metrics, events)

#### Cons:
- Requires explicit `set_session_context()` call
- Context doesn't auto-propagate to uninstrumented code

---

### Option 2: Span Processor Approach (Traceloop-Style)

**Approach**: Use a custom SpanProcessor to inject session attributes into all spans.

#### New module `session.py`:
```python
from contextvars import ContextVar
from opentelemetry.context import attach, set_value, get_value
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.trace import Span

_ASSOCIATION_PROPERTIES_KEY = "genai_association_properties"

class AssociationProperty(str, Enum):
    SESSION_ID = "session_id"
    USER_ID = "user_id"
    CUSTOMER_ID = "customer_id"

def set_association_properties(properties: dict[str, str]) -> None:
    """Set association properties for the current context."""
    attach(set_value(_ASSOCIATION_PROPERTIES_KEY, properties))

def get_association_properties() -> dict[str, str]:
    """Get current association properties."""
    return get_value(_ASSOCIATION_PROPERTIES_KEY) or {}

class SessionSpanProcessor(SpanProcessor):
    """SpanProcessor that propagates session/association properties to all spans."""
    
    def on_start(self, span: Span, parent_context=None) -> None:
        props = get_association_properties()
        if props:
            for key, value in props.items():
                span.set_attribute(f"gen_ai.association.{key}", value)
    
    def on_end(self, span: ReadableSpan) -> None:
        pass
    
    def shutdown(self) -> None:
        pass
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
```

#### Usage:
```python
from opentelemetry.util.genai.session import set_association_properties

# In application code
set_association_properties({
    "session_id": "conv-123",
    "user_id": "user-456",
})

# All subsequent spans will have these attributes
```

#### Pros:
- Propagates to ALL spans (not just GenAI)
- Similar pattern to Traceloop (easier migration)
- Works without modifying GenAI types

#### Cons:
- Requires user to configure SpanProcessor
- More complex setup
- Session context stored in OTel context, not GenAI types

---

### Option 3: Hybrid Approach (Recommended for Full Compatibility)

**Approach**: Combine both options - extend GenAI types AND provide context propagation helpers.

#### Phase 1: Core Type Extensions
- Add `session_id`, `user_id`, `customer_id` to `GenAI` base class
- Modify emitters to emit these as span attributes

#### Phase 2: Context Propagation Helpers
- Add `set_session_context()` to `TelemetryHandler`
- Auto-populate session fields from context in `start_*` methods

#### Phase 3: Framework Integration
- LangChain: Extract from `config.metadata` or `RunnableConfig`
- CrewAI: Extract from task/crew context
- OpenAI Agents: Extract from thread_id

---

## Recommended Implementation Plan

### Phase 1: Core Session Support (Week 1)

1. **Update `types.py`**:
   - Add `session_id` to `GenAI` base class with semconv metadata
   - Add `user_id` and `customer_id` as optional association properties

2. **Update `handler.py`**:
   - Add `_session_context` ContextVar
   - Add `set_session_context()` and `get_session_context()` class methods
   - Modify `start_*` methods to apply context

3. **Update span emitter**:
   - Ensure session attributes are emitted on spans

4. **Add environment variable**:
   - `OTEL_INSTRUMENTATION_GENAI_SESSION_ID` for static session ID

### Phase 2: Instrumentation Integration (Week 2)

1. **LangChain Instrumentation**:
   - Extract session from `RunnableConfig.metadata.session_id`
   - Populate `session_id` on all GenAI types

2. **Update SRE Incident Copilot Example**:
   - Use new `set_session_context()` API
   - Remove manual session tracking

### Phase 3: Documentation & Testing (Week 3)

1. **Add tests** for session propagation
2. **Update README** with session usage examples
3. **Add migration guide** from manual session tracking

---

## API Design

### Proposed Public API

```python
# Option A: Handler-level (recommended)
from opentelemetry.util.genai import get_telemetry_handler

handler = get_telemetry_handler()

# Set session context for all nested operations
with handler.session_context(session_id="conv-123", user_id="user-456"):
    # All GenAI operations here will have session attributes
    result = chain.invoke(...)

# Or without context manager
handler.set_session_context(session_id="conv-123")
# ... operations ...
handler.clear_session_context()
```

```python
# Option B: Direct on invocation
invocation = LLMInvocation(
    request_model="gpt-4",
    session_id="conv-123",  # NEW
    user_id="user-456",     # NEW
    ...
)
```

```python
# Option C: Via instrumentation (auto-extracted)
# LangChain: extracted from config.metadata
chain.invoke(
    {"input": "..."},
    config={"metadata": {"session_id": "conv-123"}}
)
```

---

## Semantic Convention Alignment

### Proposed Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `session.id` | Session/conversation identifier | `"conv-123"` |
| `user.id` | User identifier | `"user-456"` |
| `gen_ai.conversation.id` | Alias for session.id (for backwards compat) | `"conv-123"` |
| `gen_ai.association.session_id` | Traceloop-compatible | `"conv-123"` |

### Backwards Compatibility

To support Traceloop migration, emit both:
- `session.id` (standard)
- `gen_ai.association.session_id` (Traceloop-style)

This can be controlled via:
```
OTEL_INSTRUMENTATION_GENAI_EMIT_TRACELOOP_ASSOCIATIONS=true|false
```

---

## Files to Modify

### Core Package (`util/opentelemetry-util-genai`)

| File | Changes |
|------|---------|
| `types.py` | Add `session_id`, `user_id`, `customer_id` to `GenAI` |
| `handler.py` | Add session context management |
| `emitters/span.py` | Ensure session attributes emitted |
| `emitters/metrics.py` | Add session as dimension (optional) |
| `environment_variables.py` | Add `OTEL_INSTRUMENTATION_GENAI_SESSION_ID` |
| `__init__.py` | Export session APIs |

### LangChain Instrumentation

| File | Changes |
|------|---------|
| `callback_handler.py` | Extract session from metadata |

### Examples

| File | Changes |
|------|---------|
| `sre_incident_copilot/main.py` | Use new session API |

---

## Testing Strategy

1. **Unit Tests**:
   - Session context propagation
   - Session attribute emission
   - Context cleanup

2. **Integration Tests**:
   - LangChain with session metadata
   - Multi-turn conversation tracking
   - Nested agent/workflow sessions

3. **E2E Tests**:
   - Verify spans have session attributes in backend

---

## Open Questions

1. **Attribute naming**: Should we use `session.id` (generic) or `gen_ai.session.id` (GenAI-specific)?
   - Recommendation: Use `session.id` for broader applicability

2. **Relationship to `gen_ai.conversation.id`**: Should these be the same or different?
   - Recommendation: Allow both, default `conversation_id` to `session_id` if not set

3. **Metrics dimensions**: Should `session_id` be a metric attribute?
   - Recommendation: No (high cardinality), unless explicitly enabled

4. **Thread-safety**: How to handle session context in async/threaded code?
   - Recommendation: Use `contextvars` (already thread/async safe)

---

## References

1. [Traceloop OpenLLMetry - Association Properties](https://github.com/traceloop/openllmetry/blob/main/packages/traceloop-sdk/traceloop/sdk/associations/associations.py)
2. [OpenTelemetry GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)
3. [Current types.py](../util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py)
4. [Current handler.py](../util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py)
