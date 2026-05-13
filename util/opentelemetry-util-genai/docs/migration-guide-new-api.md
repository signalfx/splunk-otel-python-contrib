# Migration Guide: New-Style Invocation API (HYBIM-606)

This document is for **instrumentation package developers** who use
`opentelemetry-util-genai` as their telemetry layer. It describes the
new factory-based invocation API introduced in this release and what (if
anything) you need to change.

---

## TL;DR

- **All existing code continues to work.** The old `LLMInvocation` /
  `start_llm` / `stop_llm` / `fail_llm` pattern is unchanged and fully
  supported.
- **New-style types are the preferred path going forward.** They are
  self-contained (no back-reference to the handler after construction) and
  support context-manager usage.
- `start_embedding` and `start_workflow` now accept either the old dataclass
  or the new string/keyword arguments — old call sites are unaffected.

---

## What changed

### 1. New invocation types

Four new classes are introduced. They live in
`opentelemetry.util.genai.invocation` (or can be imported from the package
root) and replace the old dataclass types for new code:

| New class | Replaces | `TelemetryHandler` factory |
|---|---|---|
| `InferenceInvocation` | `LLMInvocation` | `handler.start_inference(provider)` |
| `EmbeddingInvocation` (new-style) | `EmbeddingInvocation` (dataclass) | `handler.start_embedding(provider)` |
| `ToolInvocation` | `ToolCall` | `handler.start_tool(name)` |
| `WorkflowInvocation` | `Workflow` (dataclass) | `handler.start_workflow(name=...)` |

All four inherit from `GenAIInvocation` (base class).

```python
# Import from the dedicated module (recommended)
from opentelemetry.util.genai.invocation import (
    GenAIInvocation,
    InferenceInvocation,
    EmbeddingInvocation,
    ToolInvocation,
    WorkflowInvocation,
)

# Or from the package root
from opentelemetry.util.genai import (
    GenAIInvocation,
    InferenceInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
# Note: EmbeddingInvocation is exported as NewEmbeddingInvocation from the root
# to avoid shadowing the legacy types.EmbeddingInvocation dataclass
from opentelemetry.util.genai import NewEmbeddingInvocation
```

### 2. New `TelemetryHandler` factory methods

```python
from opentelemetry.util.genai import get_telemetry_handler

handler = get_telemetry_handler()

# LLM inference
inv = handler.start_inference("openai", request_model="gpt-4o")
inv.input_messages = [...]
response = call_llm(...)
inv.output_messages = [...]
inv.input_tokens = response.usage.input_tokens
inv.output_tokens = response.usage.output_tokens
inv.stop()          # or inv.fail(exception)

# Tool execution
inv = handler.start_tool("search_web", arguments={"query": "..."})
inv.tool_result = search(...)
inv.stop()

# Workflow
inv = handler.start_workflow(name="my-pipeline", framework="crewai")
# ... nested operations ...
inv.stop()

# Embedding
inv = handler.start_embedding("openai", request_model="text-embedding-3-small")
inv.input_tokens = 12
inv.stop()
```

### 3. Context-manager convenience methods

Each factory has a matching context manager on `TelemetryHandler` that calls
`stop()` on success and `fail(exc)` on exception automatically:

```python
with handler.inference("openai", request_model="gpt-4o") as inv:
    inv.input_messages = [...]
    response = call_llm(...)
    inv.output_messages = [...]
    inv.input_tokens = response.usage.input_tokens
    inv.output_tokens = response.usage.output_tokens
# span ends automatically

with handler.tool("search_web", arguments={"query": "climate change"}) as inv:
    inv.tool_result = search(...)

with handler.workflow(name="my-pipeline") as inv:
    ...

with handler.embedding("openai", request_model="text-embedding-3-small") as inv:
    inv.input_tokens = 12
```

### 4. `start_embedding` and `start_workflow` — dual signatures

These two methods now accept **either** the old dataclass **or** the new
arguments. Existing call sites are unaffected:

```python
# Old path — still works
from opentelemetry.util.genai.types import EmbeddingInvocation
old_inv = EmbeddingInvocation(request_model="...", ...)
handler.start_embedding(old_inv)
handler.stop_embedding(old_inv)

# New path
new_inv = handler.start_embedding("openai", request_model="text-embedding-3-small")
new_inv.stop()
```

```python
# Old path — still works
from opentelemetry.util.genai.types import Workflow
wf = Workflow(name="my-wf", ...)
handler.start_workflow(wf)
handler.stop_workflow(wf)

# New path
inv = handler.start_workflow(name="my-wf", framework="langchain")
inv.stop()
```

---

## What did NOT change

The following are identical to the previous release — **no changes required**
in existing instrumentations:

| API | Status |
|---|---|
| `LLMInvocation` dataclass and all its fields | Unchanged |
| `handler.start_llm` / `stop_llm` / `fail_llm` | Unchanged |
| `handler.start_retrieval` / `stop_retrieval` / `fail_retrieval` | Unchanged |
| `handler.start_tool_call` / `stop_tool_call` / `fail_tool_call` (old `ToolCall`) | Unchanged |
| `handler.start_agent` / `stop_agent` / `fail_agent` | Unchanged |
| `handler.start_step` / `stop_step` / `fail_step` | Unchanged |
| `handler.stop_workflow` / `fail_workflow` | Unchanged |
| `handler.stop_embedding` / `fail_embedding` | Unchanged |
| `handler.start_mcp_operation` / `stop_mcp_operation` / `fail_mcp_operation` | Unchanged |
| `handler.create_and_start_root` / `should_use_workflow_root` | Unchanged |
| `handler.start` / `handler.finish` / `handler.fail` (generic dispatch) | Unchanged |
| `GenAIContext`, `genai_context`, `set_genai_context`, `get_genai_context`, `clear_genai_context` | Unchanged |
| All types in `opentelemetry.util.genai.types` | Unchanged |
| `get_telemetry_handler()` | Unchanged |
| `TelemetryHandler._reset_for_testing()` | Unchanged |

---

## New-style invocation field reference

### `GenAIInvocation` (base — all types inherit these)

| Field | Type | Description |
|---|---|---|
| `provider` | `str \| None` | Provider name (e.g. `"openai"`) |
| `framework` | `str \| None` | Framework name (e.g. `"langchain"`) |
| `system` | `str \| None` | System name (e.g. `"openai"`) |
| `agent_name` | `str \| None` | Inherited from agent context stack automatically |
| `agent_id` | `str \| None` | Inherited from agent context stack automatically |
| `conversation_id` | `str \| None` | Inherited from `genai_context` automatically |
| `association_properties` | `dict` | Merged from `genai_context` automatically |
| `attributes` | `dict` | Extra span attributes (set before or after start) |
| `span` | `Span \| None` | The live OTel span; readable after construction |
| `error_type` | `str \| None` | Set automatically on `fail()` |

### `InferenceInvocation` (additional fields)

| Field | Type | Notes |
|---|---|---|
| `request_model` | `str \| None` | |
| `server_address` | `str \| None` | |
| `server_port` | `int \| None` | |
| `input_messages` | `List[InputMessage]` | Set after LLM call |
| `output_messages` | `List[OutputMessage]` | Set after LLM call |
| `input_tokens` | `int \| None` | |
| `output_tokens` | `int \| None` | |
| `request_temperature` | `float \| None` | |
| `request_max_tokens` | `int \| None` | |
| `request_top_p` | `float \| None` | |
| `request_top_k` | `int \| None` | |
| `request_frequency_penalty` | `float \| None` | |
| `request_presence_penalty` | `float \| None` | |
| `request_stop_sequences` | `List[str]` | |
| `request_choice_count` | `int \| None` | |
| `request_seed` | `int \| None` | |
| `request_stream` | `bool \| None` | |
| `request_service_tier` | `str \| None` | |
| `response_model_name` | `str \| None` | |
| `response_id` | `str \| None` | |
| `response_finish_reasons` | `List[str]` | |
| `response_service_tier` | `str \| None` | |
| `response_system_fingerprint` | `str \| None` | |
| `operation` | `str` | Defaults to `"chat"` |
| `output_type` | `str \| None` | |
| `tool_definitions` | `str \| None` | |
| `request_functions` | `list` | |
| `security_event_id` | `str \| None` | |

### `ToolInvocation` (additional fields)

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Tool name (required at start) |
| `arguments` | `Any` | Tool input arguments |
| `id` | `str \| None` | Tool call ID (pass as `tool_call_id=` to factory) |
| `tool_type` | `str \| None` | |
| `tool_description` | `str \| None` | |
| `tool_result` | `Any` | Set after the tool runs |

### `WorkflowInvocation` (additional fields)

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Workflow name |
| `workflow_type` | `str \| None` | e.g. `"crewai.crew"` |
| `description` | `str \| None` | |
| `input_messages` | `List[InputMessage]` | |
| `output_messages` | `List[OutputMessage]` | |
| `conversation_root` | `bool \| None` | Auto-set to `True` when no parent span |

### `EmbeddingInvocation` (new-style, additional fields)

| Field | Type | Notes |
|---|---|---|
| `request_model` | `str` | |
| `server_address` | `str \| None` | |
| `server_port` | `int \| None` | |
| `input_texts` | `list[str]` | |
| `input_tokens` | `int \| None` | |
| `encoding_formats` | `list[str]` | |
| `dimension_count` | `int \| None` | |
| `operation_name` | `str` | Defaults to `"embeddings"` |

---

## Invocation lifecycle

New-style invocations start **immediately on construction** (the span is open
when the factory method returns). There is no separate start call.

```
handler.start_inference(provider)
  → InferenceInvocation.__init__() → _start() → emitter.on_start()   [span open]

inv.stop()
  → _finish() → emitter.on_end()                                      [span closed]

inv.fail(exc_or_Error)
  → _finish(error) → emitter.on_error()                               [span closed, error status]
```

Do not call `stop()` or `fail()` more than once on the same invocation.

---

## Error handling

`fail()` accepts either a raw exception or an `Error` dataclass:

```python
# Raw exception — wrapped automatically
inv.fail(ValueError("something went wrong"))

# Error dataclass — for finer control over classification
from opentelemetry.util.genai._error import Error, ErrorClassification
inv.fail(Error(
    message="rate limited",
    type=RateLimitError,
    classification=ErrorClassification.REAL_ERROR,
))
```

---

## Do I need to migrate?

**No immediate action required.** The old API is fully preserved.

Migration is recommended when writing **new instrumentations** or refactoring
existing ones, because the new API:

- Eliminates the handler back-reference (`invocation._handler = self`)
- Supports context-manager usage out of the box
- Has cleaner type boundaries (no `Union` return types on factory methods)
- Is where future features will land first
