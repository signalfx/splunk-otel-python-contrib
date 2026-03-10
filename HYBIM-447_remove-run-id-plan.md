# Plan: Remove `run_id` from Shared GenAI Utils (HYBIM-447)

**Branch:** `hybim-447_remove-run-id`
**Created:** 2026-03-05

## Background

The splunk repo uses a centralized `TelemetryHandler` with two registries (`_span_registry`, `_entity_registry`) keyed by `run_id` (UUID). The upstream rejected this in favor of storing `span` and `context_token` directly on each invocation object, with each instrumentation maintaining its own local store if needed.

The upstream `types.py` has no `run_id` field at all. The upstream LangChain instrumentation has a local `_InvocationManager` and `_SpanManager` inside the package itself.

---

## Execution Order

Phases must be executed in this order, as the util layer cannot be cleaned up until all instrumentation-level callers are migrated first:

**3 → 4 → 2 → 1 → 5 → 6**

---

## Phase 1 — Remove registries from `TelemetryHandler`

**File:** `util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py`

- Remove `_span_registry: dict[str, Span]` and `_entity_registry: dict[str, GenAI]`
- Remove `_agent_context_stack` if it is keyed or managed by `run_id` (needs verification — see Open Questions)
- Remove all `str(invocation.run_id)` keying inside `start_*` / `stop_*` / `fail_*` methods
- Remove public lookup methods: `get_entity()`, `get_span_by_run_id()`, `has_span()`, `finish_by_run_id()`, `fail_by_run_id()`
- Ensure `span` and `context_token` are stored on the invocation object (as upstream does) — verify these fields already exist on the splunk types

> **Note:** Any caller that uses `get_entity()` or `get_span_by_run_id()` will break. These must be resolved in Phases 3 and 4 before Phase 1 is completed.

---

## Phase 2 — Remove `run_id` / `parent_run_id` from `GenAI` base type

**File:** `util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py`

- Remove `run_id: UUID = field(default_factory=uuid4)` (line 83)
- Remove `parent_run_id: Optional[UUID] = None` (line 84)
- Remove the `uuid4` import if no longer used elsewhere
- Verify `span` and `context_token` fields exist on the base class (upstream has them on `GenAIInvocation`); add if missing
- Check `Step` and other subtypes at lines 509–510 that also reference these fields and remove them there

---

## Phase 3 — LangChain: migrate to local invocation store

**Files:** `instrumentation-genai/opentelemetry-instrumentation-langchain/`

This is the most complex phase. The splunk LangChain callback handler has rich agent/chain/tool/step/workflow support that the upstream's simplified handler does not. That functionality is preserved, but state storage moves local.

### Add a local `_InvocationManager`

Mirror the upstream pattern inside the langchain package:

- Keyed by LangChain's `run_id` (UUID — this comes from LangChain's callback API, not from `GenAI.run_id`)
- Tracks parent-child relationships using `parent_run_id` (also from LangChain's callback API)
- Stores the created invocation objects (`LLMInvocation`, `AgentInvocation`, `ToolCall`, etc.)

### Rework `callback_handler.py`

- Replace all 7 calls to `self._handler.get_entity(run_id)` with lookups into the local `_InvocationManager`:
  - Line 240: in `_find_nearest_agent()`
  - Line 339: in `on_chain_start()` — check if ToolCall exists
  - Line 404: in `on_chain_end()` — resolve entity type
  - Line 550: in `on_llm_start()` — get created LLMInvocation
  - Line 562: in `on_llm_end()` — retrieve LLMInvocation
  - Line 654: in `on_tool_start()` — check if ToolCall exists
  - Line 699: in `on_tool_end()` — retrieve ToolCall
- Rework `_find_nearest_agent()` (lines 235–246) to walk the local manager's parent chain instead of the `TelemetryHandler` entity registry — the logic is identical, only the store changes
- Remove `run_id=run_id` and `parent_run_id=parent_run_id` when constructing `AgentInvocation`, `LLMInvocation`, `ToolCall`, `Step` (these fields will no longer exist on those types after Phase 2)
- Clean up state in the local manager on `stop_*` and `fail_*` callbacks (matching upstream's `delete_invocation_state` pattern)

---

## Phase 4 — LlamaIndex: cleanup residual field assignments

**Files:** `instrumentation-genai/opentelemetry-instrumentation-llamaindex/`

LlamaIndex already has its own `_InvocationManager` and does not use the central registry. The local store itself is unaffected (it uses `event_id` strings, not the `GenAI.run_id` field).

- Audit `callback_handler.py` and `workflow_instrumentation.py` (lines 71, 181, 188) for any remaining assignments to `entity.run_id` or `entity.parent_run_id` on GenAI objects — these will be invalid after Phase 2 and must be removed

---

## Phase 5 — Evals package

**File:** `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/serialization.py` (lines 49–51, 262–273)

- Identify what `run_id` is used for in serialization — likely as a correlation key in eval payloads
- Determine replacement: use `trace_id` / `span_id` (already stored on the invocation via `span_context`) as the correlation identifier instead
- This may require a small protocol decision: confirm with evals consumers whether the key field name changes

---

## Phase 6 — Tests and examples

- `test_callback_handler_agent.py` — tests that directly assert on `parent_run_id` propagation should be updated to test observable behavior (correct span parentage) rather than internal field values
- Update or remove example files that construct GenAI objects with `run_id`:
  - `util/opentelemetry-util-genai/examples/invocation_example.py`
  - `util/opentelemetry-util-genai/examples/langgraph_agent_example.py`
  - `util/opentelemetry-util-genai/examples/langgraph_simple_agent_example.py`

---

## Open Questions

These should be resolved before or during implementation:

1. **`_agent_context_stack`** in `TelemetryHandler` — this is a stack of `(name, run_id)` tuples used for implicit agent context propagation. Does this need to survive? If so, it can be changed to store `(name, span_id)` instead, or replaced with OTel context propagation directly.

2. **`span_context.py`** (`extract_span_context`, `store_span_context`) — this module stores `trace_id`, `span_id`, `trace_flags` on the GenAI object. Does this module still serve a purpose after the registries are gone, or does its use case collapse into reading from `invocation.span` directly?

3. **`debug.py`** (lines 95–100) references `run_id` for logging. Low risk — `span_id` is a suitable replacement identifier.

4. **Evals protocol** — does the evals serialization format need a backwards-compatible transition period, or is a clean break acceptable?
