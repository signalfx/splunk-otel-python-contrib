# PR: GenAI Session & Conversation ID Support

**Branch**: `feature/session-support-main` → `main`
**Date**: 2026-02-10
**Diff**: 54 files changed, 6,807 insertions, 512 deletions

---

## Summary

This PR adds end-to-end session/conversation tracking to the SDOT GenAI
instrumentation stack. It introduces `gen_ai.conversation.id`, `user.id`,
`customer.id`, and arbitrary `genai.association.*` attributes that propagate
across service boundaries via W3C OTel Baggage.

The feature spans three packages:

| Package | Changes |
|---------|---------|
| `opentelemetry-util-genai` | Session context API, GenAI type fields, env vars, metrics support |
| `opentelemetry-instrumentation-fastmcp` | Cross-service baggage propagation, server-side allowlist, dynamic transport detection |
| Documentation & examples | Upstream proposal, research docs, e2e demo |

---

## What's New

### 1. Session Context API (`opentelemetry-util-genai`)

**New public API** exported from `opentelemetry.util.genai`:

| API | Purpose |
|-----|---------|
| `SessionContext` | Dataclass holding `session_id`, `user_id`, `customer_id`, `association_properties` |
| `set_session_context()` | Set session context for current scope (imperative) |
| `session_context()` | Context manager — auto-restores previous session on exit |
| `get_session_context()` | Read current session context |
| `clear_session_context()` | Reset session context |

**Usage:**

```python
from opentelemetry.util.genai import session_context

with session_context(
    session_id="conv-123",
    user_id="user-456",
    association_properties={"tenant": "acme"},
):
    # All GenAI spans within this scope get session attributes
    result = chain.invoke(...)
```

**Priority order** for session values: explicit invocation value > ContextVar >
OTel Baggage > environment variable.

### 2. GenAI Base Type Fields (`types.py`)

Added to the `GenAI` base dataclass (inherited by `LLMInvocation`,
`AgentInvocation`, `Workflow`, `ToolCall`, etc.):

| Field | Span Attribute | Metadata |
|-------|---------------|----------|
| `session_id` | `session.id` | `{"semconv": "session.id"}` |
| `user_id` | `user.id` | `{"semconv": "user.id"}` |
| `customer_id` | `customer.id` | `{"semconv": "customer.id"}` |
| `association_properties` | `genai.association.<key>` | `{"semconv_prefix": "genai.association."}` |

The `semantic_convention_attributes()` method auto-emits these on every span
via the existing `semconv` metadata pattern. A new `semconv_prefix` pattern
supports dictionary fields that expand into multiple span attributes.

### 3. OTel Baggage Propagation (`handler.py`)

When `OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION=baggage`:

- `set_session_context()` writes session fields to OTel Baggage
- `_get_session_from_baggage()` reads them back on the receiving side
- Baggage keys: `gen_ai.conversation.id`, `user.id`, `customer.id`,
  `genai.association.<key>`

### 4. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION` | `contextvar` or `baggage` | `contextvar` |
| `OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS` | Session attrs on metrics (⚠️ cardinality) | *(empty)* |
| `OTEL_INSTRUMENTATION_GENAI_SESSION_ID` | Static session ID | *(empty)* |
| `OTEL_INSTRUMENTATION_GENAI_USER_ID` | Static user ID | *(empty)* |
| `OTEL_INSTRUMENTATION_GENAI_CUSTOMER_ID` | Static customer ID | *(empty)* |
| `OTEL_INSTRUMENTATION_GENAI_BAGGAGE_ALLOWED_KEYS` | Server-side baggage allowlist | `gen_ai.conversation.id,user.id,customer.id` |

### 5. Session Attributes on Metrics (`emitters/utils.py`)

New `get_session_metric_attributes()` function, controlled by
`OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS`. Supports `all`,
comma-separated list, and legacy `session.id` → `gen_ai.conversation.id`
normalization.

### 6. Cross-Service Propagation (FastMCP)

**Replaced** the original `_meta`-based propagation with standard OTel
Propagation API (`propagate.inject()` / `propagate.extract()`). This single
mechanism now carries both W3C TraceContext and Baggage:

- **Client side**: `propagate.inject(carrier)` writes `traceparent`,
  `tracestate`, and `baggage` into the MCP `_meta` field
- **Server side**: `propagate.extract(carrier)` restores trace context +
  baggage, then `restore_session_from_context()` extracts session fields

### 7. Server-Side Baggage Allowlist (`propagation.py`)

New `OTEL_INSTRUMENTATION_GENAI_BAGGAGE_ALLOWED_KEYS` env var controls which
baggage keys the server extracts:

| Value | Behaviour |
|-------|-----------|
| *(unset)* | Default — `gen_ai.conversation.id`, `user.id`, `customer.id` |
| `*` | Accept all baggage keys |
| Comma list | Accept only listed keys; `genai.association.*` = wildcard |
| `none` | Reject all incoming baggage |

Implementation: `_get_allowed_keys()`, `_is_key_allowed()`,
`restore_session_from_context()` in `propagation.py`.

### 8. Dynamic Transport Detection (`utils.py`)

New functions for detecting `network.transport` at runtime:

| Function | Context | Returns |
|----------|---------|---------|
| `detect_client_network_transport()` | Client side | `"pipe"` / `"tcp"` / `"inproc"` |
| `detect_server_network_transport()` | Server side | `"pipe"` / `"tcp"` |

Replaces the previous hardcoded `"pipe"` value. Uses class name matching
(SSE/HTTP → `tcp`, Stdio → `pipe`, FastMCPTransport → `inproc`).

### 9. Agentic E2E Demo (`examples/e2e/`)

**`client.py`** rewritten as an agentic client:
- Creates `AgentInvocation` + child `ToolCall` spans via `TelemetryHandler`
- `--session-id`, `--user-id`, `--property KEY=VALUE` CLI flags
- `--otlp` flag for OTLP export
- `--console` flag for console debug output

**`server_instrumented.py`** updated:
- `get_session_info` tool returns session context + association properties
- Server-side baggage allowlist support

**`README.md`** expanded (254 → 582 lines):
- Full documentation of baggage allowlist
- Two-terminal examples with filtering
- Architecture diagram showing allowlist
- Security guidance

---

## Files Changed

### `opentelemetry-util-genai` (12 files, +1,451 / −18)

| File | Change |
|------|--------|
| `__init__.py` | Export `SessionContext`, `set_session_context`, `session_context`, etc. |
| `attributes.py` | Add `SESSION_ID`, `USER_ID`, `CUSTOMER_ID` constants |
| `types.py` | Add `session_id`, `user_id`, `customer_id`, `association_properties` to `GenAI` base |
| `handler.py` | Add `SessionContext`, `set_session_context()`, `session_context()`, `_apply_session_context()`, baggage helpers (+296 lines) |
| `environment_variables.py` | Add 6 new env vars with docstrings (+89 lines) |
| `emitters/utils.py` | Add `get_session_metric_attributes()`, `_get_session_metric_include_set()` (+81 lines) |
| `emitters/metrics.py` | Integrate session metric attributes |
| `CHANGELOG.md` | Document v0.1.10 session support |
| `docs/session-context.md` | New: detailed session context documentation |
| `examples/invocation_example.py` | Add session context usage example |
| `tests/test_session_context.py` | **New**: 439 lines — session context API tests |
| `tests/test_session_baggage.py` | **New**: 243 lines — baggage propagation tests |

### `opentelemetry-instrumentation-fastmcp` (17 files, +1,702 / −372)

| File | Change |
|------|--------|
| `propagation.py` | Rewrite: baggage allowlist (`_get_allowed_keys`, `_is_key_allowed`, `restore_session_from_context`) |
| `transport_instrumentor.py` | Use `propagate.inject/extract` instead of manual `_meta` handling; dynamic transport detection |
| `client_instrumentor.py` | Add `network.transport` detection |
| `server_instrumentor.py` | Minor: pass transport info |
| `utils.py` | Add `detect_client_network_transport()`, `detect_server_network_transport()` |
| `CHANGELOG.md` | Update propagation description |
| `examples/e2e/client.py` | Rewrite as agentic client with `--property` flag |
| `examples/e2e/server.py` | Add `get_session_info` tool |
| `examples/e2e/server_instrumented.py` | Add baggage allowlist + session info |
| `examples/e2e/README.md` | Expand with allowlist docs, architecture, security |
| `tests/test_propagation.py` | Rewrite for allowlist-based filtering |
| `tests/test_session_baggage_propagation.py` | **New**: 332 lines — cross-service baggage tests |
| `tests/test_transport_propagation.py` | **New**: 376 lines — transport propagation tests |
| `tests/test_utils.py` | **New**: 136 lines — transport detection tests |
| `tests/test_client_instrumentor.py` | Add transport detection tests |
| `tests/test_transport_instrumentor.py` | Update for new propagation |

### Documentation (7 files, +3,535 new)

| File | Description |
|------|-------------|
| `docs/feat-session-support.md` | Feature plan with implementation options |
| `docs/session-instrumentation-overview.md` | Comparison: Galileo, Langfuse, Traceloop, Phoenix, AGNTCY |
| `docs/upstream-proposal-2026-02-10.md` | Full upstream proposal for OTel GenAI SIG |
| `docs/upstream-proposal.md` | Original upstream proposal (superseded) |
| `docs/galileo-setup.md` | Galileo integration setup guide |
| `docs/langfuse-setup.md` | Langfuse integration setup guide |
| `docs/traceloop-setup.md` | Traceloop integration setup guide |

### Other

| File | Change |
|------|--------|
| `.gitignore` | Add telemetry output files |
| `instrumentation-genai/opentelemetry-instrumentation-aidefense/` | Minor import path fix |
| `instrumentation-genai/opentelemetry-instrumentation-langchain/examples/` | Session context in demo |
| Various `CHANGELOG.md` / `version.py` | Version bumps |

---

## Test Coverage

| Package | New Test Files | Test Count |
|---------|---------------|------------|
| `opentelemetry-util-genai` | `test_session_context.py` (439 lines), `test_session_baggage.py` (243 lines) | 132 total |
| `opentelemetry-instrumentation-fastmcp` | `test_session_baggage_propagation.py` (332 lines), `test_transport_propagation.py` (376 lines), `test_utils.py` (136 lines) | 111 total |

All tests pass. Lint clean (`ruff check` + `ruff format`).

---

## Breaking Changes

**None.** All new functionality is additive:

- New fields on `GenAI` base type default to `None` (no impact on existing code)
- New environment variables are opt-in with safe defaults
- Baggage propagation is disabled by default (`contextvar` mode)
- Server-side allowlist defaults to the three core session keys
- Public API exports are new additions to `__init__.py`

---

## Architecture

```
┌─────────────────────────────┐
│  Application Code           │
│  session_context(           │
│    session_id="conv-123",   │
│    user_id="user-456",      │
│    association_properties=  │
│      {"tenant": "acme"})    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  ContextVar + OTel Baggage  │
│  ┌────────────────────────┐ │
│  │ gen_ai.conversation.id │ │
│  │ user.id                │ │
│  │ customer.id            │ │
│  │ genai.association.*    │ │
│  └────────────────────────┘ │
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
┌──────────┐    ┌──────────────┐
│  Spans   │    │   Baggage    │
│ (local)  │    │ (cross-svc)  │
│          │    │              │
│ semconv  │    │ propagate.   │
│ attrs    │    │  inject()    │
└──────────┘    └──────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │ MCP / HTTP /  │
               │ gRPC carrier  │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │  Server-side  │
               │  Allowlist    │
               │  Filtering    │
               │  (BAGGAGE_    │
               │  ALLOWED_KEYS)│
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │ Server spans  │
               │ with filtered │
               │ session attrs │
               └───────────────┘
```

---

## Related Issues / Proposals

- [semantic-conventions#2883](https://github.com/open-telemetry/semantic-conventions/issues/2883) — Add `session.id` to GenAI semconv
- [semantic-conventions#1872](https://github.com/open-telemetry/semantic-conventions/issues/1872) — GenAI user and session conventions
- [semantic-conventions#3418](https://github.com/open-telemetry/semantic-conventions/issues/3418) — Entry span with session attributes
- [docs/upstream-proposal-2026-02-10.md](upstream-proposal-2026-02-10.md) — Full upstream proposal

---

## How to Test

```bash
# Install packages
pip install -e ./util/opentelemetry-util-genai
pip install -e "./instrumentation-genai/opentelemetry-instrumentation-fastmcp[instruments,test]"

# Run tests
pytest ./util/opentelemetry-util-genai/tests/ -v        # 132 tests
pytest ./instrumentation-genai/opentelemetry-instrumentation-fastmcp/tests/ -v  # 111 tests

# Lint
make lint

# E2E demo (two terminals)
# Terminal 1:
export OTEL_SERVICE_NAME="mcp-calculator-server"
export OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION="baggage"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
python instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/e2e/server_instrumented.py --sse --port 8000

# Terminal 2:
export OTEL_SERVICE_NAME="mcp-calculator-client"
export OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION="baggage"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric"
python instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/e2e/client.py \
    --server-url http://localhost:8000/sse --console \
    --session-id "conv-42" --user-id "alice" \
    --property tenant=acme --property env=staging
```
