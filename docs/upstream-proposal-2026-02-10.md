# Upstream Proposal: GenAI Session Propagation for OpenTelemetry

**Date**: 2026-02-10  
**Author**: Splunk Distro for OpenTelemetry (SDOT) Team  
**Status**: Draft  
**Target**: [open-telemetry/semantic-conventions](https://github.com/open-telemetry/semantic-conventions) — GenAI SIG

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Motivation and Use Cases](#motivation-and-use-cases)
- [Related Upstream Issues](#related-upstream-issues)
- [OTel GenAI SIG Contribution Process](#otel-genai-sig-contribution-process)
- [Proposed Semantic Conventions](#proposed-semantic-conventions)
  - [Core Model: Turn = Trace, Session = Grouping Key](#core-model-turn--trace-session--grouping-key)
  - [Custom Association Attributes](#custom-association-attributes)
- [SDOT Reference Implementation](#sdot-reference-implementation)
- [Cross-RPC Session Propagation](#cross-rpc-session-propagation)
- [Edge Cases and Security Considerations](#edge-cases-and-security-considerations)
- [Comparison with Other Platforms](#comparison-with-other-platforms)
- [Implementation Roadmap](#implementation-roadmap)
- [Open Questions for Community](#open-questions-for-community)
- [References](#references)

---

## Executive Summary

This proposal standardizes session and conversation tracking in GenAI observability
through OpenTelemetry semantic conventions. It is based on the production
implementation in Splunk Distro for OpenTelemetry (SDOT) and informed by research
across Traceloop/OpenLLMetry, Arize-AI/Phoenix, Langfuse, Galileo, and the
AGNTCY project.

The core goals are:

1. **Standardized session attribute** — Use `gen_ai.conversation.id` as the standard upstream convention for session/conversation tracking in GenAI observability
2. **Turn = Trace model** — Each conversational turn maps to a single trace ID; `gen_ai.conversation.id` is the grouping key across turns
3. **User and custom attribute propagation** — Propagate `enduser.id` and arbitrary association attributes (Traceloop-style) via OTel Baggage
4. **Cross-RPC propagation** — Via W3C Baggage header for MCP and agent-to-agent protocols
5. **Restriction mechanisms** — Policies for disabling propagation at trust boundaries

### Key Decisions Aligned with SIG Consensus

| Decision | Rationale |
|----------|-----------|
| Use `gen_ai.conversation.id` as the standard session attribute | Aligns with GenAI semconv namespace; unambiguous GenAI-specific grouping key |
| Each turn = one trace ID | The session/conversation attribute groups traces across turns; within a turn, trace context provides correlation |
| Use existing `enduser.id` (not `gen_ai.user.id`) | lmolkova confirmed in [#1872](https://github.com/open-telemetry/semantic-conventions/issues/1872): existing semconv attribute is sufficient |
| Support custom association attributes | Traceloop's `set_association_properties()` pattern is widely adopted; support arbitrary key-value propagation |
| Propagate via standard OTel Baggage | W3C standard, supported across all OTel SDKs |

---

## Motivation and Use Cases

### Why `gen_ai.conversation.id` Matters for GenAI

The GenAI SIG (specifically lmolkova in [#2883](https://github.com/open-telemetry/semantic-conventions/issues/2883))
has asked for concrete scenarios demonstrating why a session/conversation
grouping attribute is needed. Here are the key scenarios:

#### Scenario 1: Multi-Turn Conversational Agent (Turn = Trace)

A user engages in a multi-turn conversation with a chatbot powered by an LLM
orchestrator (e.g., LangChain). **Each turn produces exactly one trace** (a single
trace ID). Within that trace, all operations — tool calls, retrieval, LLM
invocations — are child spans. `gen_ai.conversation.id` is the grouping key across turns.

```
Session attribute: gen_ai.conversation.id = "conv-abc123"
User attribute:    enduser.id             = "user-456"

├─ Turn 1: "What is observability?"
│  └─ Trace A (trace_id=aaa...)        ← one trace per turn
│     ├─ Span: LangChain agent run     (root span, gen_ai.conversation.id=conv-abc123)
│     ├─ Span: Vector DB retrieval     (child span)
│     └─ Span: OpenAI chat completion  (child span)
│
├─ Turn 2: "How does it relate to monitoring?"
│  └─ Trace B (trace_id=bbb...)        ← new trace, same conversation
│     ├─ Span: LangChain agent run     (root span, gen_ai.conversation.id=conv-abc123)
│     ├─ Span: OpenAI chat completion  (child span)
│     └─ Span: Tool call               (child span)
│
└─ Turn 3: "Show me an example"
   └─ Trace C (trace_id=ccc...)        ← new trace, same conversation
      ├─ Span: LangChain agent run     (root span, gen_ai.conversation.id=conv-abc123)
      └─ Span: OpenAI code generation  (child span)
```

`gen_ai.conversation.id` groups traces A, B, C into a single conversation for
analysis like "average response quality per conversation" or "conversation abandonment rate."
Within each trace, the standard OTel parent-child span relationship provides
full causal ordering.

> **Key insight**: `trace_id` already provides per-turn correlation.
> `gen_ai.conversation.id` adds the cross-turn grouping that `trace_id` alone cannot provide.

#### Scenario 2: Multi-Agent Orchestration via MCP (Turn = Trace Across Services)

An orchestrator agent delegates tasks to specialized MCP tool servers within
a single turn. The entire turn is **one trace** — MCP calls create child spans
within it. The session attribute and user identity propagate via Baggage:

```
Turn: user asks "Analyze the latest security incidents"
Trace ID: xxx-yyy-zzz (single trace for this turn)
Session:  gen_ai.conversation.id = "conv-xyz789"
User:     enduser.id = "user-456"
Custom:   genai.association.department = "security"

├─ Orchestrator Agent (Service A)
│  ├─ Span: agent run                    (root, all attrs from Baggage)
│  ├─ Span: MCP call → Search Agent      → Service B (child span, same trace)
│  │  └─ Span: search execution          (Service B, gen_ai.conversation.id from Baggage)
│  ├─ Span: MCP call → Analysis Agent    → Service C (child span, same trace)
│  │  └─ Span: analysis execution        (Service C, gen_ai.conversation.id from Baggage)
│  └─ Span: LLM call → OpenAI           (child span, baggage cleared)
```

#### Scenario 3: A2A (Agent-to-Agent) Protocol

The AGNTCY/observe project ([91pavan's PoC in #2883](https://github.com/open-telemetry/semantic-conventions/issues/2883))
demonstrates session propagation in A2A scenarios where `gen_ai.conversation.id`
travels via OTel Baggage across agent boundaries.

#### Scenario 4: Evaluation Correlation

Session-level evaluation metrics (e.g., per-session hallucination rate, per-session
user satisfaction) require grouping all LLM invocations within a session for
aggregate analysis.

### Standard Session Attribute: `gen_ai.conversation.id`

The standard upstream convention for session/conversation tracking in GenAI
observability is **`gen_ai.conversation.id`**. This attribute:

| Aspect | `gen_ai.conversation.id` |
|--------|--------------------------|
| Scope | Session/conversation grouping key across turns |
| Cross-service | Propagated via OTel Baggage |
| Applicability | GenAI-specific, within the `gen_ai.*` semconv namespace |
| Status | Proposed as RECOMMENDED on GenAI spans |
| Multi-agent | Propagates across agent boundaries via Baggage |
| Typical use | Multi-turn chatbot, agentic workflows, MCP orchestration |

**Rationale for choosing `gen_ai.conversation.id` over `session.id`:**

- **GenAI-specific** — sits within the `gen_ai.*` namespace alongside other GenAI
  conventions (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`)
- **Unambiguous** — `session.id` is general-purpose and may conflict with web/mobile
  session concepts in mixed telemetry pipelines
- **Conversation-centric** — GenAI interactions are fundamentally conversational;
  the attribute name directly reflects its semantic meaning
- **Upstream alignment** — follows the naming pattern established by the GenAI SIG

---

## Related Upstream Issues

### Active Issues

| Issue | Title | Status | Key Discussion |
|-------|-------|--------|----------------|
| [#2883](https://github.com/open-telemetry/semantic-conventions/issues/2883) | Add `session.id` attribute to GenAI semantic conventions | `triage:deciding:needs-info` | lmolkova requests concrete scenarios; 91pavan has working PoC |
| [#1872](https://github.com/open-telemetry/semantic-conventions/issues/1872) | Add Gen AI user and session Semantic Conventions | Todo (GenAI SIG project) | lmolkova confirmed: use existing `session.id`, document in GenAI conventions |
| [#3418](https://github.com/open-telemetry/semantic-conventions/issues/3418) | Enhancing OTel GenAI Semantic Conventions with a User-Centric Entry Span | Open | Proposes `gen_ai.operation.name = enter` with `session.id` and `enduser.id` as RECOMMENDED |

### Closed PRs

| PR | Title | Outcome |
|----|-------|---------|
| [#2594](https://github.com/open-telemetry/semantic-conventions/pull/2594) | Add session.id to GenAI spans | Closed — approach needs more discussion |

### Recommended Next Steps

1. **Comment on [#2883](https://github.com/open-telemetry/semantic-conventions/issues/2883)** with the concrete scenarios above (specifically Scenarios 1–4)
2. **Reference this proposal** and the SDOT reference implementation
3. **Align with [#3418](https://github.com/open-telemetry/semantic-conventions/issues/3418)** on the entry span proposal which also recommends session attributes
4. If consensus forms, open a PR modifying the GenAI semconv YAML to add `gen_ai.conversation.id` as RECOMMENDED

---

## OTel GenAI SIG Contribution Process

### How Semantic Conventions Are Changed

Based on [semantic-conventions CONTRIBUTING.md](https://github.com/open-telemetry/semantic-conventions/blob/main/CONTRIBUTING.md):

1. **Modify the YAML model** under `model/{namespace}/registry.yaml`
   - For GenAI session conventions: modify `model/gen-ai/registry.yaml` to add `gen_ai.conversation.id`
   - Add `gen_ai.conversation.id` as RECOMMENDED on relevant span types

2. **Generate markdown** via `make generate-all`
   - This generates the human-readable convention docs from the YAML model

3. **Check backward compatibility** via `make check-policies`
   - Ensures no breaking changes to existing conventions

4. **Add changelog entry** via `.chloggen/*.yaml` file

5. **Open PR** requiring 2 approvals from `@open-telemetry/semconv-genai-approvers` (CODEOWNERS)

### GenAI SIG Governance

- **Area label**: `area:gen-ai`
- **Approvers team**: `@open-telemetry/semconv-genai-approvers`
- **Key maintainer**: lmolkova (Grafana Labs) — primary responder on session-related issues
- **Meeting cadence**: Weekly GenAI SIG meetings (check [OTel community calendar](https://github.com/open-telemetry/community#calendar))
- **Semconv status**: `Development` (breaking changes still possible)

### What We Would Propose to Change

```yaml
# model/gen-ai/spans.yaml (conceptual addition)
groups:
  - id: gen_ai.client
    attributes:
      - ref: gen_ai.conversation.id
        requirement_level: recommended
        note: >
          Conversation/session identifier for grouping related GenAI
          interactions across turns (traces). Each turn SHOULD produce
          one trace; gen_ai.conversation.id groups traces into a
          conversation/session.
      - ref: enduser.id
        requirement_level: recommended
        note: >
          User identifier for attributing GenAI interactions to a
          specific end user. Propagated via OTel Baggage for
          cross-service correlation.
```

---

## Proposed Semantic Conventions

### Attributes to Document in GenAI Conventions

These attributes already exist in the OTel semantic conventions registry. The
proposal is to **reference and recommend** them in GenAI span conventions:

| Attribute | Type | Requirement | Description | Example |
|-----------|------|-------------|-------------|---------|
| `gen_ai.conversation.id` | string | Recommended | Conversation/session identifier grouping related GenAI interactions across turns | `"conv-abc123"` |
| `enduser.id` | string | Recommended | End-user identifier | `"user-456"` |

### Optional Extension Attributes

These are used in SDOT but may be proposed later once `gen_ai.conversation.id` is accepted:

| Attribute | Type | Requirement | Description | Example |
|-----------|------|-------------|-------------|---------|
| `customer.id` | string | Opt-In | Customer/tenant identifier for multi-tenant systems | `"customer-789"` |

### Relationship to Existing Conventions

| Existing Attribute | Relationship to `gen_ai.conversation.id` |
|---|---|
| `session.id` | General-purpose session attribute (web, mobile); `gen_ai.conversation.id` is the GenAI-specific equivalent |
| `mcp.session.id` | MCP protocol-level session (transport layer); distinct from application-level `gen_ai.conversation.id` |
| `gen_ai.agent.id` | Agent instance ID (per-invocation); orthogonal to conversation |
| `enduser.id` | User who owns the conversation/session |

---

## SDOT Reference Implementation

### Architecture Overview

SDOT separates **instrumentation capture** from **telemetry emission** using a
plugin-based architecture. Session support is woven into the core types and
lifecycle handler.

```
┌─────────────────────────────────────────────────┐
│  Application Code                               │
│  with session_scope(session_id="conv-123"):     │
│      result = chain.invoke(...)                  │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Instrumentation Layer (e.g., LangChain)        │
│  Creates GenAI types (LLMInvocation, etc.)      │
│  Session fields auto-populated from context     │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  TelemetryHandler (util-genai)                  │
│  • start_llm() / stop_llm() lifecycle           │
│  • SessionContext from ContextVar               │
│  • Agent context stack for child propagation    │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Emitters (pluggable via entry points)          │
│  • SpanEmitter → OTel Spans with session attrs  │
│  • MetricsEmitter → OTel Metrics                │
│  • EventEmitter → OTel Log Events               │
└─────────────────────────────────────────────────┘
```

### Core GenAI Types with Session Support

Session fields are part of the `GenAI` base dataclass, automatically propagated
to all invocation types via Python inheritance:

```python
# util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py
@dataclass(kw_only=True)
class GenAI:
    """Base type for all GenAI telemetry entities."""
    # ... existing fields ...

    # Session/User Context (association properties for session tracking)
    session_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "gen_ai.conversation.id"},
    )
    user_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "user.id"},
    )
    customer_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "customer.id"},
    )

    # Custom association attributes (Traceloop-style)
    # Arbitrary key-value pairs propagated to all spans in scope
    association_properties: Optional[dict[str, str]] = field(
        default=None,
        metadata={"semconv_prefix": "genai.association."},
    )
```

**Key design decisions:**

- Uses `metadata={"semconv": "gen_ai.conversation.id"}` pattern for automatic attribute emission
  via the `semantic_convention_attributes()` method
- Session fields flow through all GenAI types: `LLMInvocation`, `AgentInvocation`,
  `Workflow`, `ToolCall`, `EmbeddingInvocation`
- `association_properties` supports arbitrary custom key-value pairs, emitted
  with a configurable prefix (default: `genai.association.*`)

### TelemetryHandler SessionContext

The `TelemetryHandler` manages session context via `ContextVar` for async safety.
The `SessionContext` carries both well-known attributes and custom association
properties:

```python
# util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py

@dataclass
class SessionContext:
    """Immutable session context container.

    Carries session identity, user identity, and custom association
    properties. All values are propagated to child spans and optionally
    across service boundaries via OTel Baggage.
    """
    session_id: str | None = None
    user_id: str | None = None
    customer_id: str | None = None
    association_properties: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([self.session_id, self.user_id, self.customer_id,
                        self.association_properties])

    def to_baggage(self, ctx: Context | None = None) -> Context:
        """Convert to OTel baggage context for cross-service propagation."""
        ctx = ctx or context.get_current()
        if self.session_id:
            ctx = baggage.set_baggage("gen_ai.conversation.id", self.session_id, ctx)
        if self.user_id:
            ctx = baggage.set_baggage("enduser.id", self.user_id, ctx)
        if self.customer_id:
            ctx = baggage.set_baggage("customer.id", self.customer_id, ctx)
        # Propagate custom association properties
        for key, value in self.association_properties.items():
            ctx = baggage.set_baggage(
                f"genai.association.{key}", value, ctx
            )
        return ctx
```

### Agent Context Stack for Child Span Propagation

The `TelemetryHandler` maintains an agent context stack so nested LLM calls
automatically inherit parent agent context:

```python
class TelemetryHandler:
    # Active agent identity stack (name, id) for implicit propagation
    _agent_context_stack: list[tuple[str, str]] = []

    def start_agent(self, agent: AgentInvocation) -> AgentInvocation:
        # Push agent identity context
        if isinstance(agent, AgentInvocation) and agent.name:
            self._agent_context_stack.append((agent.name, str(agent.run_id)))

    def start_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        # Implicit agent inheritance — no manual threading required
        if (not invocation.agent_name or not invocation.agent_id) \
                and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
```

### Proposed SessionPropagator API

```python
# Proposed: opentelemetry-util-genai session.py

from contextvars import ContextVar
from contextlib import contextmanager
from opentelemetry import baggage, context

_session_context: ContextVar[SessionContext] = ContextVar(
    "genai_session", default=SessionContext()
)

def set_session(
    session_id: str | None = None,
    user_id: str | None = None,
    customer_id: str | None = None,
    association_properties: dict[str, str] | None = None,
    propagate_via_baggage: bool = True,
) -> object:
    """Set session context for current execution scope.

    Args:
        session_id: Session/conversation identifier (emitted as
            gen_ai.conversation.id)
        user_id: End-user identifier (emitted as enduser.id)
        customer_id: Customer/tenant identifier
        association_properties: Custom key-value pairs propagated to
            all spans in scope (Traceloop-style). Emitted with
            configurable prefix (default: genai.association.*)
        propagate_via_baggage: Whether to propagate via OTel Baggage
            for cross-service visibility (default: True)
    """
    session = SessionContext(
        session_id, user_id, customer_id,
        association_properties=association_properties or {},
    )
    token = _session_context.set(session)
    if propagate_via_baggage:
        ctx = session.to_baggage()
        context.attach(ctx)
    return token

def set_association_properties(
    properties: dict[str, str],
    propagate_via_baggage: bool = True,
) -> object:
    """Set custom association properties (Traceloop-compatible API).

    Merges with existing session context. Properties are propagated
    to all child spans and optionally across service boundaries.

    Usage:
        set_association_properties({
            "chat_id": "chat-123",
            "tenant": "acme-corp",
            "environment": "production",
        })
    """
    current = _session_context.get()
    merged = {**current.association_properties, **properties}
    return set_session(
        session_id=current.session_id,
        user_id=current.user_id,
        customer_id=current.customer_id,
        association_properties=merged,
        propagate_via_baggage=propagate_via_baggage,
    )

@contextmanager
def session_scope(
    session_id: str | None = None,
    user_id: str | None = None,
    association_properties: dict[str, str] | None = None,
    propagate_via_baggage: bool = True,
):
    """Context manager for scoped session tracking.

    Usage:
        # Basic session with user
        with session_scope(session_id="conv-123", user_id="user-456"):
            result = chain.invoke(...)

        # With custom association properties
        with session_scope(
            session_id="conv-123",
            user_id="user-456",
            association_properties={
                "chat_id": "chat-789",
                "department": "engineering",
            },
        ):
            result = chain.invoke(...)
    """
    token = set_session(
        session_id, user_id,
        association_properties=association_properties,
        propagate_via_baggage=propagate_via_baggage,
    )
    try:
        yield get_session()
    finally:
        clear_session(token)
```

---

## Cross-RPC Session Propagation

### Mechanism: OTel Baggage via W3C Header

Session context, user identity, and custom association properties propagate
across service boundaries using the standard W3C Baggage header alongside
`traceparent`/`tracestate`:

```
HTTP Header:
  traceparent: 00-abc123def456...
  tracestate: vendor=splunk
  baggage: gen_ai.conversation.id=conv-123,enduser.id=user-456,genai.association.chat_id=chat-789,genai.association.department=engineering
```

Each turn starts a new trace (`traceparent` has a fresh `trace_id`), but the
`baggage` header carries the same session, user, and association attributes
across all turns and service boundaries.

### Non-HTTP Protocols (MCP, gRPC, etc.)

The propagation mechanism is always OTel Baggage — the same
`propagate.inject()` / `propagate.extract()` API used by HTTP instrumentations.
Each protocol simply provides its own carrier:

| Protocol | Carrier |
|----------|--------|
| HTTP | Request headers |
| gRPC | Metadata |
| MCP | `params.meta` object |

The SDOT FastMCP instrumentor calls `propagate.inject(carrier)` on the client
side and `propagate.extract(carrier)` on the server side. The standard
`CompositePropagator` handles both W3C TraceContext and Baggage propagators,
so a single inject/extract round-trip carries **all** context — trace identity,
session attributes, user identity, and association properties:

```python
# Client side — standard OTel propagation (protocol-agnostic)
carrier = {}
propagate.inject(carrier)
# carrier = {"traceparent": "00-...", "tracestate": "...",
#            "baggage": "gen_ai.conversation.id=conv-123,enduser.id=user-456,..."}

# Server side — standard OTel extraction
ctx = propagate.extract(carrier)
context.attach(ctx)  # Restores trace context + baggage
```

> **Key point:** No protocol-specific propagation logic is needed. Any
> OTel-instrumented protocol that provides a metadata carrier can propagate
> session context identically via standard Baggage.

---

## Edge Cases and Security Considerations

This section documents scenarios where session attribute propagation should be
**disabled**, **restricted**, or **sanitized** for security and privacy reasons,
following OpenTelemetry best practices.

### 1. Trust Boundary Crossings

#### The Problem

The [W3C Baggage specification §4.1](https://www.w3.org/TR/baggage/#security-considerations)
explicitly warns:

> Application owners should either ensure that no proprietary or confidential
> information is stored in baggage, or they should ensure that baggage isn't
> present in requests that cross trust-boundaries.

The [OTel Baggage concepts documentation](https://opentelemetry.io/docs/concepts/signals/baggage/)
reinforces this:

> Sensitive Baggage items can be shared with unintended resources, like
> third-party APIs. This is because automatic instrumentation includes Baggage
> in most of your service's network requests.

#### When `gen_ai.conversation.id` Crosses Trust Boundaries

| Scenario | Risk | Recommended Action |
|----------|------|--------------------|
| Internal service → internal service | Low | Propagate normally |
| Internal service → third-party LLM API (OpenAI, Anthropic) | **Medium** | Clear baggage before outbound call |
| Internal service → public MCP server | **High** | Reject incoming baggage; do not propagate `gen_ai.conversation.id` |
| Public-facing gateway → internal services | **High** | Validate and sanitize incoming baggage |
| Multi-tenant service receiving baggage from tenant A | **High** | Never trust incoming `gen_ai.conversation.id`; assign server-side |

#### OTel Baggage API: Clear Before Untrusted Calls

The OTel Baggage API specification mandates:

> To avoid sending any name/value pairs to an untrusted process, the Baggage
> API MUST provide a way to remove all baggage entries from a context.

Implementation:

```python
from opentelemetry import context, baggage

def call_third_party_llm(prompt: str) -> str:
    """Call external LLM API with baggage cleared."""
    # Create a clean context with no baggage
    clean_ctx = baggage.clear(context.get_current())

    # Or selectively remove session attributes
    clean_ctx = baggage.remove_baggage("gen_ai.conversation.id", context.get_current())
    clean_ctx = baggage.remove_baggage("enduser.id", clean_ctx)

    # Make the call with the clean context
    with context.use_context(clean_ctx):
        return openai_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
```

### 2. PII in Session Attributes

#### The Problem

The [OTel Handling Sensitive Data guide](https://opentelemetry.io/docs/security/handling-sensitive-data/)
states:

> As the implementer, you are responsible for ensuring compliance with
> applicable privacy laws and regulations [and] protecting sensitive
> information in your telemetry data.

`gen_ai.conversation.id` and `enduser.id` may constitute Personally Identifiable
Information (PII) under GDPR, CCPA, and other regulations, especially when:

- `gen_ai.conversation.id` contains user-derived data (e.g., email hashes, JWT claims)
- `enduser.id` is a real user identifier (email, username)
- These values are stored in telemetry backends with long retention periods

#### Data Minimization Recommendations

Following OTel's [data minimization principle](https://opentelemetry.io/docs/security/handling-sensitive-data/#data-minimization):

| Recommendation | Implementation |
|----------------|----------------|
| Use opaque session IDs | Generate UUIDs, not user-derived values |
| Hash `enduser.id` before emission | Use SHA-256 of actual user ID |
| Do not propagate when unnecessary | Set `propagate_via_baggage=False` for local-only sessions |
| Use Collector processors to redact | Configure `redaction` processor for telemetry pipeline |

#### Collector-Side Redaction

```yaml
# OTel Collector configuration for sensitive data handling
processors:
  # Hash user.id to prevent PII in telemetry storage
  transform/hash-user:
    trace_statements:
      - context: span
        statements:
          - set(attributes["enduser.hash"],
                SHA256(attributes["enduser.id"]))
          - delete_key(attributes, "enduser.id")

  # Redact gen_ai.conversation.id entirely for external-facing exports
  redaction/external:
    allow_all_keys: false
    allowed_keys:
      - gen_ai.system
      - gen_ai.operation.name
      - gen_ai.request.model
      # gen_ai.conversation.id intentionally excluded
```

### 3. Session Propagation Restriction Policies

#### The Problem

Not all services should accept or forward session context. A public-facing
MCP server should not trust `gen_ai.conversation.id` from an unknown client, as:

- A malicious client could inject arbitrary session IDs to pollute telemetry
- Cross-tenant session ID leakage in multi-tenant systems
- Session fixation attacks where an attacker pre-sets a session ID

#### Proposed `SessionRestrictionPolicy`

```python
from enum import Enum

class SessionRestrictionPolicy(Enum):
    """Policy for handling incoming session context at trust boundaries."""

    ACCEPT_ALL = "accept_all"
    # Accept all incoming session context.
    # Use ONLY in fully trusted internal environments.

    ACCEPT_NONE = "reject_all"
    # Reject all incoming session context; use local-only.
    # Recommended for public-facing services.

    ACCEPT_TRUSTED = "trusted_only"
    # Accept from trusted origins only (allowlist).
    # Recommended for semi-trusted environments (partner APIs).

    ACCEPT_BAGGAGE_ONLY = "baggage_only"
    # Accept only from OTel Baggage (not from application-level metadata).
    # Ensures session was propagated through OTel-instrumented services.
```

#### Configuration via Environment Variables

```bash
# Restriction policy for session propagation
OTEL_INSTRUMENTATION_GENAI_SESSION_POLICY=reject_all

# Trusted origins allowlist (comma-separated)
OTEL_INSTRUMENTATION_GENAI_SESSION_TRUSTED_ORIGINS=service-a.internal,service-b.internal
```

#### Implementation

```python
class SessionPropagator:
    """Handles session propagation with restriction policies."""

    def __init__(
        self,
        policy: SessionRestrictionPolicy = SessionRestrictionPolicy.ACCEPT_ALL,
        trusted_origins: set[str] | None = None,
    ):
        self.policy = policy
        self.trusted_origins = trusted_origins or set()

    def extract_session(
        self,
        carrier: dict,
        origin: str | None = None,
    ) -> SessionContext | None:
        """Extract session with policy enforcement.

        Returns None if the policy rejects the incoming session.
        """
        if self.policy == SessionRestrictionPolicy.ACCEPT_NONE:
            return None

        if self.policy == SessionRestrictionPolicy.ACCEPT_TRUSTED:
            if origin not in self.trusted_origins:
                return None

        return SessionContext.from_baggage(propagate.extract(carrier))
```

### 4. When to Disable Propagation Entirely

#### Scenarios Where `gen_ai.conversation.id` Should NOT Be Propagated

| Scenario | Reason | Action |
|----------|--------|--------|
| **Batch processing / offline jobs** | No interactive session exists | Do not set `gen_ai.conversation.id` at all |
| **Async background workers** | Worker may process items from different sessions | Clear inherited context; assign per-item if needed |
| **Calls to third-party LLM providers** | Baggage visible in HTTP headers | Clear baggage before outbound HTTP call |
| **Shared/pooled connections** | Session leaks between unrelated requests | Clear context on connection reuse |
| **Serverless functions (Lambda, Cloud Functions)** | Execution context reuse across invocations | Always set fresh context per invocation |
| **Load testing / synthetic traffic** | Synthetic sessions pollute production metrics | Use distinct `gen_ai.conversation.id` prefix or exclude via sampling |

#### Implementation Pattern: Disable Baggage Propagation

```python
from opentelemetry.util.genai.session import session_scope

# Local-only session — NOT propagated via baggage
with session_scope(
    session_id="batch-job-123",
    propagate_via_baggage=False,  # ← Key: disables cross-service propagation
):
    for item in batch_items:
        process(item)
```

### 5. Baggage Header Size Limits

#### The Problem

The [W3C Baggage specification §3.3.2](https://www.w3.org/TR/baggage/#limits)
defines limits:

- Maximum **180 `list-member`s** (key=value pairs)
- Maximum **8192 bytes** total header size

If session attributes cause the baggage header to exceed these limits, platforms
MAY drop entries **without notification**.

#### Mitigation

| Risk | Mitigation |
|------|------------|
| Excessive baggage entries from multiple instrumentation layers | Limit to essential session attributes only (`gen_ai.conversation.id`, `enduser.id`) |
| Long session IDs (e.g., JWTs used as session IDs) | Use short opaque IDs; store full context server-side |
| Accumulation across deep call chains | Clear non-essential baggage at service boundaries |

### 6. Baggage Integrity and Spoofing

#### The Problem

The [W3C Baggage specification §5.1](https://www.w3.org/TR/baggage/#privacy-of-the-baggage-header)
warns:

> Applications using baggage should be aware that the keys and values can be
> propagated to other systems.

There are **no built-in integrity checks** in the W3C Baggage header. Any
intermediary can:

- Read all baggage entries (they travel as plain-text HTTP headers)
- Modify or inject arbitrary baggage entries
- Remove entries without detection

#### Mitigation for GenAI Session Propagation

| Risk | Mitigation |
|------|------------|
| Session ID spoofing by malicious client | Server-side validation: verify `gen_ai.conversation.id` exists in session store |
| Session ID injection in multi-tenant system | Server assigns `gen_ai.conversation.id`; ignore client-provided value |
| Eavesdropping on session metadata | Use TLS for all inter-service communication |
| Tampering with `enduser.id` | Verify against authentication context (OAuth token, JWT) |

#### Server-Side Validation Pattern

```python
def validate_incoming_session(
    incoming_session: SessionContext,
    auth_context: AuthContext,
) -> SessionContext:
    """Validate incoming session against server-side truth.

    Never blindly trust client-provided session attributes.
    """
    # Verify session exists in server-side session store
    if incoming_session.session_id:
        stored_session = session_store.get(incoming_session.session_id)
        if not stored_session:
            # Session doesn't exist — possible spoofing
            logger.warning(
                "Unknown gen_ai.conversation.id received: %s",
                incoming_session.session_id,
            )
            return SessionContext()  # Return empty context

        # Verify user matches authenticated identity
        if stored_session.user_id != auth_context.user_id:
            logger.warning(
                "Session user mismatch: session=%s, auth=%s",
                stored_session.user_id,
                auth_context.user_id,
            )
            return SessionContext()

    return incoming_session
```

### 7. Trace Context vs. Baggage: When to Break the Chain

#### The OTel Pattern: New Trace for Untrusted Boundaries

The OTel specification describes a pattern where services at trust boundaries
**start a new trace** rather than continuing an incoming one:

> This can be used when a Trace enters trusted boundaries of a service and
> service policy requires the generation of a new Trace rather than trusting
> the incoming Trace context.
> — [OTel Specification: Links between spans](https://opentelemetry.io/docs/specs/otel/overview/#links-between-spans)

The same pattern applies to session context:

```python
def handle_untrusted_request(request):
    """Handle request from untrusted origin."""
    # Extract incoming context for linking only
    incoming_ctx = propagate.extract(request.headers)
    incoming_span = trace.get_current_span(incoming_ctx)

    # Start a NEW trace (don't continue the incoming one)
    with tracer.start_as_current_span(
        "handle_request",
        links=[Link(incoming_span.get_span_context())],
        # No parent — this is a new trace root
    ) as span:
        # Assign server-side session (don't trust incoming)
        server_session = create_or_lookup_session(request.auth_token)
        span.set_attribute("gen_ai.conversation.id", server_session.id)

        return process_request(request)
```

### 8. Summary: Decision Matrix for Propagation

| Environment | Propagate `gen_ai.conversation.id`? | Propagate `enduser.id`? | Policy |
|-------------|------------------------|------------------------|--------|
| Internal microservices (same trust zone) | ✅ Yes, via Baggage | ✅ Yes, via Baggage | `ACCEPT_ALL` |
| Internal → third-party LLM API | ❌ No, clear baggage | ❌ No, clear baggage | Clear context |
| Internal → partner MCP server | ⚠️ Yes, if trusted | ❌ No | `ACCEPT_TRUSTED` |
| Public gateway → internal services | ⚠️ Validate, then assign | ⚠️ Derive from auth | `ACCEPT_NONE` + server-assign |
| Public MCP server (open internet) | ❌ No, reject incoming | ❌ No, reject incoming | `ACCEPT_NONE` |
| Batch / async processing | 🔧 Conditional | 🔧 Conditional | `propagate_via_baggage=False` |
| Multi-tenant platform | ✅ Server-assigned only | ✅ Server-assigned only | `ACCEPT_NONE` + server-assign |
| Development / local testing | ✅ Yes | ✅ Yes | `ACCEPT_ALL` |

---

## Comparison with Other Platforms

### Session Handling Architecture Comparison

| Aspect | SDOT (this proposal) | Traceloop/OpenLLMetry | Arize-AI/Phoenix | Langfuse |
|--------|-----|----------------------|------------------|---------|
| **Turn model** | 1 turn = 1 trace | 1 turn = 1 trace (workflow decorator) | 1 turn = 1 trace | 1 turn = 1 trace |
| **Session storage** | GenAI types + ContextVar | ContextVar (association_properties) | ContextVar (using_session) | OTel Baggage |
| **Session attribute** | `gen_ai.conversation.id` (standard upstream convention) | `traceloop.association.properties.session_id` | `session.id` (OpenInference) | `langfuse.session.id` |
| **User propagation** | `enduser.id` via Baggage | `traceloop.association.properties.user_id` | `user.id` via ContextVar | `langfuse.user.id` |
| **Custom association attrs** | `genai.association.*` via Baggage | `traceloop.association.properties.*` | `using_metadata({...})` | Custom span attributes |
| **Child propagation** | Handler context stack | SpanProcessor.on_start | OTel context + SpanProcessor | BaggageSpanProcessor |
| **Cross-RPC mechanism** | OTel Baggage (standard propagation) | TraceContext propagator | TraceContext propagator | W3C Baggage header |
| **Cross-RPC association attrs** | ✅ Via Baggage | ❌ Not propagated | ❌ Not propagated | ❌ Not propagated |
| **Restriction/Security** | SessionRestrictionPolicy (env var) | Content allow lists | N/A | `as_baggage=False` flag |
| **Trust boundary handling** | Policy-based (4 levels) | N/A | N/A | Binary (on/off) |

### Key Architectural Insights from Research

From [ARCHITECTURE.comparison.md](docs/research/ARCHITECTURE.comparison.md):

1. **Traceloop** uses a SpanProcessor that copies `association_properties` to every
   span at creation time — simple but no trust boundary controls
2. **Phoenix** leverages `using_session()` context manager with ContextVar — similar
   to our `session_scope()` approach
3. **Langfuse** is the most OTel-native, using `BaggageSpanProcessor` for automatic
   attribute propagation, with `as_baggage=False` to opt out of cross-service propagation
4. **All platforms** converge on a session/conversation attribute (with vendor prefixes)
5. **All platforms** implement the turn = trace model (one trace per user interaction)
6. **Only SDOT** proposes cross-service propagation of custom association attributes via Baggage

---

## Implementation Roadmap

### Phase 1: Core Session & Association API *(mostly complete in SDOT)*

- [x] `gen_ai.conversation.id`, `user.id`, `customer.id` in GenAI base type (`types.py`)
- [x] `semantic_convention_attributes()` auto-emission
- [x] `SessionContext` dataclass in `handler.py`
- [ ] `set_session()`, `get_session()`, `session_scope()` public API
- [ ] `set_association_properties()` — Traceloop-compatible custom attribute API
- [ ] `association_properties` field on `GenAI` base type
- [ ] Handler auto-population from ContextVar (session + user + association attrs)

### Phase 2: User & Association Propagation *(partially complete)*

- [x] MCP trace context propagation via standard OTel propagation API (FastMCP instrumentor)
- [ ] `enduser.id` propagation via OTel Baggage
- [ ] `genai.association.*` custom attributes propagation via Baggage
- [ ] Baggage propagation alongside `traceparent`
- [ ] `SessionRestrictionPolicy` for public servers
- [ ] Environment variable configuration

### Phase 3: Upstream Contribution

- [ ] Comment on [#2883](https://github.com/open-telemetry/semantic-conventions/issues/2883) with concrete scenarios (incl. turn=trace model)
- [ ] Align with [#3418](https://github.com/open-telemetry/semantic-conventions/issues/3418) entry span proposal
- [ ] PR to add `gen_ai.conversation.id` as RECOMMENDED on GenAI spans
- [ ] Propose `genai.association.*` prefix for custom association attributes
- [ ] Document security edge cases in semconv (this proposal's §Edge Cases)

### Phase 4: Framework Integration

- [ ] LangChain: Extract from `config.metadata.session_id` → `gen_ai.conversation.id`
- [ ] LangChain: Extract from `config.metadata` → association properties
- [ ] CrewAI: Extract from crew/task context
- [ ] OpenAI Agents: Extract from `thread_id` → `gen_ai.conversation.id`
- [ ] LlamaIndex: Extract from service context

---

## Open Questions for Community

| # | Question | SDOT Recommendation | SIG Status |
|---|----------|---------------------|------------|
| 1 | Should `gen_ai.conversation.id` be RECOMMENDED on GenAI spans? | Yes, as the standard session/conversation grouping attribute | Not yet discussed |
| 2 | Should `gen_ai.conversation.id` coexist with `session.id` on the same span? | Optional — `session.id` can be added for correlation with non-GenAI telemetry | lmolkova leans toward documenting, not mandating |
| 3 | Should the "turn = trace" model be documented in GenAI semconv? | Yes, as a RECOMMENDED practice | Not yet discussed |
| 4 | Should custom association attributes use `genai.association.*` prefix? | Yes, for vendor-neutral interop with Traceloop-style patterns | Not yet discussed |
| 5 | Should `enduser.id` propagation via Baggage be RECOMMENDED? | Yes, with security caveats from W3C §4 | Not yet discussed |
| 6 | Should GenAI semconv define security guidance for baggage propagation? | Yes, reference W3C §4 and OTel sensitive data guide | Not yet discussed |
| 7 | Should the entry span proposal ([#3418](https://github.com/open-telemetry/semantic-conventions/issues/3418)) be the vehicle for `gen_ai.conversation.id`? | Yes, natural fit | Needs alignment discussion |
| 8 | Should restriction policies be standardized or left to implementations? | Standardize env var naming; leave policy logic to SDKs | Not yet discussed |
| 9 | Default policy for public services? | `ACCEPT_NONE` (reject incoming session context) | Not yet discussed |

---

## References

### SDOT Documentation

| Document | Description |
|----------|-------------|
| [session-instrumentation-overview.md](docs/session-instrumentation-overview.md) | Comparison of session handling across Galileo, Langfuse, Traceloop, Phoenix, AGNTCY |
| [feat-session-support.md](docs/feat-session-support.md) | Session support feature plan with implementation options |
| [ARCHITECTURE.comparison.md](docs/research/ARCHITECTURE.comparison.md) | Deep architecture comparison of Traceloop, Phoenix, Langfuse |
| [upstream-proposal.md](docs/upstream-proposal.md) | Original upstream proposal (superseded by this document) |

### OpenTelemetry Specifications

| Document | Relevance |
|----------|-----------|
| [OTel Baggage API](https://opentelemetry.io/docs/specs/otel/baggage/api/) | Stable spec for baggage operations, including `Clear Baggage in the Context` |
| [OTel Propagators API](https://opentelemetry.io/docs/specs/otel/context/api-propagators/) | TextMapPropagator for cross-service context propagation |
| [W3C Baggage Specification](https://www.w3.org/TR/baggage/) | Wire format for baggage header, including §4 Security and §5 Privacy |
| [OTel Handling Sensitive Data](https://opentelemetry.io/docs/security/handling-sensitive-data/) | Data minimization, Collector-side redaction, hashing guidance |
| [OTel GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai) | Current GenAI semconv (Development status) |

### Community Issues

| Issue | Title |
|-------|-------|
| [semantic-conventions#2883](https://github.com/open-telemetry/semantic-conventions/issues/2883) | Add `session.id` attribute to GenAI semantic conventions |
| [semantic-conventions#1872](https://github.com/open-telemetry/semantic-conventions/issues/1872) | Add Gen AI user and session Semantic Conventions |
| [semantic-conventions#3418](https://github.com/open-telemetry/semantic-conventions/issues/3418) | Enhancing OTel GenAI Semantic Conventions with Entry Span |
| [semantic-conventions#2594](https://github.com/open-telemetry/semantic-conventions/pull/2594) | (Closed PR) Add session.id to GenAI spans |
