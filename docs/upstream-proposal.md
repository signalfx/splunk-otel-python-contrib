Upstream Community Proposal: GenAI Session Propagation for OpenTelemetry
Executive Summary
This proposal outlines a standardized approach for session/conversation tracking in GenAI observability, based on the implementation in Splunk Distro for OpenTelemetry (SDOT). The goal is to establish community consensus on:
Session attribute semantics (session.id, user.id, customer.id)
Child span propagation via OTel Context/Baggage
Cross-RPC propagation for MCP and other agent protocols
Restriction mechanisms for public services

Current SDOT Implementation Summary
1. Core GenAI Types with Session Support
Session fields are part of the GenAI base dataclass, automatically propagated to all invocation types:
# util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py
@dataclass(kw_only=True)
class GenAI:
    """Base type for all GenAI telemetry entities."""
    # ... existing fields ...
    
    # Session/User Context (association properties for session tracking)
    session_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "session.id"},
    )
    user_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "user.id"},
    )
    customer_id: Optional[str] = field(
        default=None,
        metadata={"semconv": "customer.id"},
    )

Key Design Decisions:
Uses session.id (generic) rather than gen_ai.session.id (GenAI-specific) for broader applicability
Leverages semantic_convention_attributes() method for automatic attribute emission
Session fields flow through all GenAI types: LLMInvocation, AgentInvocation, Workflow, ToolCall, EmbeddingInvocation
2. Agent Context Stack for Child Span Propagation
The TelemetryHandler maintains an agent context stack for implicit propagation:
# util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py
class TelemetryHandler:
    # Active agent identity stack (name, id) for implicit propagation to nested operations
    _agent_context_stack: list[tuple[str, str]] = []
    
    def start_agent(self, agent: AgentInvocation) -> AgentInvocation:
        # Push agent identity context
        if isinstance(agent, AgentInvocation) and agent.name:
            self._agent_context_stack.append((agent.name, str(agent.run_id)))
        # ...
    
    def start_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        # Implicit agent inheritance
        if (not invocation.agent_name or not invocation.agent_id) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id

Benefits:
Nested LLM calls automatically inherit parent agent context
Works with Python's async/await via contextvars
No manual threading of context required
3. MCP Cross-RPC Propagation via Standard OTel Propagation
SDOT implements trace context propagation for MCP using the standard OTel
Propagation API (`propagate.inject()` / `propagate.extract()`), with the MCP
`params._meta` object serving as the carrier (analogous to HTTP headers):

# transport_instrumentor.py — uses standard OTel propagation

# Client side: inject into _meta carrier
carrier = {}
propagate.inject(carrier)  # Standard API — injects traceparent, tracestate, baggage
for key, value in carrier.items():
    setattr(params.meta, key, value)  # _meta is the MCP carrier

# Server side: extract from _meta carrier
carrier = {key: getattr(request_meta, key) for key in ("traceparent", "tracestate", "baggage")}
ctx = propagate.extract(carrier)  # Standard API

Transport-level Integration:
# transport_instrumentor.py - Client side
async def traced_send_request(wrapped, instance, args, kwargs):
    # Inject traceparent/tracestate into request.params.meta
    carrier = {}
    propagate.inject(carrier)
    for key, value in carrier.items():
        setattr(params.meta, key, value)
    return await wrapped(*args, **kwargs)

# Server side  
async def traced_handle_request(wrapped, instance, args, kwargs):
    # Extract from message.request_meta
    carrier = {"traceparent": getattr(request_meta, "traceparent", None)}
    ctx = propagate.extract(carrier)
    token = context.attach(ctx)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        context.detach(token)


Upstream Proposal: Session Propagation via OTel Baggage
Approach: Standard OTel Baggage Propagation
MCP uses `params._meta` as its metadata carrier (equivalent to HTTP headers).
The standard OTel `propagate.inject()` call handles everything — trace context
AND baggage — through the same mechanism. Requirements:
Propagate session.id via OTel Baggage alongside traceparent/tracestate
Allow servers to restrict session propagation from untrusted clients
Maintain compatibility with existing MCP implementations
Solution: Standard OTel Baggage
1. OTel Baggage for Session Context
Use OTel Baggage to carry session context alongside trace context:
from opentelemetry import baggage
from opentelemetry.propagate import inject, extract

# Client: Set session in baggage before MCP call
ctx = baggage.set_baggage("session.id", "session-123", context=current_ctx)
ctx = baggage.set_baggage("user.id", "user-456", context=ctx)

# Inject both trace context AND baggage into _meta
carrier = {}
inject(carrier, context=ctx)
# carrier now contains: {
#   "traceparent": "00-...",
#   "tracestate": "...",
#   "baggage": "session.id=session-123,user.id=user-456"
# }

2. MCP Carrier Structure
The MCP `_meta` carrier carries all OTel-propagated context:
# MCP request with full context propagation
{
    "method": "tools/call",
    "params": {
        "name": "search",
        "arguments": {...},
        "_meta": {
            "traceparent": "00-abc123...",
            "tracestate": "vendor=splunk",
            "baggage": "session.id=conv-123,user.id=user-456"
        }
    }
}

3. Server-Side Extraction
def extract_mcp_context(params: dict) -> tuple[Context, dict[str, str]]:
    """Extract trace context and session attributes from MCP _meta."""
    meta = params.get("_meta", {})
    
    # Extract trace context (traceparent, tracestate)
    trace_ctx = propagate.extract(meta)
    
    # Extract session from baggage
    session_attrs = {}
    if "baggage" in meta:
        # Parse baggage header (key=value pairs)
        baggage_ctx = extract({"baggage": meta["baggage"]})
        session_attrs = {
            "session.id": baggage.get_baggage("session.id", baggage_ctx),
            "user.id": baggage.get_baggage("user.id", baggage_ctx),
        }
    
    return trace_ctx, {k: v for k, v in session_attrs.items() if v}

Proposed SessionPropagator Component
New reusable component for GenAI frameworks:
# Proposed: opentelemetry-util-genai/src/opentelemetry/util/genai/session.py

from contextvars import ContextVar
from dataclasses import dataclass
from opentelemetry import baggage, context, propagate
from opentelemetry.context import Context

@dataclass
class SessionContext:
    """Immutable session context container."""
    session_id: str | None = None
    user_id: str | None = None
    customer_id: str | None = None
    
    def to_baggage(self, ctx: Context | None = None) -> Context:
        """Convert to OTel baggage context."""
        ctx = ctx or context.get_current()
        if self.session_id:
            ctx = baggage.set_baggage("session.id", self.session_id, ctx)
        if self.user_id:
            ctx = baggage.set_baggage("user.id", self.user_id, ctx)
        if self.customer_id:
            ctx = baggage.set_baggage("customer.id", self.customer_id, ctx)
        return ctx
    
    @classmethod
    def from_baggage(cls, ctx: Context | None = None) -> "SessionContext":
        """Extract from OTel baggage context."""
        ctx = ctx or context.get_current()
        return cls(
            session_id=baggage.get_baggage("session.id", ctx),
            user_id=baggage.get_baggage("user.id", ctx),
            customer_id=baggage.get_baggage("customer.id", ctx),
        )

# Thread-local session context (ContextVar for async safety)
_session_context: ContextVar[SessionContext] = ContextVar("genai_session", default=SessionContext())

def set_session(
    session_id: str | None = None,
    user_id: str | None = None,
    customer_id: str | None = None,
    propagate_via_baggage: bool = True,
) -> object:
    """Set session context for current execution scope.
    
    Args:
        session_id: Session/conversation identifier
        user_id: User identifier
        customer_id: Customer/tenant identifier
        propagate_via_baggage: Whether to also set OTel baggage (default: True)
    
    Returns:
        Token for resetting context (use in finally block)
    """
    session = SessionContext(session_id, user_id, customer_id)
    token = _session_context.set(session)
    
    if propagate_via_baggage:
        # Also attach to OTel context for cross-service propagation
        ctx = session.to_baggage()
        context.attach(ctx)
    
    return token

def get_session() -> SessionContext:
    """Get current session context."""
    return _session_context.get()

def clear_session(token: object = None) -> None:
    """Clear session context."""
    if token:
        _session_context.reset(token)
    else:
        _session_context.set(SessionContext())

# Context manager for scoped sessions
from contextlib import contextmanager

@contextmanager
def session_scope(
    session_id: str | None = None,
    user_id: str | None = None,
    propagate_via_baggage: bool = True,
):
    """Context manager for scoped session tracking.
    
    Usage:
        with session_scope(session_id="conv-123"):
            # All GenAI operations here inherit session context
            result = chain.invoke(...)
    """
    token = set_session(session_id, user_id, propagate_via_baggage=propagate_via_baggage)
    try:
        yield get_session()
    finally:
        clear_session(token)

TelemetryHandler Integration
# handler.py modifications
class TelemetryHandler:
    def start_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        # Auto-populate from session context if not explicitly set
        session = get_session()
        if not invocation.session_id and session.session_id:
            invocation.session_id = session.session_id
        if not invocation.user_id and session.user_id:
            invocation.user_id = session.user_id
        # ... rest of start_llm


Restriction Mechanism for Public Services
When running public MCP servers, restrict propagation of untrusted session context:
Proposed: SessionRestrictionPolicy
# Proposed: session.py extension

from enum import Enum

class SessionRestrictionPolicy(Enum):
    ACCEPT_ALL = "accept_all"        # Accept all incoming session context
    ACCEPT_NONE = "reject_all"       # Reject all incoming, use local only
    ACCEPT_TRUSTED = "trusted_only"  # Accept from trusted origins only
    BAGGAGE_ONLY = "baggage"         # Accept only via OTel Baggage propagation

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
        """Extract session with policy enforcement."""
        if self.policy == SessionRestrictionPolicy.ACCEPT_NONE:
            return None
        
        if self.policy == SessionRestrictionPolicy.ACCEPT_TRUSTED:
            if origin not in self.trusted_origins:
                return None
        
        # Extract from baggage
        return SessionContext.from_baggage(propagate.extract(carrier))
    
    def inject_session(
        self,
        session: SessionContext,
        carrier: dict,
    ) -> dict:
        """Inject session into carrier for outgoing requests."""
        ctx = session.to_baggage()
        propagate.inject(carrier, context=ctx)
        return carrier

# Environment variable configuration
# OTEL_INSTRUMENTATION_GENAI_SESSION_POLICY=accept_all|reject_all|trusted_only|baggage
# OTEL_INSTRUMENTATION_GENAI_SESSION_TRUSTED_ORIGINS=service1.internal,service2.internal


Proposed Semantic Conventions
New Attributes
Attribute
Type
Description
Example
session.id
string
Session/conversation identifier
"conv-abc123"
user.id
string
User identifier
"user-456"
customer.id
string
Customer/tenant identifier
"customer-789"

Relationship to Existing Conventions
Existing Attribute
Relationship
gen_ai.conversation.id
Alias for session.id in GenAI context
mcp.session.id
MCP protocol-specific session (different from GenAI session)
gen_ai.agent.id
Agent instance ID (per-invocation, not per-session)


Comparison with Other Platforms
Platform
Session Storage
Child Propagation
Cross-RPC
Restriction
SDOT (this proposal)
GenAI types + ContextVar
Handler context stack
OTel Baggage (standard propagation)
Policy-based
Traceloop
ContextVar
SpanProcessor.on_start
TraceContext propagator
Content allow lists
Langfuse
OTel Baggage
SpanProcessor
Baggage header
as_baggage=False
Phoenix
ContextVar
OTel context
TraceContext
N/A
Galileo
ContextVar
SpanProcessor
HTTP middleware
Per-service env


Implementation Roadmap
Phase 1: Core Session API (Ready for upstream)
[x] session.id, user.id, customer.id in GenAI base type
[x] semantic_convention_attributes() auto-emission
[ ] SessionContext dataclass
[ ] set_session(), get_session(), session_scope() API
[ ] Handler auto-population from context
Phase 2: MCP Integration
[x] Trace context propagation via standard OTel API (FastMCP instrumentor)
[ ] Baggage propagation alongside traceparent
[ ] SessionRestrictionPolicy for public servers
Phase 3: Framework Integration
[ ] LangChain: Extract from config.metadata.session_id
[ ] CrewAI: Extract from crew/task context
[ ] OpenAI Agents: Extract from thread_id

Open Questions for Community
Attribute naming: Should session.id be in root namespace or gen_ai.session.id?
Recommendation: session.id for broader applicability
Baggage vs custom header: Should session propagate via standard OTel baggage or custom header?
Recommendation: OTel baggage for interoperability, but support custom for legacy
MCP protocol standardization: Should session propagation be proposed to MCP spec?
Recommendation: Yes, standardize OTel Baggage support in _meta carrier
Restriction defaults: Should public-facing servers restrict by default?
Recommendation: ACCEPT_ALL for development, ACCEPT_TRUSTED for production

References
SDOT session-instrumentation-overview.md - Platform comparison
SDOT feat-session-support.md - Original feature plan
OpenTelemetry Baggage API
MCP Protocol Specification
OpenTelemetry GenAI Semantic Conventions

