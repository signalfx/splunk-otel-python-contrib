# Session Instrumentation Overview

This document provides a comprehensive comparison of session instrumentation support across major LLM observability platforms: **Galileo**, **Langfuse**, **Traceloop**, **Arize AI (Phoenix)**, and **AGNTCY (Outshift)**.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Platform Comparison Matrix](#platform-comparison-matrix)
3. [Galileo](#galileo)
4. [Langfuse](#langfuse)
5. [Traceloop (OpenLLMetry)](#traceloop-openllmetry)
6. [Arize AI (Phoenix)](#arize-ai-phoenix)
7. [AGNTCY (Outshift)](#agntcy-outshift)
8. [Cross-RPC Session Propagation](#cross-rpc-session-propagation)
9. [Evaluation Capabilities](#evaluation-capabilities)
10. [Recommendations](#recommendations)

---

## Executive Summary

| Aspect | Galileo | Langfuse | Traceloop | Arize Phoenix | AGNTCY |
|--------|---------|----------|-----------|---------------|--------|
| **Session ID Attribute** | `galileo.session.id` | `langfuse.session.id` or `session.id` | Custom association | `session.id` | HTTP headers |
| **Child Span Propagation** | GalileoContext | OTel Baggage | Traceloop SDK | OTel Context | Manual HTTP |
| **Cross-RPC Support** | Limited | OTel Baggage | Trace context | OTel propagators | SLIM/A2A/MCP |
| **Session-Level Evals** | ✅ Yes | ✅ Yes | ❌ Limited | ✅ Yes | ❌ No |

---

## Platform Comparison Matrix

### Session Instrumentation Support

| Feature | Galileo | Langfuse | Traceloop | Phoenix | AGNTCY |
|---------|---------|----------|-----------|---------|--------|
| Native SDK session management | ✅ | ✅ | ✅ | ✅ | ✅ |
| OTel-compatible session attributes | ⚠️ Custom | ✅ | ✅ | ✅ | ✅ |
| Automatic child span inheritance | ✅ | ⚠️ Manual | ✅ | ✅ | ⚠️ Manual |
| Session UI view | ✅ | ✅ | ✅ | ✅ | ⚠️ ClickHouse |
| Session-level metrics | ✅ | ✅ | ⚠️ Limited | ✅ | ⚠️ Limited |

---

## Galileo

### Session Management

Galileo uses a hierarchical structure: **Log Streams → Sessions → Traces → Spans**

#### SDK Session APIs

```python
from galileo import GalileoLogger

# Initialize logger
logger = GalileoLogger(project="my-project")

# Start a new session
session = logger.start_session(
    name="customer-chat-session",
    external_id="user-123-session-456"  # Optional: your own session ID
)

# Continue an existing session
logger.set_session(session_id="existing-session-id")

# Clear session context
logger.clear_session()
```

#### Context Manager Pattern

```python
from galileo import galileo_context

# Using context manager
with galileo_context.start_session(name="conversation"):
    # All traces created here belong to this session
    with logger.start_trace(input="Hello"):
        response = call_llm("Hello")
        logger.log(output=response)
```

### Child Span Propagation

Galileo automatically propagates session context through `GalileoContext`:

```python
from galileo import galileo_context

galileo_context.set_session(session_id)
# All subsequent traces/spans inherit the session_id
```

**Limitation**: Galileo's context is not OTel-native; it uses its own context propagation mechanism.

### Third-Party Integration Auto-Sessions

- **OpenAI Wrapper**: Auto-creates sessions per conversation
- **LangChain Callback**: Groups traces by callback handler lifecycle
- **OpenAI Agents Trace Processor**: Associates spans with agent sessions

### Documentation Links

- [Sessions Overview](https://v2docs.galileo.ai/concepts/logging/sessions/sessions-overview)
- [GalileoLogger API](https://v2docs.galileo.ai/sdk-api/logging/galileo-logger)
- [GalileoContext API](https://v2docs.galileo.ai/sdk-api/logging/galileo-context)

---

## Langfuse

### Session Management

Langfuse provides flexible session support through span attributes that work with both native SDKs and OpenTelemetry.

#### Native SDK Session Propagation

```python
from langfuse import observe, propagate_attributes

@observe()
def process_request():
    # Propagate session_id to all child observations
    with propagate_attributes(session_id="your-session-id"):
        # All nested observations automatically inherit session_id
        result = process_chat_message()
        return result
```

#### Direct Span Creation

```python
from langfuse import get_client, propagate_attributes

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="process-chat-message"
) as root_span:
    # Propagate session_id to all child observations
    with propagate_attributes(session_id="chat-session-123"):
        with root_span.start_as_current_observation(
            as_type="generation",
            name="generate-response",
            model="gpt-4o"
        ) as gen:
            # This generation automatically has session_id
            pass
```

### OpenTelemetry Session Attributes

Langfuse accepts session information via span attributes:

| Attribute | Description |
|-----------|-------------|
| `langfuse.session.id` | Primary session identifier |
| `session.id` | Alternative (OpenInference compatible) |
| `langfuse.user.id` or `user.id` | User identifier |

#### Using OTel Baggage for Propagation

```python
from opentelemetry import baggage
from opentelemetry.processor.baggage import BaggageSpanProcessor

# Set baggage at trace start
ctx = baggage.set_baggage("langfuse.session.id", "session-123")

# BaggageSpanProcessor automatically copies to span attributes
```

> ⚠️ **Security Note**: Baggage is propagated across service boundaries. Do not include sensitive data.

### Attribute Mapping for OTLP Ingestion

When sending via OpenTelemetry Collector:

```yaml
processors:
  attributes/langfuse:
    actions:
      - key: langfuse.session.id
        value: "${SESSION_ID}"
        action: upsert
      - key: langfuse.user.id
        value: "${USER_ID}"
        action: upsert
```

### Documentation Links

- [Sessions](https://langfuse.com/docs/observability/features/sessions)
- [OpenTelemetry Integration](https://langfuse.com/docs/integrations/opentelemetry)
- [Attribute Mapping](https://langfuse.com/integrations/native/opentelemetry#property-mapping)

---

## Traceloop (OpenLLMetry)

### Entity Association

Traceloop uses **association properties** to link traces with application entities:

```python
from traceloop.sdk import Traceloop

# Associate current trace with entities
Traceloop.set_association_properties({
    "user_id": "user12345",
    "chat_id": "chat12345",
    "session_id": "session-789"  # Custom key
})
```

### Workflow Annotations

Traceloop provides decorators for structuring traces:

```python
from traceloop.sdk.decorators import workflow, task, agent, tool

@workflow(name="conversation_flow")
def handle_conversation():
    response = create_response()
    return process_response(response)

@task(name="create_response")
def create_response():
    return llm.complete("...")

@agent(name="support_agent")
def support_agent(query: str):
    return agent.run(query)

@tool(name="search_tool")
def search_tool(query: str):
    return search_engine.search(query)
```

### Session Propagation

Association properties are automatically propagated to child spans within the same trace context:

```python
from traceloop.sdk import Traceloop

Traceloop.set_association_properties({"session_id": "abc123"})

@workflow(name="my_workflow")
def my_workflow():
    # All child tasks/agents inherit association properties
    result = child_task()
    return result
```

### Documentation Links

- [Association Properties](https://www.traceloop.com/docs/openllmetry/tracing/association)
- [Workflow Annotations](https://www.traceloop.com/docs/openllmetry/tracing/annotations)

---

## Arize AI (Phoenix)

### Session Management with OpenInference

Phoenix uses the **OpenInference** instrumentation library with context managers:

```python
from openinference.instrumentation import using_session, using_user

# Set session for all child spans
with using_session(session_id="my-session-123"):
    with using_user(user_id="user-456"):
        # All spans created here have session.id and user.id
        response = llm.complete("Hello")
```

### Context Managers

```python
from openinference.instrumentation import (
    using_session,
    using_user,
    using_metadata,
    using_tags,
    using_attributes,
)

# Combine multiple context managers
with using_session("session-123"):
    with using_user("user-456"):
        with using_metadata({"environment": "production"}):
            with using_tags(["important", "customer-facing"]):
                # Rich context on all child spans
                result = agent.run(query)
```

### Automatic Propagation

OpenInference auto-instrumentors read context and propagate attributes:

```python
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

# Instrument once at startup
OpenAIInstrumentor().instrument()
LangChainInstrumentor().instrument()

# Context is automatically propagated
with using_session("session-123"):
    # OpenAI and LangChain calls get session.id attribute
    openai.chat.completions.create(...)
```

### LangChain-Specific Session Handling

LangChain metadata keys are automatically mapped:

| LangChain Metadata Key | Phoenix Attribute |
|------------------------|-------------------|
| `session_id` | `session.id` |
| `thread_id` | `session.id` |
| `conversation_id` | `session.id` |

```python
# LangChain with session
chain.invoke(
    {"input": "Hello"},
    config={"metadata": {"session_id": "my-session"}}
)
```

### Session View in UI

Phoenix provides:
- **Session list**: View all sessions with metrics
- **Conversation replay**: Step through session messages
- **Session-level metrics**: Aggregate stats per session

### Documentation Links

- [Setup Sessions](https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-sessions)
- [Customize Spans](https://arize.com/docs/phoenix/tracing/how-to-tracing/add-metadata/customize-spans)
- [OpenInference Package](https://pypi.org/project/openinference-instrumentation/)

---

## AGNTCY (Outshift)

### Overview

AGNTCY provides the **IOA Observe SDK** for multi-agent system observability, with built-in session management and cross-agent communication support.

### Session Management

```python
from ioa_observe.sdk import Observe
from ioa_observe.sdk.tracing import session_start

# Initialize
Observe.init(
    app_name="your_app_name",
    api_endpoint="http://localhost:4318"
)

# Start session as context manager
with session_start() as session_id:
    result = my_agent_function({"input": "data"})
```

### Decorators for Agents and Graphs

```python
from ioa_observe.sdk.decorators import agent, graph, tool

@graph(name="multi_agent_graph")
def build_graph():
    """Captures MAS topology for observability"""
    return StateGraph(GraphState).compile()

@agent(name="processing_agent", description="Processes user input")
def processing_node(state):
    return {"messages": [response]}

@tool(name="search", description="Search tool")
def search_tool(query: str):
    return results
```

### HTTP Session Propagation

AGNTCY supports manual session propagation via HTTP headers:

```python
from ioa_observe.sdk.tracing import session_start
from ioa_observe.sdk.tracing.context_utils import set_context_from_headers

# Client: Include session in headers
def send_http_request(payload: dict, session_id: dict):
    headers = session_id or {}
    headers["Content-Type"] = "application/json"
    response = requests.post(endpoint, json=payload, headers=headers)
    return response.json()

# Server: Extract session from headers
@app.route("/runs", methods=["POST"])
def process_message():
    headers = request.headers
    if headers:
        set_context_from_headers(headers)
    # Process with session context
```

### Protocol Support

AGNTCY provides instrumentors for cross-agent protocols:

```python
# SLIM (Secure Low-Latency Interactive Messaging)
from ioa_observe.sdk.instrumentations.slim import SLIMInstrumentor
SLIMInstrumentor().instrument()

# A2A (Agent-to-Agent Protocol - Google)
from ioa_observe.sdk.instrumentations.a2a import A2AInstrumentor
A2AInstrumentor().instrument()

# MCP (Model Context Protocol)
from ioa_observe.sdk.instrumentations.mcp import McpInstrumentor
McpInstrumentor().instrument()
```

### Documentation Links

- [AGNTCY Observe SDK](https://github.com/agntcy/observe)
- [Getting Started Guide](https://github.com/agntcy/observe/blob/main/GETTING-STARTED.md)
- [AGNTCY Documentation](https://docs.agntcy.org/)

---

## Cross-RPC Session Propagation

### Comparison of Propagation Mechanisms

| Platform | Mechanism | Cross-Service | Cross-Protocol |
|----------|-----------|---------------|----------------|
| Galileo | GalileoContext | ❌ | ❌ |
| Langfuse | OTel Baggage | ✅ | ⚠️ OTel only |
| Traceloop | OTel Context | ✅ | ⚠️ OTel only |
| Phoenix | OTel Context | ✅ | ⚠️ OTel only |
| AGNTCY | HTTP Headers + SLIM/A2A/MCP | ✅ | ✅ |

### OpenTelemetry Baggage Pattern

For OTel-based platforms (Langfuse, Traceloop, Phoenix):

```python
from opentelemetry import baggage
from opentelemetry.context import get_current
from opentelemetry.propagate import inject, extract

# Set baggage before making RPC call
ctx = baggage.set_baggage("session.id", "session-123")

# Inject into HTTP headers
headers = {}
inject(headers, context=ctx)

# On receiving side
ctx = extract(headers)
session_id = baggage.get_baggage("session.id", ctx)
```

### MCP Protocol Considerations

For MCP (Model Context Protocol) servers:

```python
# Currently, MCP doesn't have native session propagation
# Use custom metadata in tool calls

@mcp_tool
async def my_tool(args: dict, context: MCPContext):
    session_id = context.metadata.get("session_id")
    # Propagate to child spans manually
```

### A2A Protocol Considerations

Google's Agent-to-Agent protocol includes task context that can carry session information:

```python
# A2A task payload can include session metadata
task_payload = {
    "task_id": "task-123",
    "metadata": {
        "session_id": "session-456"
    }
}
```

---

## Evaluation Capabilities

### Session vs Trace Level Evaluations

| Platform | Trace-Level Evals | Session-Level Evals | Execution Timing |
|----------|-------------------|---------------------|------------------|
| Galileo | ✅ | ✅ | Async (Log streams) + Sync (Experiments) |
| Langfuse | ✅ | ⚠️ Via annotations | Async (LLM-as-Judge) + Manual |
| Traceloop | ⚠️ Limited | ❌ | Platform-specific |
| Phoenix | ✅ | ✅ | Async + Interactive |
| AGNTCY | ❌ | ❌ | N/A (observability only) |

### Galileo Evaluation Metrics

Galileo provides out-of-the-box metrics at different levels:

**Trace/Span Level:**
- Response quality (correctness, relevance)
- Agentic performance (tool selection quality)
- Safety and compliance (PII, toxicity, bias)
- Model confidence (uncertainty)

**Session Level:**
- Aggregate scores across conversation
- Session completion success
- Multi-turn coherence

```python
# Galileo metrics configuration
from galileo import configure_metrics

configure_metrics(
    metrics=["correctness", "toxicity", "tool_selection_quality"],
    apply_to="session"  # or "trace" or "span"
)
```

### Langfuse Evaluation

Langfuse supports scores at trace and observation levels:

```python
from langfuse import get_client

langfuse = get_client()

# Score a trace
langfuse.score(
    trace_id="trace-123",
    name="relevance",
    value=0.9,
    comment="Response was highly relevant"
)

# Score an observation (span)
langfuse.score(
    trace_id="trace-123",
    observation_id="span-456",
    name="hallucination",
    value=0.1
)
```

**LLM-as-a-Judge** evaluators run asynchronously on ingested traces.

### Phoenix Evaluation

Phoenix provides evaluation at trace level with session aggregation:

```python
from phoenix.evals import run_evals
from phoenix.evals.models import OpenAIModel
from phoenix.evals.templates import RAG_RELEVANCY_PROMPT_TEMPLATE

# Evaluate traces
evaluator = OpenAIModel(model="gpt-4")
results = run_evals(
    traces=traces,
    evaluators=[
        ("relevance", RAG_RELEVANCY_PROMPT_TEMPLATE, evaluator),
    ]
)
```

---

## Recommendations

### For Single-Service Applications

**Recommended**: Use platform-native SDK session management

```python
# Example with Langfuse
with propagate_attributes(session_id="session-123", user_id="user-456"):
    # Your application code
    pass
```

### For Multi-Service/Distributed Applications

**Recommended**: Use OpenTelemetry Baggage with BaggageSpanProcessor

```python
# At service entry point
from opentelemetry.processor.baggage import BaggageSpanProcessor

provider.add_span_processor(BaggageSpanProcessor())

# Set baggage for propagation
ctx = baggage.set_baggage("session.id", session_id)
```

### For Multi-Agent Systems

**Recommended**: AGNTCY IOA Observe SDK with protocol-specific instrumentors

```python
from ioa_observe.sdk import Observe
from ioa_observe.sdk.instrumentations.slim import SLIMInstrumentor
from ioa_observe.sdk.instrumentations.a2a import A2AInstrumentor

Observe.init("my-mas-app", api_endpoint="http://localhost:4318")
SLIMInstrumentor().instrument()
A2AInstrumentor().instrument()
```

### Collector-Based Session Injection

For adding session IDs in the collector pipeline:

```yaml
processors:
  attributes/session:
    actions:
      # Map from baggage or resource attributes
      - key: session.id
        from_attribute: baggage.session_id
        action: upsert
      - key: langfuse.session.id
        from_attribute: session.id
        action: upsert
      - key: galileo.session.id
        from_attribute: session.id
        action: upsert
```

---

## Related Documentation

- [Galileo Setup](galileo-setup.md) - Collector configuration for Galileo
- [Langfuse Setup](langfuse-setup.md) - Collector configuration for Langfuse
- [Traceloop Setup](traceloop-setup.md) - Collector configuration for Traceloop
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

---

*Last updated: 2025*
