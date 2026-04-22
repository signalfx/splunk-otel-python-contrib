# Splunk Distribution of OpenTelemetry (SDOT) - Gen AI Semantic Conventions Reference

This document describes the semantic conventions used in the Splunk Distribution of OpenTelemetry Python instrumentation for Generative AI, how they relate to upstream [OTel Gen AI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai), and how customers can use the resulting telemetry in Splunk APM.

> **Status**: The upstream OTel Gen AI semantic conventions are in **Development** status. SDOT implements and extends them with additional attributes for agentic AI workflows, evaluations, and context propagation.

---

## Table of Contents

- [1. Telemetry Types Overview](#1-telemetry-types-overview)
- [2. Span Types and Hierarchy](#2-span-types-and-hierarchy)
- [3. Agent Name and Workflow Name Propagation](#3-agent-name-and-workflow-name-propagation)
- [4. Agent Name Changes on Child invoke_agent](#4-agent-name-changes-on-child-invoke_agent)
- [5. Conversation ID and Association Properties](#5-conversation-id-and-association-properties)
- [6. Complete Attribute Reference](#6-complete-attribute-reference)
- [7. Metrics Reference](#7-metrics-reference)
- [8. Events Reference](#8-events-reference)
- [9. MCP (Model Context Protocol) Telemetry](#9-mcp-model-context-protocol-telemetry)
- [10. Evaluation Telemetry](#10-evaluation-telemetry)
- [11. SDOT vs Upstream OTel Semconv Comparison](#11-sdot-vs-upstream-otel-semconv-comparison)
- [12. Configuration Reference](#12-configuration-reference)
- [13. Supported Frameworks](#13-supported-frameworks)

---

## 1. Telemetry Types Overview

SDOT generates four types of telemetry for Gen AI operations:

### Spans (Distributed Traces)

Hierarchical parent-child spans representing the full execution tree of an AI workflow. Each span captures operation type, duration, attributes, and status.

**Span naming**: `{operation_name} {qualifier}` (e.g., `chat gpt-4`, `invoke_agent researcher`, `workflow customer_support`)

**Span kind**: `CLIENT` for remote calls (LLM API, remote agents), `INTERNAL` for in-process operations (local agents, workflows, tools)

### Metrics

Histogram metrics capturing operation duration, token usage, and evaluation scores. Metrics are dimensioned by operation type, model, provider, and agent context.

### Log Records / Content Events

Structured log records capturing the full input/output content of Gen AI operations. Content capture is **disabled by default** and must be opted in.

### Evaluation Events

Structured events capturing quality/accuracy scores for Gen AI outputs, emitted through the OTel Events API.

---

## 2. Span Types and Hierarchy

SDOT produces spans for each layer of an agentic AI application. The span hierarchy mirrors the logical execution flow:

```
invoke_workflow "customer_support_crew"           # Workflow root
  |
  +-- invoke_agent "classifier"                   # Agent invocation
  |     |
  |     +-- chat gpt-4                            # LLM call (inherits agent context)
  |     +-- execute_tool "classify_intent"         # Tool execution
  |
  +-- invoke_agent "resolver"                     # Different agent
  |     |
  |     +-- chat gpt-4                            # LLM call (now inherits "resolver")
  |     +-- execute_tool "search_knowledge_base"   # Tool execution
  |     +-- chat gpt-4                            # Follow-up LLM call
  |
  +-- invoke_agent "summarizer"                   # Third agent
        |
        +-- chat gpt-4                            # LLM call (inherits "summarizer")
```

### GenAI Operation Types

| `gen_ai.operation.name` | Span Name Format | Description | OTel Status |
|---|---|---|---|
| `chat` | `chat {model}` | LLM chat completion | Standard |
| `text_completion` | `text_completion {model}` | Legacy text completion | Standard |
| `embeddings` | `embeddings {model}` | Embedding generation | Standard |
| `invoke_agent` | `invoke_agent {agent_name}` | Agent invocation (local or remote) | Standard |
| `invoke_workflow` | `invoke_workflow {workflow_name}` | static sequence of GenAI operations | Standard |
| `create_agent` | `create_agent {agent_name}` | Remote agent creation | Standard |
| `execute_tool` | `execute_tool {tool_name}` | Tool/function execution | Standard |
| `retrieval` | `retrieval {data_source}` | RAG retrieval operation | Standard |

---

## 3. Agent Name and Workflow Name Propagation

### How `gen_ai.agent.name` Propagates to Child Spans

SDOT maintains an **agent context stack** -- a thread-safe/async-safe stack that tracks the currently active agent. When an agent starts, its name and ID are pushed onto the stack. All child operations (LLM calls, embeddings, retrievals, tool executions) automatically inherit the agent context from the top of the stack.

**Propagation rules:**

1. When a `chat`, `embeddings`, `retrieval`, or `execute_tool` span starts, if no explicit `gen_ai.agent.name` is provided, the instrumentation copies it from the top of the agent context stack.
2. When the agent span ends (success or failure), it is popped from the stack, restoring the parent agent context.
3. Metrics also include the agent context as dimensions, so you can break down token usage and latency by agent.

**Example: Agent context inheritance**

```
invoke_agent "researcher"                    # pushes ("researcher", id) onto stack
  |
  +-- chat gpt-4                             # gen_ai.agent.name = "researcher" (inherited)
  +-- execute_tool "web_search"              # gen_ai.agent.name = "researcher" (inherited)
  +-- embeddings text-embedding-3-small      # gen_ai.agent.name = "researcher" (inherited)
```

### How `gen_ai.workflow.name` Propagates

The workflow name is set on the root `invoke_workflow` span. Unlike agent name, workflow name does **not** automatically propagate to child spans via the context stack. It exists only on the workflow span itself. Child agents and their operations are associated to the workflow through the parent-child span relationship.

To query all operations within a workflow, filter by the trace ID of the workflow span.

---

## 4. Agent Name Changes on Child invoke_agent

When agents delegate to other agents (nested agent invocations), the agent context stack handles the transition:

```
invoke_workflow "support_crew"
  |
  +-- invoke_agent "triage"              # stack: [("triage", id1)]
  |     |
  |     +-- chat gpt-4                   # agent.name = "triage"
  |     +-- invoke_agent "specialist"    # stack: [("triage", id1), ("specialist", id2)]
  |     |     |
  |     |     +-- chat gpt-4             # agent.name = "specialist" (top of stack)
  |     |     +-- execute_tool "db_query" # agent.name = "specialist"
  |     |                                 # specialist ends -> stack: [("triage", id1)]
  |     +-- chat gpt-4                   # agent.name = "triage" (restored)
  |                                       # triage ends -> stack: []
  +-- invoke_agent "closer"              # stack: [("closer", id3)]
        |
        +-- chat gpt-4                   # agent.name = "closer"
```

**Key behaviors:**

- **Push on start**: When `invoke_agent` starts, the new agent is pushed onto the stack.
- **Pop on end**: When the agent completes (success, failure, or interruption), it is popped only if it matches the top of the stack (both name AND id must match).
- **Safety**: If `agent_id` is `None`, no pop occurs to prevent false-positive stack corruption.
- **Metrics isolation**: Each agent's LLM calls are attributed to that specific agent in the `gen_ai.client.operation.duration` and `gen_ai.client.token.usage` metrics.

---

## 5. Conversation ID and Association Properties

### `gen_ai.conversation.id`

The conversation ID ties together all operations that belong to the same user conversation or session. It propagates to **all** child spans and operations automatically.

**Setting conversation context (Python):**

```python
from opentelemetry.util.genai import genai_context

# Context manager (recommended)
with genai_context(conversation_id="conv-abc-123"):
    result = agent.invoke({"input": "What's the weather?"})
    # All spans created within this block get gen_ai.conversation.id = "conv-abc-123"

# Imperative API
from opentelemetry.util.genai import set_genai_context
set_genai_context(conversation_id="conv-abc-123")
```

**Propagation priority (highest to lowest):**
1. Explicit value on the invocation object (set by instrumentation)
2. Value from the `ContextVar` (set via `genai_context()` or `set_genai_context()`)
3. Framework inference (e.g., LangGraph `configurable.thread_id` automatically maps to `conversation_id`)

**LangGraph auto-inference**: If using LangGraph with `thread_id` in the config, the instrumentation automatically maps it to `gen_ai.conversation.id` -- no manual wrapping needed:

```python
# LangGraph thread_id is automatically used as conversation_id
result = graph.invoke(
    {"messages": [("user", "hello")]},
    config={"configurable": {"thread_id": "my-session-42"}}
)
# All spans will have gen_ai.conversation.id = "my-session-42"
```

### Association Properties

Association properties are custom key-value pairs that propagate through the context to all child spans. They appear as span attributes under the `gen_ai.association.properties.<key>` prefix.

```python
with genai_context(
    conversation_id="conv-123",
    properties={
        "user.id": "alice",
        "user.tier": "enterprise",
        "session.source": "web_chat"
    }
):
    result = agent.invoke(...)
    # All spans get:
    #   gen_ai.conversation.id = "conv-123"
    #   gen_ai.association.properties.user.id = "alice"
    #   gen_ai.association.properties.user.tier = "enterprise"
    #   gen_ai.association.properties.session.source = "web_chat"
```

**Use cases for association properties:**
- Correlating AI operations with business entities (user ID, tenant, order ID)
- Adding custom dimensions for filtering in Splunk APM
- Passing application context through the instrumentation layer

Association properties from the context are applied first, then invocation-level properties override for the same keys.

**Including association properties in metrics**: Use `OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS` to add specific context attributes as metric dimensions:

```bash
export OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS="user.tier,session.source"
```

---

## 6. Complete Attribute Reference

### Core Operation Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.operation.name` | string | Operation type (see table above) | Standard |
| `gen_ai.provider.name` | string | LLM provider (`openai`, `anthropic`, etc.) | Standard |
| `gen_ai.framework` | string | Framework (`langchain`, `crewai`, `fastmcp`, etc.) | **SDOT extension** |
| `gen_ai.request.model` | string | Requested model name | Standard |
| `gen_ai.response.model` | string | Actual model that responded | Standard |
| `gen_ai.response.id` | string | Response identifier | Standard |

### Agent Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.agent.name` | string | Human-readable agent name | Standard |
| `gen_ai.agent.id` | string | Unique agent identifier | Standard |
| `gen_ai.agent.description` | string | Agent description | Standard |
| `gen_ai.agent.tools` | string[] | Available tool names | **SDOT extension** |
| `gen_ai.agent.type` | string | Agent type (`researcher`, `planner`, `executor`, `critic`) | **SDOT extension** |
| `gen_ai.agent.system_instructions` | string | Agent system prompt/instructions | **SDOT extension** |

### Workflow Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.workflow.name` | string | Workflow identifier | Standard |
| `gen_ai.workflow.type` | string | Orchestration type (`sequential`, `parallel`, `graph`, `dynamic`) | **SDOT extension** |
| `gen_ai.workflow.description` | string | Workflow description | **SDOT extension** |

### Step Attributes (Agentic AI Extension)

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.step.name` | string | Step name within agent/workflow | **SDOT extension** |
| `gen_ai.step.type` | string | Step type (`planning`, `execution`, `reflection`, `tool_use`) | **SDOT extension** |
| `gen_ai.step.objective` | string | What the step aims to achieve | **SDOT extension** |
| `gen_ai.step.source` | string | Origin: `workflow` or `agent` | **SDOT extension** |
| `gen_ai.step.assigned_agent` | string | Agent assigned to this step | **SDOT extension** |
| `gen_ai.step.status` | string | Status (`pending`, `in_progress`, `completed`, `failed`) | **SDOT extension** |

### Conversation and Context Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.conversation.id` | string | Conversation/session/thread identifier | Standard |
| `gen_ai.association.properties.<key>` | any | Custom propagated key-value properties | **SDOT extension** |

### Token Usage Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.usage.input_tokens` | int | Input token count | Standard |
| `gen_ai.usage.output_tokens` | int | Output token count | Standard |
| `gen_ai.usage.cache_creation.input_tokens` | int | Tokens written to provider cache | Standard |
| `gen_ai.usage.cache_read.input_tokens` | int | Tokens served from provider cache | Standard |

### Request Parameters

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.request.temperature` | double | Temperature | Standard |
| `gen_ai.request.top_p` | double | Top-p sampling | Standard |
| `gen_ai.request.top_k` | double | Top-k sampling | Standard |
| `gen_ai.request.max_tokens` | int | Max tokens to generate | Standard |
| `gen_ai.request.frequency_penalty` | double | Frequency penalty | Standard |
| `gen_ai.request.presence_penalty` | double | Presence penalty | Standard |
| `gen_ai.request.stop_sequences` | string[] | Stop sequences | Standard |
| `gen_ai.request.seed` | int | Random seed | Standard |
| `gen_ai.request.stream` | boolean | Streaming mode | Standard |
| `gen_ai.request.choice_count` | int | Number of completions | Standard |
| `gen_ai.request.encoding_formats` | string[] | Embedding encoding formats | Standard |

### Response Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.response.finish_reasons` | string[] | Why the model stopped generating | Standard |
| `gen_ai.response.time_to_first_chunk` | double | Time to first streamed token (seconds) | Standard |

### Tool Execution Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.tool.name` | string | Tool/function name | Standard |
| `gen_ai.tool.call.id` | string | Tool call identifier | Standard |
| `gen_ai.tool.type` | string | Type: `function`, `extension`, `datastore` | Standard |
| `gen_ai.tool.description` | string | Tool description | Standard |
| `gen_ai.tool.call.arguments` | any | Function arguments (opt-in) | Standard |
| `gen_ai.tool.call.result` | any | Function result (opt-in) | Standard |
| `gen_ai.tool.definitions` | any | JSON-serialized tool schemas (opt-in) | Standard |

### Content Capture Attributes (Opt-In)

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.input.messages` | any | JSON-serialized input messages | Standard |
| `gen_ai.output.messages` | any | JSON-serialized output messages | Standard |
| `gen_ai.system_instructions` | any | System prompt/instructions | Standard |

### Retrieval Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.retrieval.query.text` | string | Query text (opt-in) | Standard |
| `gen_ai.retrieval.documents` | any | Retrieved documents (opt-in) | Standard |
| `gen_ai.retrieval.top_k` | int | Top K results requested | **SDOT extension** |
| `gen_ai.retrieval.documents_retrieved` | int | Number of documents retrieved | **SDOT extension** |
| `gen_ai.retrieval.type` | string | Retrieval type | **SDOT extension** |
| `gen_ai.data_source.id` | string | Data source identifier | Standard |

### Embedding Attributes

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.embeddings.dimension.count` | int | Embedding dimension count | Standard |
| `gen_ai.embeddings.input.texts` | string[] | Input texts to embed | **SDOT extension** |

### Finish Reason (Agentic AI)

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `gen_ai.finish_reason` | string | Completion reason: `interrupted`, `cancelled`, `failed` | **SDOT extension** |
| `gen_ai.finish_reason_description` | string | Human-readable description | **SDOT extension** |

### Server Context

| Attribute | Type | Description | OTel Semconv |
|---|---|---|---|
| `server.address` | string | Server address | Standard |
| `server.port` | int | Server port | Standard |

---

## 7. Metrics Reference

### Standard Gen AI Client Metrics

| Metric | Type | Unit | Description | OTel Semconv |
|---|---|---|---|---|
| `gen_ai.client.operation.duration` | Histogram | `s` | LLM/embedding/retrieval operation duration | Standard |
| `gen_ai.client.token.usage` | Histogram | `{token}` | Input and output token counts | Standard |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram | `s` | Time to first streamed token | Standard |

**Metric dimensions**: `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `server.address`, `error.type` (if error), plus agent context when applicable.

### SDOT Extension Metrics

| Metric | Type | Unit | Description |
|---|---|---|---|
| `gen_ai.workflow.duration` | Histogram | `s` | End-to-end workflow duration |
| `gen_ai.agent.duration` | Histogram | `s` | Individual agent invocation duration |
| `gen_ai.retrieval.duration` | Histogram | `s` | RAG retrieval operation duration |
| `gen_ai.evaluation.score` | Histogram | (0-1) | Evaluation metric scores (single metric mode, default) |
| `gen_ai.evaluation.client.operation.duration` | Histogram | `s` | Evaluator latency |
| `gen_ai.evaluation.client.usage.cost` | Histogram | `USD` | Evaluator cost |

### MCP Metrics

| Metric | Type | Unit | Description | OTel Semconv |
|---|---|---|---|---|
| `mcp.client.operation.duration` | Histogram | `s` | MCP client request/notification duration | Standard |
| `mcp.server.operation.duration` | Histogram | `s` | MCP server request processing duration | Standard |
| `mcp.client.session.duration` | Histogram | `s` | MCP client session duration | Standard |
| `mcp.server.session.duration` | Histogram | `s` | MCP server session duration | Standard |
| `mcp.tool.output.size` | Histogram | (bytes) | MCP tool output payload size | **SDOT extension** |

---

## 8. Events Reference

### `gen_ai.client.inference.operation.details`

A structured log record capturing the full details of an LLM call, including input/output messages. One event per LLM invocation. **Opt-in only.**

Contains all attributes from the parent span plus `gen_ai.input.messages` and `gen_ai.output.messages` in structured form.

### `gen_ai.evaluation.result`

Captures evaluation scores for Gen AI outputs.

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.evaluation.name` | string | Metric name (e.g., `Relevance`, `Hallucination`, `Toxicity`) |
| `gen_ai.evaluation.score.value` | double | Numeric score |
| `gen_ai.evaluation.score.label` | string | Categorical label (`pass`, `fail`, `relevant`, `not_relevant`) |
| `gen_ai.evaluation.explanation` | string | Evaluator's reasoning |
| `gen_ai.response.id` | string | Links to the evaluated completion |

---

## 9. MCP (Model Context Protocol) Telemetry

SDOT instruments MCP tool calls following the OTel MCP semantic conventions. MCP spans integrate into the Gen AI span hierarchy:

```
invoke_agent "assistant"
  |
  +-- chat gpt-4                                    # LLM decides to call a tool
  +-- tools/call get-weather (MCP CLIENT)            # MCP client span
  |     |
  |     +-- tools/call get-weather (MCP SERVER)      # MCP server span (remote)
  +-- chat gpt-4                                    # LLM processes tool result
```

Key MCP attributes: `mcp.method.name`, `mcp.session.id`, `mcp.protocol.version`, `gen_ai.tool.name`, `gen_ai.operation.name` (set to `execute_tool`).

---

## 10. Evaluation Telemetry

SDOT supports inline evaluation of Gen AI outputs. Evaluation results are emitted as both events and metrics.

**Evaluation attributes on spans:**

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.evaluation.sampled` | boolean | Whether this span was sampled for evaluation |

**Evaluation metrics:**

In single-metric mode (default, `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC=true`):
- `gen_ai.evaluation.score` -- histogram with `gen_ai.evaluation.name` as a dimension

In legacy mode:
- `gen_ai.evaluation.{metric_type}` -- separate histogram per evaluation type

---

## 11. SDOT vs Upstream OTel Semconv Comparison

### Attributes Aligned with Upstream OTel

These attributes follow the current OTel Gen AI semantic conventions exactly:

| Category | Attributes |
|---|---|
| **Core** | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id` |
| **Agent** | `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.agent.description`, `gen_ai.agent.version` |
| **Workflow** | `gen_ai.workflow.name` |
| **Conversation** | `gen_ai.conversation.id` |
| **Tokens** | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.usage.cache_read.input_tokens` |
| **Request params** | `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.top_k`, `gen_ai.request.max_tokens`, `gen_ai.request.frequency_penalty`, `gen_ai.request.presence_penalty`, `gen_ai.request.stop_sequences`, `gen_ai.request.seed`, `gen_ai.request.stream`, `gen_ai.request.choice.count`, `gen_ai.request.encoding_formats` |
| **Response** | `gen_ai.response.finish_reasons`, `gen_ai.response.time_to_first_chunk` |
| **Tool** | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`, `gen_ai.tool.description`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.tool.definitions` |
| **Content** | `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` |
| **Retrieval** | `gen_ai.retrieval.query.text`, `gen_ai.retrieval.documents`, `gen_ai.data_source.id` |
| **Embedding** | `gen_ai.embeddings.dimension.count` |
| **Evaluation** | `gen_ai.evaluation.name`, `gen_ai.evaluation.score.value`, `gen_ai.evaluation.score.label`, `gen_ai.evaluation.explanation` |
| **Server** | `server.address`, `server.port` |

### SDOT Extensions (Ahead of Upstream)

These attributes are defined by SDOT and are not yet part of the upstream OTel semconv. They represent areas where SDOT is pioneering agentic AI observability:

| Category | Attributes | Purpose |
|---|---|---|
| **Framework** | `gen_ai.framework` | Identifies the AI framework (LangChain, CrewAI, etc.) |
| **Agent extended** | `gen_ai.agent.tools`, `gen_ai.agent.type`, `gen_ai.agent.system_instructions` | Richer agent metadata |
| **Workflow extended** | `gen_ai.workflow.type`, `gen_ai.workflow.description` | Workflow orchestration details |
| **Steps** | `gen_ai.step.*` (name, type, objective, source, assigned_agent, status) | Fine-grained step tracking within agents/workflows |
| **Context propagation** | `gen_ai.association.properties.<key>` | Custom business context propagation |
| **Retrieval extended** | `gen_ai.retrieval.type`, `gen_ai.retrieval.top_k`, `gen_ai.retrieval.documents_retrieved` | Richer RAG telemetry |
| **Embedding extended** | `gen_ai.embeddings.input.texts` | Input text capture for embeddings |
| **Finish reason** | `gen_ai.finish_reason`, `gen_ai.finish_reason_description` | Agentic completion reasons (interrupted, cancelled) |
| **Evaluation extended** | `gen_ai.evaluation.sampled`, `gen_ai.evaluation.attributes.<key>` | Evaluation sampling and custom evaluator attributes |
| **Security** | `gen_ai.security.event_id` | Cisco AI Defense integration |
| **Internal** | `gen_ai.conversation_root`, `gen_ai.command` | Internal root span markers, resume detection |

### SDOT Extension Metrics (Beyond Upstream)

| Metric | Purpose |
|---|---|
| `gen_ai.workflow.duration` | Workflow-level latency (upstream only has operation-level) |
| `gen_ai.agent.duration` | Agent-level latency |
| `gen_ai.retrieval.duration` | RAG retrieval latency |
| `gen_ai.evaluation.score` | Evaluation quality scores |
| `gen_ai.evaluation.client.operation.duration` | Evaluator latency |
| `gen_ai.evaluation.client.usage.cost` | Evaluator cost tracking |
| `mcp.tool.output.size` | MCP tool output payload sizing |

---

## 12. Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Enable input/output message capture on spans and events |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | `SPAN_AND_EVENT` | Where to emit content: `SPAN`, `EVENT`, or `SPAN_AND_EVENT` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | `span` | Emitter selection: `span`, `span_metric`, `span_metric_event` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS` | `false` | Include tool schemas in spans |
| `OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION` | `true` | Enable context propagation (conversation_id, properties) to child spans |
| `OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS` | (none) | Comma-separated list of association property keys to add as metric dimensions |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC` | `true` | Use single `gen_ai.evaluation.score` metric vs separate per-type |
| `OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW` | (none) | Create Workflow root instead of AgentInvocation at the top level |

---

## 13. Supported Frameworks

SDOT provides auto-instrumentation for the following AI frameworks:

| Framework | Package | Key Features |
|---|---|---|
| **LangChain / LangGraph** | `opentelemetry-instrumentation-langchain` | Chains, agents, tools, `thread_id` to `conversation_id` mapping |
| **CrewAI** | `opentelemetry-instrumentation-crewai` | Crews (workflows), agents, tasks (steps), tools |
| **OpenAI SDK** | `opentelemetry-instrumentation-openai-v2` | Chat completions, embeddings, assistants |
| **OpenAI Agents** | `opentelemetry-instrumentation-openai-agents-v2` | Agent-specific instrumentation |
| **LlamaIndex** | `opentelemetry-instrumentation-llamaindex` | Query engines, retrievers, chat stores |
| **Vertex AI** | `opentelemetry-instrumentation-vertexai` | Gemini, PaLM, embeddings |
| **Weaviate** | `opentelemetry-instrumentation-weaviate` | Vector database operations |
| **FastMCP** | `opentelemetry-instrumentation-fastmcp` | MCP tool execution |
| **AI Defense** | `opentelemetry-instrumentation-aidefense` | Cisco AI Defense security scanning |
