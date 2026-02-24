# Emitter Architecture Tutorial

A guide for new team members to understand the GenAI telemetry emitter system.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [How Data Flows](#how-data-flows)
5. [The Four Emitter Categories](#the-four-emitter-categories)
6. [Configuration Guide](#configuration-guide)
7. [Creating Custom Emitters](#creating-custom-emitters)
8. [Distributed Evaluation Architecture](#distributed-evaluation-architecture)
9. [Common Scenarios](#common-scenarios)
10. [Debugging Tips](#debugging-tips)
11. [Key Files Reference](#key-files-reference)

---

## Overview

The **emitter system** is the telemetry emission layer that transforms GenAI invocations (LLM calls, agent interactions, embeddings, etc.) into OpenTelemetry signals: **spans**, **metrics**, **logs**, and **events**.

### Why This Architecture?

| Goal | How We Achieve It |
|------|-------------------|
| **Separation of concerns** | Instrumentation captures data; emitters decide *how* to emit it |
| **Composability** | Mix-and-match emitters without touching core code |
| **Extensibility** | Plugin system allows vendor-specific emitters (e.g., Splunk) |
| **Zero-code config** | Environment variables control behavior without code changes |
| **Scalability** | Distributed evaluation support for high-volume scenarios |

### Mental Model

Think of emitters like "subscribers" to GenAI lifecycle events:

```
Your GenAI App
     │
     ├── LLM call starts ────────┐
     │                           │
     │   [LLM API call...]       │
     │                           ▼
     └── LLM call ends ──── TelemetryHandler
                                 │
                                 ▼
                          CompositeEmitter
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
      SpanEmitter         MetricsEmitter      ContentEventsEmitter
            │                    │                    │
            ▼                    ▼                    ▼
    OpenTelemetry Span   OTel Histogram      OTel Log Record
```

---

## Core Concepts

### GenAI Types (Data Model)

Before emitters can do anything, instrumentation creates **invocation objects** that represent GenAI operations:

| Type | Represents | Example |
|------|------------|---------|
| `LLMInvocation` | Chat/completion calls | `gpt-4` call with messages |
| `EmbeddingInvocation` | Text embedding calls | `text-embedding-3-small` |
| `AgentInvocation` | Agent execution | CrewAI agent running |
| `Workflow` | Multi-step orchestration | CrewAI crew kickoff |
| `Step` | Individual step in workflow | Agent task execution |
| `ToolCall` | Tool/function execution | Web search, calculator |
| `RetrievalInvocation` | RAG retrieval | Vector DB query |

These types are defined in `util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py`.

### EmitterProtocol

Every emitter implements this simple interface:

```python
class EmitterProtocol(Protocol):
    """Protocol implemented by all telemetry emitters."""

    def on_start(self, obj: Any) -> None:
        """Called when an invocation starts."""
        ...

    def on_end(self, obj: Any) -> None:
        """Called when an invocation completes successfully."""
        ...

    def on_error(self, error: Error, obj: Any) -> None:
        """Called when an invocation fails."""
        ...

    def on_evaluation_results(
        self, results: Sequence[EvaluationResult], obj: Any | None = None
    ) -> None:
        """Called when evaluation results are ready."""
        ...
```

### EmitterMeta

Provides metadata for the plugin system:

```python
class EmitterMeta:
    role: str = "span"      # Category hint (informational)
    name: str = "legacy"    # Unique identifier for plugin system
    override: bool = False  # Whether to replace existing emitters

    def handles(self, obj: Any) -> bool:
        """Filter: should this emitter process this invocation type?"""
        return True
```

---

## Architecture Deep Dive

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TelemetryHandler                             │
│  (Lifecycle manager - start_llm, stop_llm, fail_llm, etc.)          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CompositeEmitter                             │
│  (Orchestrator - dispatches to category-ordered emitters)           │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │ span_emitters│  │metrics_emit. │  │content_event │  │ eval_    ││
│  │              │  │              │  │ _emitters    │  │ emitters ││
│  │ • SpanEmitter│  │ • MetricsEm. │  │ • ContentEv. │  │ • EvalMet││
│  │              │  │              │  │ • SplunkConv.│  │ • EvalEv.││
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OpenTelemetry SDK                                │
│  Tracer Provider │ Meter Provider │ Logger Provider                 │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                          OTLP Exporter → Splunk/Jaeger/etc.
```

### Lifecycle Ordering

The CompositeEmitter ensures emitters run in a specific order:

**On Start** (invocation begins):
```
1. span       → Creates the span, stores context
2. metrics    → (usually no-op on start)
3. content_events → (usually no-op on start)
```

**On End** (invocation completes):
```
1. evaluation     → Flush evaluation results
2. metrics        → Record duration, token counts
3. content_events → Emit input/output as log record
4. span           → Set finish attributes, close span
```

**Why this order?**
- Span starts first so other emitters can access span context
- Span ends last so other emitters can add attributes before closure
- Evaluation flushes first so results are available to other emitters

---

## How Data Flows

### Complete Example: LLM Invocation

```
1. YOUR CODE
   ─────────
   response = openai.chat.completions.create(
       model="gpt-4",
       messages=[{"role": "user", "content": "Hello"}]
   )

2. INSTRUMENTATION (e.g., OpenAI instrumentor)
   ──────────────────────────────────────────
   # Creates LLMInvocation dataclass
   inv = LLMInvocation(
       request_model="gpt-4",
       input_messages=[InputMessage(role="user", parts=[Text("Hello")])],
       provider="openai"
   )

   # Notifies handler
   handler.start_llm(inv)

   # ... API call happens ...

   # Updates invocation with response
   inv.output_messages = [OutputMessage(...)]
   inv.input_tokens = 10
   inv.output_tokens = 5

   handler.stop_llm(inv)

3. TELEMETRY HANDLER
   ─────────────────
   def stop_llm(self, inv):
       # Check if should sample for evaluation
       inv.sample_for_evaluation = self._should_sample(...)

       # Notify evaluation manager (async)
       self._notify_completion(inv)

       # Dispatch to all emitters
       self._emitter.on_end(inv)

4. COMPOSITE EMITTER
   ─────────────────
   def on_end(self, inv):
       # Dispatch in order: evaluation → metrics → content_events → span
       for category in ("evaluation", "metrics", "content_events", "span"):
           for emitter in self._categories[category]:
               if emitter.handles(inv):
                   emitter.on_end(inv)

5. INDIVIDUAL EMITTERS
   ───────────────────
   # SpanEmitter.on_end():
   span.set_attribute("gen_ai.response.model", inv.response_model)
   span.set_attribute("gen_ai.usage.input_tokens", inv.input_tokens)
   span.set_attribute("gen_ai.usage.output_tokens", inv.output_tokens)
   span.end()

   # MetricsEmitter.on_end():
   duration_histogram.record(inv.duration_ms, attributes={...})
   token_histogram.record(inv.input_tokens, attributes={...})

   # ContentEventsEmitter.on_end():
   log_record = LogRecord(
       body={"input": [...], "output": [...]},
       trace_id=inv.span.context.trace_id,
       span_id=inv.span.context.span_id
   )
   event_logger.emit(log_record)

6. OPENTELEMETRY SDK → OTLP EXPORTER → SPLUNK
```

---

## The Four Emitter Categories

### 1. Span Emitters (`span`)

**Purpose:** Create and populate OpenTelemetry spans

**Built-in:** `SpanEmitter` (984 lines in `emitters/span.py`)

**What it does:**
- Creates spans with semantic convention names (e.g., `"chat gpt-4"`)
- Sets attributes per OpenTelemetry GenAI semantic conventions
- Handles optional content capture (input/output messages)
- Manages span context for parent-child relationships

**Example output:**
```
Span: "chat gpt-4"
  ├── gen_ai.operation.name: "chat"
  ├── gen_ai.request.model: "gpt-4"
  ├── gen_ai.response.model: "gpt-4-0613"
  ├── gen_ai.usage.input_tokens: 10
  ├── gen_ai.usage.output_tokens: 25
  └── gen_ai.system: "openai"
```

### 2. Metrics Emitters (`metrics`)

**Purpose:** Record timing and token usage to histograms

**Built-in:** `MetricsEmitter` (461 lines in `emitters/metrics.py`)

**Metrics recorded:**

| Invocation Type | Metrics |
|-----------------|---------|
| LLM | `gen_ai.client.operation.duration`, token usage |
| Agent | `gen_ai.agent.duration` |
| Workflow | `gen_ai.workflow.duration` |
| Tool | `gen_ai.tool.duration`, MCP metrics |
| Retrieval | `gen_ai.retrieval.duration` |

### 3. Content Events Emitters (`content_events`)

**Purpose:** Emit input/output content as separate log events

**Built-in:**
- `ContentEventsEmitter` - Standard OpenTelemetry format
- `SplunkConversationEventsEmitter` - Splunk-specific format (plugin)

**Why separate from spans?**
- Reduces span attribute cardinality
- Content can be large; logs handle this better
- Enables content-specific indexing/search in backends
- Can be enabled/disabled independently

**Example log record:**
```json
{
  "event.name": "gen_ai.content.prompt",
  "gen_ai.prompt": "[{\"role\": \"user\", \"content\": \"Hello\"}]",
  "trace_id": "abc123...",
  "span_id": "def456..."
}
```

### 4. Evaluation Emitters (`evaluation`)

**Purpose:** Emit evaluation results as metrics and events

**Built-in:**
- `EvaluationMetricsEmitter` - Records scores to histograms
- `EvaluationEventsEmitter` - Emits results as log records

**What they do:**
- Receive results from async evaluation workers
- Canonicalize metric names (e.g., `"answer_relevancy"` → `"relevance"`)
- Record to `gen_ai.evaluation.score` histogram
- Emit evaluation events with trace correlation

---

## Configuration Guide

### Environment Variables

#### Emitter Selection

```bash
# Baseline selection (comma-separated)
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk

# Options:
# - span           → Only spans
# - span_metric    → Spans + metrics
# - span_metric_event → Spans + metrics + content events (recommended)
# - splunk         → Add Splunk-specific emitters
# - rate_limit_predictor → Add rate limit prediction events
```

#### Content Capture

```bash
# Enable capturing message content
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# Where to put content
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
# Options: SPAN_ONLY, EVENT_ONLY, SPAN_AND_EVENT, NONE
```

#### Evaluations

```bash
# Configure evaluators (grammar syntax)
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="Deepeval(LLMInvocation(bias,toxicity))"

# Sampling (0-1, percentage of invocations to evaluate)
OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=0.1

# Rate limiting
OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS=1
OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST=4
```

#### Category Overrides

```bash
# Replace entire category
OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS=replace:SplunkConversationEvents

# Append to category
OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS=append:CustomMetrics

# Prepend to category (runs first)
OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN=prepend:CustomSpan
```

### Common Configurations

#### Development (verbose)
```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

#### Production (Splunk)
```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="Deepeval(LLMInvocation(bias,toxicity))"
export OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=0.1
```

#### Minimal (spans only)
```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false
```

---

## Creating Custom Emitters

### Step 1: Implement the Protocol

```python
# my_emitter.py
from opentelemetry.util.genai.interfaces import EmitterMeta, EmitterProtocol
from opentelemetry.util.genai.types import LLMInvocation, Error, EvaluationResult

class MyCustomEmitter(EmitterMeta):
    """Custom emitter that logs invocations to a file."""

    role = "content_events"  # Category hint
    name = "my_custom"       # Unique identifier

    def __init__(self, log_file: str = "/tmp/genai.log"):
        self.log_file = log_file

    def handles(self, obj) -> bool:
        """Only process LLM invocations."""
        return isinstance(obj, LLMInvocation)

    def on_start(self, obj: LLMInvocation) -> None:
        with open(self.log_file, "a") as f:
            f.write(f"START: model={obj.request_model}\n")

    def on_end(self, obj: LLMInvocation) -> None:
        with open(self.log_file, "a") as f:
            f.write(f"END: tokens={obj.input_tokens}+{obj.output_tokens}\n")

    def on_error(self, error: Error, obj: LLMInvocation) -> None:
        with open(self.log_file, "a") as f:
            f.write(f"ERROR: {error.type}: {error.message}\n")

    def on_evaluation_results(self, results, obj=None) -> None:
        # Not interested in evaluation results
        pass
```

### Step 2: Create EmitterSpec

```python
# my_emitter_spec.py
from opentelemetry.util.genai.emitters.spec import EmitterSpec

def create_my_emitter(context):
    """Factory function called during pipeline construction."""
    from my_emitter import MyCustomEmitter
    return MyCustomEmitter(log_file=context.get("log_file", "/tmp/genai.log"))

my_emitter_spec = EmitterSpec(
    name="my_custom",
    category="content_events",  # Which category to add to
    factory=create_my_emitter,
    mode="append",              # append, prepend, replace-category, replace-same-name
    invocation_types=["LLMInvocation"],  # Optional filter
)
```

### Step 3: Register via Entry Points

In your `pyproject.toml`:

```toml
[project.entry-points."opentelemetry_util_genai_emitters"]
my_custom = "my_package.my_emitter_spec:my_emitter_spec"
```

### Step 4: Enable via Environment

```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,my_custom
```

---

## Distributed Evaluation Architecture

For high-volume scenarios, evaluations can run in a separate service:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Application Service                              │
│                                                                     │
│  ┌──────────────┐         ┌────────────────┐                       │
│  │ Instrumentation│────────▶│ SQLite Queue   │                       │
│  │  (CrewAI)    │ publish │ (eval requests)│                       │
│  └──────┬───────┘         └────────────────┘                       │
│         │ emit                                                      │
│         ▼                                                           │
│  ┌──────────────────────────────────────────┐                      │
│  │ Emitters (Spans, Metrics, Events)        │                      │
│  │ (NO evaluation results here)             │                      │
│  └──────────────────────────────────────────┘                      │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                              │
│  │  OTLP Exporter   │                                              │
│  └──────────────────┘                                              │
└─────────┼───────────────────────────────────────────────────────────┘
          │
          │ Traces, Metrics, Events
          ▼
   ┌────────────────────┐
   │  OTLP Collector    │◀─────────────────────────────┐
   │  (Splunk)          │                              │
   └────────────────────┘                              │
                                        Evaluation Events ONLY
                                                       │
┌──────────────────────────────────────────────────────┼──────────────┐
│                 Evaluation Service                   │              │
│              (Separate Process/Pod)                  │              │
│                                                      │              │
│  ┌────────────────┐                                 │              │
│  │ SQLite Queue   │                                 │              │
│  │ (consume)      │                                 │              │
│  └────────┬───────┘                                 │              │
│           │                                         │              │
│           ▼                                         │              │
│  ┌────────────────┐         ┌─────────────────┐    │              │
│  │ Evaluators     │────────▶│ Result Publisher │────┘              │
│  │ (DeepEval)     │         │ (with trace_id)  │                   │
│  └────────────────┘         └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Evaluation results include the original `trace_id` and `span_id`, so they appear correlated in Splunk even though they came from a different service!

Configuration for distributed mode:

```bash
# Application service
export OTEL_INSTRUMENTATION_GENAI_EVALS_MODE=external
export OTEL_INSTRUMENTATION_GENAI_EVALS_BROKER_TYPE=sqlite
export OTEL_INSTRUMENTATION_GENAI_EVALS_SQLITE_PATH=/var/lib/genai/queue.db

# Evaluation service
export OTEL_INSTRUMENTATION_GENAI_EVALS_MODE=consumer
export OTEL_INSTRUMENTATION_GENAI_EVALS_SQLITE_PATH=/var/lib/genai/queue.db
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
```

---

## Common Scenarios

### Scenario 1: Add Splunk-specific events alongside standard events

```bash
# This REPLACES content_events category entirely
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
# BAD: Only Splunk events, no standard ContentEvents

# Fix: Explicitly configure content_events category
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS=append:SplunkConversationEvents
# GOOD: Both standard + Splunk events
```

### Scenario 2: Only evaluate a subset of invocations

```bash
# Sample 10% of invocations for evaluation
OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=0.1

# Or use rate limiting (1 per second, burst of 4)
OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS=1
OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST=4
```

### Scenario 3: Different emitters for different invocation types

```python
# In EmitterSpec
EmitterSpec(
    name="agent_only_metrics",
    category="metrics",
    factory=create_agent_metrics,
    invocation_types=["AgentInvocation", "Workflow"],  # Only agents/workflows
)
```

### Scenario 4: Replace default span emitter with custom implementation

```bash
OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN=replace:MyCustomSpanEmitter
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.getLogger("opentelemetry.util.genai").setLevel(logging.DEBUG)
```

### Trace Emitter Dispatch

The CompositeEmitter logs dispatch events:

```python
# In your code
from opentelemetry.util.genai.debug import enable_genai_debug_log

enable_genai_debug_log()
# Now see: composite.dispatch.begin, composite.dispatch.emit, composite.dispatch.end
```

### Inspect Loaded Emitters

```python
from opentelemetry.util.genai.handler import get_telemetry_handler

handler = get_telemetry_handler()
emitter = handler._emitter  # CompositeEmitter

# List all emitters by category
for category, emitters in emitter.categories().items():
    print(f"\n{category}:")
    for e in emitters:
        print(f"  - {e.name} ({e.role})")
```

### Check Entry Points

```bash
# List registered emitter plugins
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='opentelemetry_util_genai_emitters')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"
```

---

## Key Files Reference

### Core Emitter Implementation

| File | Lines | Purpose |
|------|-------|---------|
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/interfaces.py` | 59 | EmitterProtocol, EmitterMeta definitions |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/composite.py` | 187 | CompositeEmitter orchestrator |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/span.py` | 984 | SpanEmitter - creates OTel spans |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/metrics.py` | 461 | MetricsEmitter - records histograms |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/content_events.py` | 154 | ContentEventsEmitter - emits log records |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/evaluation.py` | 300+ | Evaluation metric/event emitters |

### Configuration & Plugin System

| File | Purpose |
|------|---------|
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/configuration.py` | Emitter pipeline construction |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/spec.py` | EmitterSpec dataclass |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/plugins.py` | Entry point loading |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/config.py` | Environment variable parsing |

### Handler & Types

| File | Purpose |
|------|---------|
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py` | TelemetryHandler lifecycle manager |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py` | GenAI dataclasses (LLMInvocation, etc.) |

### Splunk Plugin

| File | Purpose |
|------|---------|
| `util/opentelemetry-util-genai-emitters-splunk/src/opentelemetry/util/genai/emitters/splunk.py` | Splunk-specific emitters |

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────────┐
│                    EMITTER QUICK REFERENCE                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  LIFECYCLE ORDER                                                   │
│  ───────────────                                                   │
│  on_start: span → metrics → content_events                         │
│  on_end:   evaluation → metrics → content_events → span            │
│                                                                    │
│  CATEGORIES                                                        │
│  ──────────                                                        │
│  span           → OpenTelemetry spans                              │
│  metrics        → Histograms (duration, tokens)                    │
│  content_events → Log records (input/output content)               │
│  evaluation     → Evaluation results (metrics + events)            │
│                                                                    │
│  KEY ENV VARS                                                      │
│  ────────────                                                      │
│  OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk      │
│  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true           │
│  OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="Deepeval(...)"       │
│                                                                    │
│  ENTRY POINT GROUP                                                 │
│  ────────────────                                                  │
│  opentelemetry_util_genai_emitters                                 │
│                                                                    │
│  DEBUG                                                             │
│  ─────                                                             │
│  logging.getLogger("opentelemetry.util.genai").setLevel(DEBUG)     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Read the code**: Start with `interfaces.py` → `composite.py` → `span.py`
2. **Run an example**: `instrumentation-genai/opentelemetry-instrumentation-crewai/examples/`
3. **Enable debug logs**: See what emitters are doing
4. **Try different configs**: Experiment with `OTEL_INSTRUMENTATION_GENAI_EMITTERS`
5. **Create a custom emitter**: Follow the steps above

Welcome to the team!
