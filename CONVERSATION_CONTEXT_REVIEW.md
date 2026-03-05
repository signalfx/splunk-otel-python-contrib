# Conversation Context Review

## 1. Code Review

### Scope reviewed
- Branch: `feature/conversation-context`
- Focus: context propagation, metric dimensions, DRY/SOLID alignment
- Primary files:
  - `util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py`
  - `util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py`
  - `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/metrics.py`
  - `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/utils.py`
  - docs/examples/tests updates

### Key findings
1. **Context propagation design is correct and useful**
   - `genai_context(...)` + `ContextVar` approach is clean and framework-agnostic.
   - Priority rule is well-defined: explicit invocation value > context value.
2. **Metric context inclusion is partially applied**
   - Context metric attrs are applied for some success paths but not consistently across all end/error paths.
3. **Association property validation is permissive**
   - Keys/values are emitted directly. This is flexible but can allow high-cardinality or non-ideal attribute shapes.
4. **Docs/examples substantially improved**
   - Strong API docs and practical examples; context usage is easy to adopt.

### Review flow diagram

```text
User/App sets context
   |
   v
genai_context(conversation_id, properties)
   |
   v
TelemetryHandler.start_*()
   |
   +--> _apply_genai_context()
          - explicit invocation values win
          - context values fill missing fields
          - association properties merged
   |
   v
Emitters
   +--> Spans: conversation.id + association.properties.*
   +--> Metrics: optional include list / all
```

---

## 2. High-level analysis of support for conversation id and session id

### LangChain / LangGraph
- Native concepts:
  - `run_id`/`parent_run_id` for execution lineage
  - LangGraph `thread_id` for persisted graph thread/checkpoints
- Persistence:
  - `thread_id` state persists through configured checkpointer backend
  - `run_id` is execution-scoped
- Observation:
  - Great for orchestration state, but not a universal business conversation ID by default.

### CrewAI
- Native concepts:
  - No universal first-class `conversation_id` in current wrapper path
  - Has memory features and persistence options by scope/backend
- Persistence:
  - Memory persistence exists, but memory keying is not automatically equivalent to telemetry conversation correlation.
- Observation:
  - Best to provide app-defined conversation ID and propagate with `genai_context`.

### OpenAI Agents
- Native concepts:
  - Trace/run identity from SDK
  - Session APIs and conversation-oriented continuation options (mode-dependent)
- Persistence:
  - Depends on selected session backend (local/remote)
- Observation:
  - Supports session/conversation mechanisms, but cross-framework consistency still benefits from app-provided `conversation_id`.

### Comparative matrix

```text
Framework      | Native session/convo concept | Persisted by framework | Best telemetry strategy
---------------|-------------------------------|------------------------|-----------------------
LangChain      | run_id / parent_run_id        | partially (lineage)    | app-provided conversation_id
LangGraph      | thread_id                     | yes (checkpointer)     | map app session -> conversation_id
CrewAI         | memory scopes (not convo id)  | yes (memory backend)   | app-provided conversation_id
OpenAI Agents  | sessions / conversation modes | yes (backend dependent)| app-provided + optional native mapping
```

---

## 3. CrewAI verification

### What was executed
- Created Python 3.12 venv for verification
- Installed provided requirements set
- Installed local util package in editable mode:
  - `pip install -e util/opentelemetry-util-genai`
- Ran:
  - `customer_support.py --conversation-id verify-conv-ctx-001`
- Env vars supplied inline (including OAuth2 and OTel config)

### Verification result
- `gen_ai.conversation.id: verify-conv-ctx-001` appeared on workflow, step, and agent spans.
- `gen_ai.association.properties.app.framework: crewai` propagated correctly.
- Trace hierarchy was correct (workflow -> step -> agent).

### Wireframe trace (observed)

```text
workflow customer_support_crew
  attrs:
    gen_ai.conversation.id = verify-conv-ctx-001
    gen_ai.association.properties.app.framework = crewai
    gen_ai.operation.name = invoke_workflow
  |
  +-- step <task description>
  |     attrs:
  |       gen_ai.conversation.id = verify-conv-ctx-001
  |       gen_ai.step.assigned_agent = Senior Support Representative
  |
  +-- invoke_agent Senior Support Representative
        attrs:
          gen_ai.conversation.id = verify-conv-ctx-001
          gen_ai.agent.name = Senior Support Representative
          gen_ai.operation.name = invoke_agent
```

### Notes from runtime
- Example run completed and produced output.
- OTLP endpoint `localhost:4317` was unavailable during one run attempt (export retry/fail logs).
- A shutdown-thread warning occurred after completion, but telemetry context propagation was already validated.

---

## 4. Final notes

### Summary
- Conversation context feature is working as intended.
- It provides a clean cross-framework telemetry correlation primitive.
- CrewAI and OpenAI Agents examples now demonstrate explicit context usage.

### Pros
- Framework-agnostic correlation (`gen_ai.conversation.id`)
- Clear propagation semantics and override priority
- Minimal app integration overhead (context manager)
- Good docs/tests coverage

### Cons
- Potential high-cardinality risk if users include dynamic IDs in metric dimensions
- Context metric inclusion currently not fully uniform across all emission paths
- Association property validation/normalization is intentionally flexible but can lead to inconsistent attribute hygiene

### Recommended production guidance
- Treat `conversation_id` as an application-defined correlation key.
- Keep metric context inclusion selective (avoid `all` unless cardinality is controlled).
- Standardize property keys (`user.id`, `tenant.id`, `request.id`) across teams.
