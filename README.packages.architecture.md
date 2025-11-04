
# OpenTelemetry GenAI Utility - Packages architecture

## 1. Goals (Why this utility exists)
Provide a stable, extensible core abstraction (GenAI Types + TelemetryHandler + CompositeEmitter + Evaluator hooks) separating *instrumentation capture* from *telemetry flavor emission* so that:
- Instrumentation authors create neutral GenAI data objects once.
- Different telemetry flavors (semantic conventions, vendor enrichments, events vs attributes, aggregated evaluation results, cost / agent metrics) are produced by pluggable emitters without touching instrumentation code.
- Evaluations (LLM-as-a-judge, quality metrics) run asynchronously and re-emit results through the same handler/emitter pipeline.
- Third parties can add / replace / augment emitters in well-defined category chains.
- Configuration is primarily environment-variable driven; complexity is opt-in.

Non-goal: Replace the OpenTelemetry SDK pipeline. Emitters sit *above* the SDK using public Span / Metrics / Logs / Events APIs.

## Core Package: `opentelemetry-util-genai-dev`

Purpose: Neutral GenAI data model + handler façade + builtin emitters + evaluator manager integration. This is the refactor target that will eventually publish as stable `opentelemetry-util-genai`.

Recent changes (since prior snapshot):

* Added `debug.py` (central guarded debug log helper used by plugins/evaluators).
* Added `span_context.py` (store/extract span context for evaluator sampling when original span is not directly accessible).
* Extended evaluator normalization (`evaluators/normalize.py`) for skip policies (tool-only LLM, non-invoke agent operations, agent creation, etc.).
* Deepeval integration now emits richer sentiment decomposition attributes and supports dynamic metric option coercion.
* Evaluation sampling via `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` using `TraceIdRatioBased`.

Directory (trimmed & updated):

```text
src/opentelemetry/util/genai/
  __init__.py               # public API exports
  version.py                # version constant
  config.py                 # runtime config helpers
  environment_variables.py  # OTEL_INSTRUMENTATION_GENAI_* parsing
  interfaces.py             # Protocols (EmitterProtocol, CompletionCallback, Sampler, Evaluator)
  types.py                  # Core GenAI types (LLMInvocation, Agent/Workflow/Task/ToolCall, EmbeddingInvocation, EvaluationResult[s])
  attributes.py             # Semantic attribute extraction & mapping
  handler.py                # TelemetryHandler façade (start/end + evaluation dispatch)
  callbacks.py              # Completion callback registration (Manager hook)
  instruments.py            # Metric instruments (counters, histograms, observable gauges)
  plugins.py                # Entry point discovery (emitters, evaluators)
  utils.py                  # Truncation, hashing, safe serialization helpers
  upload_hook.py            # Optional artifact/fsspec upload registration
  span_context.py           # SpanContext capture/restore for async evaluation sampling
  debug.py                  # Opt-in debug logging facility (no-op if disabled)
  _fsspec_upload/           # Internal helper modules (implementation detail)
  emitters/
    __init__.py
    spec.py                 # EmitterSpec (name, kind, factory, mode, position, filter)
    composite.py            # CompositeEmitter (chains + fan-out + ordering)
    configuration.py        # Env-var chain directive parsing (replace-category, position, mode)
    span.py                 # Semantic-convention span emitter
    metrics.py              # Metrics emitter (invocation counters, token usage, latency)
    content_events.py       # Message content events/log emission (optional capture)
    evaluation.py           # Evaluation result(s) emitter bridging Manager -> OTel
    utils.py                # Shared mapping helpers & safe value transforms
```

Removed (from previous doc): `evaluation_emitters.py` (responsibility merged into `emitters/evaluation.py` and handler path).

### Key Interfaces (summary – abridged)

```python
class GenAI: ...                            # Base for all invocation lifecycle objects
class LLMInvocation(GenAI): ...             # request model, input/output messages, token usage
class AgentInvocation(GenAI): ...           # agent type, operation (invoke_agent), messages
class Workflow(GenAI): ...                  # workflow_type, description, initial/final IO
class ToolCall(GenAI): ...                  # tool name, input/output, error
class EmbeddingInvocation(GenAI): ...       # input texts, model, vector counts
class EvaluationResult:                     # metric_name, score, label, explanation, error?, attrs

class TelemetryHandler:
    def start_llm(...)->LLMInvocation: ...   # or context manager variant
    def stop_llm(inv) -> None: ...
    def evaluation_results(inv, list[EvaluationResult]) -> None: ...
    def register_completion_callback(cb: CompletionCallback) -> None: ...

class EmitterProtocol(Protocol):
    def on_start(inv: GenAI) -> None: ...
    def on_end(inv: GenAI) -> None: ...
    def on_evaluation_results(inv: GenAI, results: list[EvaluationResult]) -> None: ...

class CompositeEmitter:
    def register_emitter(emitter, category, *, position="last", invocation_types=None, mode="append") -> None: ...

class CompletionCallback: def on_completion(inv: GenAI) -> None: ...

class Evaluator:
    metrics: Sequence[str]
    invocation_type: str
    def evaluate(inv: GenAI) -> list[EvaluationResult]: ...
    def default_metrics_by_type(self) -> Mapping[str, Sequence[str]]: ...
```

Entry points:

```text
opentelemetry_util_genai_emitters             # returns list[EmitterSpec]
opentelemetry_util_genai_evaluators           # returns list[Evaluator factory/spec]
opentelemetry_util_genai_completion_callbacks # returns completion callback instances or factories
```

Environment variables (subset – updated):

```text
OTEL_INSTRUMENTATION_GENAI_ENABLE=true|false
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN|EVENT|SPAN_AND_EVENT
OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN=...                # chain directives
OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS=...
OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS=...
OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=...
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=...             # grammar for evaluator plans
OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true|false
OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL=5.0               # async worker poll interval (seconds)
OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=1.0       # 0.0–1.0 trace-based sampling for evaluations
OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS=...         # filter completion plugin names
OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS=true|false
```

---

## Evaluations Package: `opentelemetry-util-genai-evals`

Purpose: Provides the pluggable evaluation manager, registry, and environment parsing used by
`opentelemetry-util-genai` when evaluation features are installed. Completion callbacks are exposed
via the `opentelemetry_util_genai_completion_callbacks` entry point group so downstream packages can
participate without hard dependencies.

Key modules (`src/opentelemetry/util/genai/evals/`):

```text
bootstrap.py  # EvaluatorCompletionCallback factory invoked by completion callback loader
manager.py    # Asynchronous evaluation queue, sampler, and result publication pipeline
registry.py   # Evaluator registration + entry point discovery helpers
env.py        # Centralised parsing for evaluation-related environment variables
builtins.py   # Lightweight builtin evaluators (e.g., length) for default smoke coverage
```

Tests reside in `util/opentelemetry-util-genai-evals/tests/` and cover manager behaviour,
environment parsing, and handler integration via the exported completion callback.

---

## Emitters Package: `opentelemetry-util-genai-emitters-splunk`

Purpose: Splunk-specific evaluation aggregation + extra metrics/events.

```text
src/opentelemetry/util/genai/emitters/splunk.py
  SplunkEvaluationAggregator  # kind="evaluation" (often replace-category)
  SplunkExtraMetricsEmitter   # kind="metrics" (append)
  load_emitters() -> list[EmitterSpec]
version.py
```
---

## Translator Package: `opentelemetry-util-genai-traceloop-translator`

Purpose: Automatically translates Traceloop-instrumented spans into OpenTelemetry GenAI semantic conventions using a span processor approach.

Key mappings:
- `traceloop.workflow.name` → `gen_ai.workflow.name`
- `traceloop.entity.name` → `gen_ai.agent.name` 
- `traceloop.entity.path` → `gen_ai.workflow.path`
- `traceloop.correlation.id` → `gen_ai.conversation.id`
- `traceloop.entity.input` → `gen_ai.input.messages`
- `traceloop.entity.output` → `gen_ai.output.messages`

The translator automatically activates on import via the `opentelemetry_configurator` entry point and intercepts spans with `traceloop.*` attributes, creating corresponding spans with `gen_ai.*` attributes for seamless integration with GenAI observability tools.

---

## Evaluators Package: `opentelemetry-util-genai-evals-deepeval`

Purpose: Deepeval metrics (bias, toxicity, answer_relevancy, hallucination, sentiment, etc.) with LLM-as-a-judge backend.

Updated structure:

```text
src/opentelemetry/util/evaluator/
  deepeval.py          # DeepevalEvaluator + registration() / register()
  deepeval_adapter.py  # Build test cases from GenAI invocations
  deepeval_metrics.py  # Metric registry + instantiation + option coercion
  deepeval_runner.py   # Execution wrapper (run_evaluation)
  version.py
```

Notes:

* Internal Deepeval telemetry disabled by default (`DEEPEVAL_TELEMETRY_OPT_OUT=1`).
* Sentiment post-processing derives pos/neg/neu strengths + compound normalization.
* Metric option coercion handles bool/number/string unification.
* Default model fallback: `gpt-4o-mini` if none of `DEEPEVAL_EVALUATION_MODEL|DEEPEVAL_MODEL|OPENAI_MODEL` set.

---

## Langchain Instrumentation Package: `opentelemetry-instrumentation-langchain`

Purpose: Instrumentation package for Langchain/Langgraph frameworks

```text
instrumentation-genai/opentelemetry-instrumentation-langchain/src/opentelemetry/instrumentation/langchain
  LangchainCallbackHandler               # implements Langchain callback handler to aquire telemetry 
```

---

## Invocation Lifecycle (ASCII – LLM + async evaluations)

```text
Instrumentation         Emitters (Composite)                     Evaluators
--------------          ---------------------                    ----------
with handler.start_llm_invocation() as inv:  on_start(span, metrics, ...)
    model_call()                             (spans begun, metrics prealloc)
    inv.add_output_message(...)
handler.end(inv) --------> on_end(span, metrics, content_events)
        |                        |     |         |
        |                        |     |         +--> message events/logs
        |                        |     +------------> latency / tokens metrics
        |                        +------------------> span attrs + end
        v
  CompletionCallbacks (Evaluator Manager) enqueue(inv)
        |
  async loop ------------> evaluators.evaluate(inv) -> [EvaluationResult]
        | aggregate? (env toggle)
        v
handler.evaluation_results(batch|single) -> on_evaluation_results(evaluation emitters)
        |
  evaluation events/metrics (e.g. Splunk aggregated)
        v
OTel SDK exporters send spans / metrics / logs
```

---

## Replacement / Augmentation Examples

```text
Add Traceloop extras:
  (install package) -> auto append TraceloopSpanEmitter

Replace evaluation emission with Splunk aggregator:
  OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationAggregator

Custom metrics only for LLM:
  composite.register_emitter(MyLLMCostMetrics(), 'metrics', invocation_types={'LLMInvocation'})
```

---

## Error & Performance Notes (updated)

```text
Evaluator sampling: trace-based ratio (sample rate env) before enqueue.
Skip policies: tool-only LLM responses, agent creation events, non-invoke agent operations.
Emitter errors caught; increment genai.emitter.errors(emitter,category,phase).
Truncation + hashing before emitting large message content.
Aggregation flag sets attribute gen_ai.evaluation.aggregated=true on invocation.
SpanContext persisted for evaluations even if original span object released.
Metric option coercion + normalization reduce downstream cardinality.
Heavy semantic judgments delegated to evaluator layer; emitters remain lightweight.
```

---

## Out of Scope (Initial)

```text
Async emitters, dynamic hot-swap reconfig, advanced PII redaction, large queue backpressure.
```

---
