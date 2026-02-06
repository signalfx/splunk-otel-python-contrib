# OpenTelemetry GenAI Utility

This software is for Alpha preview only. This code may be discontinued, include breaking changes and may require code changes to use it.

## 1. Why this utility exists

Provide a stable, extensible core abstraction (GenAI Types + TelemetryHandler + CompositeEmitter + Evaluator hooks) separating *instrumentation capture* from *telemetry flavor emission* so that:

- Instrumentation authors create neutral GenAI data objects once.
- Different telemetry flavors (semantic conventions, vendor enrichments, events vs attributes, aggregated evaluation results, cost / agent metrics) are produced by pluggable emitters without touching instrumentation code.
- Evaluations (LLM-as-a-judge, quality metrics) run asynchronously and re-emit results through the same handler/emitter pipeline.
- Third parties can add / replace / augment emitters in well-defined category chains.
- Configuration is primarily environment-variable driven; complexity is opt-in.

Non-goal: Replace the OpenTelemetry SDK pipeline. Emitters sit *above* the SDK using public Span / Metrics / Logs / Events APIs.

## 2. Core Concepts

### 2.1 GenAI Types (Data Model)

Implemented dataclasses (in `types.py`):

- `GenAI` - base class
- `LLMInvocation`
- `EmbeddingInvocation`
- `RetrievalInvocation`
- `Workflow`
- `AgentInvocation`
- `Step`
- `ToolCall`
- `EvaluationResult`

Base dataclass:  – fields include timing (`start_time`, `end_time`), identity (`run_id`, `parent_run_id`), context (`provider`, `framework`, `agent_*`, `system`, `conversation_id`, `data_source_id`), plus `attributes: dict[str, Any]` for free-form metadata.

Semantic attributes: fields tagged with `metadata={"semconv": <attr name>}` feed `semantic_convention_attributes()` which returns only populated values; emitters rely on this reflective approach (no hard‑coded attribute lists).

Messages: `InputMessage` / `OutputMessage` each hold `role` and `parts` (which may be `Text`, `ToolCall`, `ToolCallResponse`, or arbitrary parts). Output messages have an optional `finish_reason` (meaningful for LLM responses, omitted for agent/workflow outputs).

`EvaluationResult` fields: `metric_name`, optional `score` (float), `label` (categorical outcome), `explanation`, `error` (contains `type`, `message`), `attributes` (additional evaluator-specific key/values). No aggregate wrapper class yet.

### 2.2 TelemetryHandler

`TelemetryHandler` provides external APIs for GenAI Types lifecycle

Capabilities:

- Type-specific lifecycle: `start_llm`, `stop_llm`, `fail_llm`, plus `start/stop/fail` for embedding, tool call, workflow, agent, step.
- Generic dispatchers: `start(obj)`, `finish(obj)`, `fail(obj, error)`.
- Dynamic content capture refresh (`_refresh_capture_content`) each LLM / agentic start (re-reads env + experimental gating).
- Delegation to `CompositeEmitter` (`on_start`, `on_end`, `on_error`, `on_evaluation_results`).
- Completion callback registry (`CompletionCallback`); Evaluation Manager auto-registers if evaluators present.
- Evaluation emission via `evaluation_results(invocation, list[EvaluationResult])`.

### 2.3 Span / Trace Correlation

Invocation objects hold a `span` reference.

## 3. Emitter Architecture

### 3.1 Protocol & Meta

`EmitterProtocol` offers: `on_start(obj)`, `on_end(obj)`, `on_error(error, obj)`, `on_evaluation_results(results, obj=None)`. 

`EmitterMeta` supplies `role`, `name`, optional `override`, and a default `handles(obj)` returning `True`. Role names are informational and may not match category names (e.g., `MetricsEmitter.role == "metric"`).

### 3.2 CompositeEmitter

Defines ordered category dispatch with explicit sequences:

- Start order: `span`, `metrics`, `content_events`
- End/error order: `evaluation`, `metrics`, `content_events`, `span` (span ends last so other emitters can enrich attributes first; evaluation emitters appear first in end sequence to allow flush behavior).

Public API (current): `iter_emitters(categories)`, `emitters_for(category)`, `add_emitter(category, emitter)`. A richer `register_emitter(..., position, mode)` API is **not yet implemented**.

### 3.3 EmitterSpec & Discovery

Entry point group: `opentelemetry_util_genai_emitters` (vendor packages contribute specs).

`EmitterSpec` fields:

- `name`
- `category` (`span`, `metrics`, `content_events`, `evaluation`)
- `factory(context)`
- `mode` (`append`, `prepend`, `replace-category`, `replace-same-name`)
- `after`, `before` (ordering hints – **currently unused / inert**)
- `invocation_types` (allow-list; implemented via dynamic `handles` wrapping)

Ordering hints will either gain a resolver or be removed (open item).

### 3.4 Configuration (Emitters)

Baseline selection: `OTEL_INSTRUMENTATION_GENAI_EMITTERS` (comma-separated tokens):

- `span` (default)
- `span_metric`
- `span_metric_event`
- Additional tokens -> extra emitters (e.g. `traceloop_compat`). If the only token is `traceloop_compat`, semconv span is suppressed (`only_traceloop_compat`).

Category overrides (`OTEL_INSTRUMENTATION_GENAI_EMITTERS_<CATEGORY>` with `<CATEGORY>` = `SPAN|METRICS|CONTENT_EVENTS|EVALUATION`) support directives: `append:`, `prepend:`, `replace:` (alias for `replace-category`), `replace-category:`, `replace-same-name:`.

### 3.5 Invocation-Type Filtering

Implemented through `EmitterSpec.invocation_types`; configuration layer replaces/augments each emitter’s `handles` method to short‑circuit dispatch cheaply. No explicit positional insertion API yet; runtime additions can call `add_emitter` (append only).

### 3.6 Replace vs Append Semantics

Supported modes: `append`, `prepend`, `replace-category` (alias `replace`), `replace-same-name`. Ordering hints (`after` / `before`) are present but inactive.

### 3.7 Error Handling

CompositeEmitter wraps all emitter calls; failures are debug‑logged. Error metrics hook (`genai.emitter.errors`) is **not yet implemented** (planned enhancement).

## 4. Built-In Telemetry Emitters

### 4.1 SpanEmitter

Emits semantic attributes, optional input/output message content, system instructions, function definitions, token usage, and agent context. Finalization order ensures attributes set before span closure.

### 4.2 MetricsEmitter

Records durations and token usage to histograms: `gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, plus agentic histograms (`gen_ai.workflow.duration`, `gen_ai.agent.duration`, `gen_ai.step.duration`). Role string is `metric` (singular) – may diverge from category name `metrics`.

### 4.3 ContentEventsEmitter

Emits **one** structured log record summarizing an entire LLM invocation (inputs, outputs, system instructions) — a deliberate deviation from earlier message-per-event concept to reduce event volume. Agent/workflow/step event emission is commented out (future option).

### 4.4 Evaluation Emitters

Always present:

- `EvaluationMetricsEmitter` – emits evaluation scores to histograms. Behavior depends on `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC`:
  - **Single metric mode** (default, when unset or `true`): All evaluation scores are emitted to a single histogram `gen_ai.evaluation.score` with the evaluation type distinguished by the `gen_ai.evaluation.name` attribute.
  - **Multiple metric mode** (when `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC=false`): Separate histograms per evaluation type:
    - `gen_ai.evaluation.relevance`
    - `gen_ai.evaluation.hallucination`
    - `gen_ai.evaluation.sentiment`
    - `gen_ai.evaluation.toxicity`
    - `gen_ai.evaluation.bias`
  (Legacy dynamic `gen_ai.evaluation.score.<metric>` instruments removed.)
- `EvaluationEventsEmitter` – event per `EvaluationResult`; optional legacy variant via `OTEL_GENAI_EVALUATION_EVENT_LEGACY`.

Aggregation flag affects batching only (emitters remain active either way).

Emitted attributes (core):

- `gen_ai.evaluation.name` – metric name (always present; distinguishes evaluation type in single metric mode)
- `gen_ai.evaluation.score.value` – numeric score (events only; histogram carries values)
- `gen_ai.evaluation.score.label` – categorical label (pass/fail/neutral/etc.)
- `gen_ai.evaluation.score.units` – units of the numeric score (currently `score`)
- `gen_ai.evaluation.passed` – boolean derived when label clearly indicates pass/fail (e.g. `pass`, `success`, `fail`); numeric-only heuristic currently disabled to prevent ambiguous semantics
- Agent/workflow identity: `gen_ai.agent.name`, `gen_ai.workflow.id` when available.
- Provider/model context: `gen_ai.provider.name`, `gen_ai.request.model` when available.
- Server context: `server.address`, `server.port` when available.
- `gen_ai.operation.name` – set to `"evaluation"` only in multiple metric mode (not set in single metric mode).

## 5. Third-Party Emitters (External Packages)

An example of the third-party emitter:

- Splunk evaluation aggregation / extra metrics (`opentelemetry-util-genai-emitters-splunk`).

## 6. Configuration & Environment Variables

| Variable | Purpose                                                                                 | Notes                                                                         |
|----------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Baseline + extras selection                                                             | Values: `span`, `span_metric`, `span_metric_event`, plus extras               
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS_<CATEGORY>` | Category overrides                                                                      | Directives: append / prepend / replace / replace-category / replace-same-name |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Enable/disable message capture                                                          | Truthy enables capture; default disabled                                      |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | `SPAN_ONLY` or `EVENT_ONLY` or `SPAN_AND_EVENT` or `NONE`                               | Defaults to `SPAN_AND_EVENT` when capture enabled                             |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` | Evaluator config grammar                                                                | `Evaluator(Type(metric(opt=val)))` syntax supported                           |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION` | Aggregate vs per-evaluator emission                                                     | Boolean                                                                       |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL` | Eval worker poll interval                                                               | Default 5.0 seconds                                                           |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` | Trace-id ratio sampling                                                                 | Float (0–1], default 1.0                                                      |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_ENABLE` | Enable evaluation rate limiting                                                        | Boolean (default: true). Set to 'false' to disable rate limiting             |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS` | Evaluation request rate limit (requests per second)                                      | int (default: 0, disabled). Example: 1 = 1 request per second                 |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST` | Maximum burst size for rate limiting                                                  | int (default: 4). Allows short bursts beyond the base rate                   |
| `OTEL_GENAI_EVALUATION_EVENT_LEGACY` | Emit legacy evaluation event shape                                                      | Adds second event per result                                                  |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC` | Use single `gen_ai.evaluation.score` histogram vs separate histograms per evaluation type | Boolean (default: true)                                                       |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE` | Evaluation queue size                                                              | int (default: 100)                                                            |

## 7. Extensibility Mechanics

### 7.1 Entry Point Flow

1. Parse baseline & extras.
2. Register built-ins (span/metrics/content/evaluation).
3. Load entry point emitter specs & register.
4. Apply category overrides.
5. Instantiate `CompositeEmitter` with resolved category lists.


### 7.2 Invocation Type Filtering

`EmitterSpec.invocation_types` drives dynamic `handles` wrapper (fast pre-dispatch predicate). Evaluation emitters see results independently of invocation type filtering.

## 8. Evaluations Integration

Note: Evaluators depend on `opentelemetry-util-genai-evals` to be installed as a completion_callback.

Evaluator package entry point groups:

- `opentelemetry_util_genai_completion_callbacks` (completion callback plug-ins; evaluation manager registers here).
- `opentelemetry_util_genai_evaluators` (per-evaluator factories/registrations discovered by the evaluation manager).

Default loading honours two environment variables:

- `OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS` – optional comma-separated filter applied before instantiation.
- `OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS` – when truthy, skips loading built-in callbacks (e.g., evaluation manager).

Evaluation Manager behaviour (shipped from `opentelemetry-util-genai-evals`):

- Instantiated lazily when the evaluation completion callback binds to `TelemetryHandler`.
- Trace-id ratio sampling via `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` (falls back to enqueue if span context missing).
- Parses evaluator grammar into per-type plans (metric + options) sourced from registered evaluators.
- Aggregation flag merges buckets into a single list when true (`OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION`).
- Emits lists of `EvaluationResult` to `handler.evaluation_results`.
- Marks invocation `attributes["gen_ai.evaluation.executed"] = True` post emission.

## 9. Lifecycle Overview

```
start_* -> CompositeEmitter.on_start(span, metrics, content_events)
finish_* -> CompositeEmitter.on_end(evaluation, metrics, content_events, span)
  -> completion callbacks (Evaluation Manager enqueues)
Evaluation worker -> evaluate -> handler.evaluation_results(list) -> CompositeEmitter.on_evaluation_results(evaluation)
```

## 10. Replacement & Augmentation Scenarios

| Scenario | Configuration | Outcome |
|----------|---------------|---------|
| Add Traceloop compat span | `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span,traceloop_compat` | Semconv + compat span |
| Only Traceloop compat span | `OTEL_INSTRUMENTATION_GENAI_EMITTERS=traceloop_compat` | Compat span only |
| Replace evaluation emitters | `OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace:SplunkEvaluationAggregator` | Only Splunk evaluation emission |
| Prepend custom metrics | `OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS=prepend:MyMetrics` | Custom metrics run first |
| Replace content events | `OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS=replace:VendorContent` | Vendor events only |
| Agent-only cost metrics | (future) programmatic add with invocation_types filter | Metrics limited to agent invocations |

## 11. Error & Performance Considerations

- Emitters sandboxed (exceptions suppressed & debug logged).
- No error metric yet (planned: `genai.emitter.errors`).
- Content capture gated by experimental opt-in to prevent accidental large data egress.
- Single content event per invocation reduces volume.
- Invocation-type filtering occurs before heavy serialization.

## 12. Shared Utilities

`emitters/utils.py` includes: semantic attribute filtering, message serialization, enumeration builders (prompt/completion), function definition mapping, finish-time token usage application. Truncation / hashing helpers & PII redaction are **not yet implemented** (privacy work deferred).

## 13. Future Considerations

- Implement ordering resolver for `after` / `before` hints.
- Programmatic rich registration API (mode + position) & removal.
- Error metrics instrumentation.
- Aggregated `EvaluationResults` wrapper (with evaluator latency, counts).
- Privacy redaction & size-limiting/truncation helpers.
- Async emitters & dynamic hot-reload (deferred).
- Backpressure strategies for high-volume content events.

## 14. Development setup

Get the packages installed:

Setup a virtual env (Note: will erase your .venv in the current folder)

```bash
deactivate ; rm -rf .venv; python --version ; python -m venv .venv && . .venv/bin/activate && python -m ensurepip && python -m pip install --upgrade pip && python -m pip install pre-commit -c dev-requirements.txt && pre-commit install && python -m pip install rstcheck
```

```bash
pip install -e util/opentelemetry-util-genai --no-deps
pip install -e util/opentelemetry-util-genai-evals --no-deps
pip install -e util/opentelemetry-util-genai-evals-deepeval --no-deps
pip install -e util/opentelemetry-util-genai-emitters-splunk --no-deps
pip install -r dev-genai-requirements.txt
pip install -r instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/requirements.txt

export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
# configure which GenAI types to evaluate and which evaluations
export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="Deepeval(LLMInvocation(bias,toxicity))" 
# Deepeval optimization
export DEEPEVAL_FILE_SYSTEM=READ_ONLY
export DEEPEVAL_TELEMETRY_OPT_OUT=YES
# set environment and service names for ease of filtering
export OTEL_SERVICE_NAME=genai-eval-test
export OTEL_RESOURCE_ATTRIBUTES='deployment.environment=genai-dev'
```

For telemetry to properly work with Splunk Platform instrumentation, set the env var to enable Splunk format for aggregated evaluation results.

```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION="replace-category:SplunkEvaluationResults"
export OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true
```

### Deepeval evaluator integration configuration

Instrumentation-side evaluations can be configured using `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` environment variable

```bash
# uses defaults - evaluates LLMInvocation and AgentInvocation with 5 metrics:
# (bias,toxicity,answer_relevancy,hallucination,sentiment)
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval"

# Specific metrics for LLMInvocation
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(bias,toxicity))"

# Multiple types with metrics
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(bias,toxicity),AgentInvocation(hallucination))"

# With metric options
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(hallucination(threshold=0.8)))"
```

```bash
export OTEL_INSTRUMENTATION_GENAI_DEBUG=true
```

### to install an instrumentation library

```bash
pip install -e instrumentation-genai/opentelemetry-instrumentation-langchain --no-deps
```

Examples for each instrumentation library or package can be found in `<that package folder>/examples`, i.e.

```bash
util/opentelemetry-util-genai/examples/
```

### Installing a Translator library

To use exiting 3rd partu instrumentations and convert it to Splunk Distro semantic conventions/run instrumentation-side evaluations you can install a translator library. 

For example for existing traseloop instrumentations
```bash
pip install -e util/opentelemetry-util-genai-traceloop-translator --no-deps
```

## Installing aidefence instrumentation

```bash
pip install -e instrumentation-genai/opentelemetry-instrumentation-aidefense

export AI_DEFENSE_API_KEY="your-ai-defense-key"

python instrumentation-genai/opentelemetry-instrumentation-aidefense/examples/multi_agent_travel_planner/main.py
```

## In-code instrumentation example

Sudo-code to create LLMInvocation for your in-code for an application:

```python
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import LLMInvocation, InputMessage, OutputMessage, Text

handler = get_telemetry_handler()
user_input = "Hello"
inv = LLMInvocation(request_model="gpt-5-nano", input_messages=[InputMessage(role="user", parts=[Text(user_input))])], provider="openai")
handler.start_llm(inv)
# your code which actuall invokes llm here
# response = client.chat.completions.create(...)
# ....
inv.output_messages = [OutputMessage(role="assistant", parts=[Text("Hi!")], finish_reason="stop")]
handler.stop_llm(inv)
```

Additionally, you can run a simple example reporting an LLM Invocation

```bash
python util/opentelemetry-util-genai/examples/invocation_example.py llm --exporter otlp
```

## 15. Linting and Formatting

This project uses [pre-commit](https://pre-commit.com/) hooks to automatically check and fix linting and formatting issues before committing.

### Setting Up Pre-Commit Hooks

Install and configure pre-commit hooks (recommended to run in a virtual environment):

```bash
pip install pre-commit
pre-commit install
```

Once installed, the hooks will automatically run on every `git commit` and will:
- Fix linting issues with ruff
- Format code with ruff
- Check RST documentation files
- Update dependency locks

### Running Pre-Commit Manually

To run pre-commit checks on all files (not just staged files):

```bash
pre-commit run --all-files
```

This is useful for:
- Fixing existing lint failures in CI
- Checking the entire codebase before pushing
- Running checks without committing

### Resolving CI Lint Failures

If the CI lint job fails on your PR:

#### Option 1: Using Make (Recommended for instrumentation packages)

Some instrumentation packages include a Makefile with a lint recipe that automatically fixes all linting and formatting issues.

**Note:** It's recommended to run this in a virtual environment to avoid conflicts with system packages.

```bash
cd instrumentation-genai/opentelemetry-instrumentation-weaviate
make lint
```

This will:
- Install the correct version of ruff
- Fix all linting issues with `ruff check --fix`
- Format all code with `ruff format`
- Verify that all fixes pass CI checks

Then commit and push the changes:
```bash
git add .
git commit -m "fix: auto-fix linting issues"
git push
```

#### Option 2: Using Pre-Commit

1. **Run pre-commit on all files:**
   ```bash
   pre-commit run --all-files
   ```

2. **Review and stage the fixes:**
   ```bash
   git add .
   ```

3. **Commit and push:**
   ```bash
   git commit -m "fix: auto-fix linting issues"
   git push
   ```

The CI lint job checks:
- **Linting**: `ruff check .` - code quality issues (unused imports, undefined names, etc.)
- **Formatting**: `ruff format --check .` - code formatting consistency

Pre-commit hooks use the same ruff version and configuration as CI, ensuring local checks match CI requirements.

## 16. Test Emitter & Evaluation Performance Testing

The `splunk-otel-genai-emitters-test` package provides tools for testing and validating the evaluation framework:

- **Test Emitter**: Captures all telemetry in memory for testing and validation
- **Evaluation Performance Test**: CLI tool for validating evaluation metrics against known test samples

For detailed usage instructions, see [util/opentelemetry-util-genai-emitters-test/README.md](util/opentelemetry-util-genai-emitters-test/README.md).

Quick example:
```bash
# Install the test emitter (development only, not published to PyPI)
pip install -e ./util/opentelemetry-util-genai-emitters-test
pip install -e ./util/opentelemetry-util-genai-evals-deepeval

# Run evaluation performance test
python -m opentelemetry.util.genai.emitters.eval_perf_test \
    --samples 120 --concurrent --workers 4 --output results.json
```

## 17. Validation Strategy

- Unit tests: env parsing, category overrides, evaluator grammar, sampling, content capture gating.
- Future: ordering hints tests once implemented.
- Smoke: vendor emitters (Traceloop + Splunk) side-by-side replacement/append semantics.
