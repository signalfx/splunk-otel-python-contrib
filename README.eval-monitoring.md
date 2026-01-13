# OpenTelemetry GenAI Evaluations - Evaluator Monitoring Plan

This document proposes instrumentation-side monitoring for the evaluation pipeline implemented in `util/opentelemetry-util-genai-evals`.

Project context:
- Instrumentation packages live under `instrumentation-genai/`.
- Shared GenAI core utilities live under `util/` (notably `util/opentelemetry-util-genai` and `util/opentelemetry-util-genai-evals`).
- Evaluations run asynchronously via the completion callback / `Manager` queue (`util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/manager.py`).

## 1. Goals

- Provide customers visibility into **evaluator performance and reliability** (latency, token usage, backpressure, enqueue failures).
- Clearly separate **evaluation (LLM-as-a-judge)** traffic from the application’s “real” GenAI traffic.
- Keep emitted signals **low-cardinality** and safe-by-default (no prompts/content in metrics; logs only on errors).

Non-goals:
- Replacing evaluator-specific telemetry (e.g., Deepeval’s internals) or creating a full tracing model for evaluation by default.
- Emitting evaluation prompts/responses as telemetry (content capture remains handled by existing emitters and gating).

## 2. Metric Plan

The following metrics are emitted by evaluator instrumentation (not the main GenAI client instrumentation).

Emission gating:
- These monitoring metrics are emitted only when `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true` (default disabled).

### 2.1 Metric: `gen_ai.evaluation.client.operation.duration`

Tracks duration of **LLM-as-a-judge** client operations.

- Instrument: Histogram
- Unit: `s`
- Source of conventions: mirror attribute semantics from `gen_ai.client.operation.duration` in OpenTelemetry GenAI semantic conventions (`docs/gen-ai/gen-ai-metrics.md` in `open-telemetry/semantic-conventions`).

Attributes (recommended baseline, aligned with GenAI semconv):
- `gen_ai.operation.name` (Required): operation used by the judge client (e.g., `chat`, `text_completion`).
- `gen_ai.provider.name` (Required): provider used for the judge model (e.g., `openai`, `azure.ai.openai`).
- `error.type` (Conditionally Required): set when the judge call fails.
- `gen_ai.request.model` (Conditionally Required): judge request model, if known.
- `gen_ai.response.model` (Recommended): judge response model, if known.
- `server.address` / `server.port` (Recommended/Conditional): if known.

Additional low-cardinality attributes (optional, implementation-defined):
- `gen_ai.evaluation.name`: canonical evaluation metric being computed (e.g., `toxicity`, `relevance`).
- `gen_ai.evaluation.evaluator.name`: evaluator implementation identifier (e.g., `deepeval`).
- `gen_ai.invocation.type`: invocation class being evaluated (`LLMInvocation`, `AgentInvocation`, `Workflow`).

### 2.2 Metric: `gen_ai.evaluation.client.token.usage`

Tracks token usage for **LLM-as-a-judge** client operations.

- Instrument: Histogram
- Unit: `{token}`
- Source of conventions: mirror attribute semantics from `gen_ai.client.token.usage` in OpenTelemetry GenAI semantic conventions.
- Emission rule: only emit when token usage is readily available; do not guess. (Consistent with semconv guidance.)

Attributes (recommended baseline, aligned with GenAI semconv):
- `gen_ai.operation.name` (Required)
- `gen_ai.provider.name` (Required)
- `gen_ai.token.type` (Required): `input` or `output`
- `gen_ai.request.model` (Conditionally Required): if known
- `gen_ai.response.model` (Recommended): if known
- `server.address` / `server.port` (Recommended/Conditional): if known

Additional optional attributes mirror `gen_ai.evaluation.client.operation.duration` (evaluator name, evaluation metric name, invocation type).

### 2.3 Metric: `gen_ai.evaluation.client.queue.size`

Reports current evaluation queue size (backpressure/lag indicator).

- Instrument: `ObservableUpDownCounter` (preferred) or `UpDownCounter`
- Unit: `1`
- Value: number of invocations currently queued for evaluation (best-effort; `queue.qsize()` is acceptable as an approximation).

Attributes:
- None by default (keep cardinality minimal). If we later need breakdowns, add low-cardinality dimensions like `gen_ai.invocation.type`.

### 2.4 Metric: `gen_ai.evaluation.client.enqueue.errors`

Counts failures to enqueue sampled invocations for evaluation.

- Instrument: Counter
- Unit: `1`
- Increment when: `Manager.offer()` fails to enqueue due to exception or queue state.

Attributes (recommended):
- `error.type`: exception class name (or other low-cardinality error identifier).
- `gen_ai.invocation.type`: invocation type that failed to enqueue (if available).

## 3. Logging Plan (errors only)

Emit logs on evaluator pipeline failures to support debugging without relying on metric-only signals.

Baseline events:
- Enqueue failure in `Manager.offer()` (include exception info; increment `gen_ai.evaluation.client.enqueue.errors`).
- Worker loop processing failure (already logs via `_LOGGER.exception("Evaluator processing failed")`; ensure it remains informative and low-noise).
- Evaluator failures inside `_evaluate_invocation()` (today: debug log and continue). Consider promoting to warning only when the evaluator is configured and repeatedly failing (rate-limit to avoid log storms).

Correlation:
- When possible, include trace/span correlation using stored span context on the invocation (evaluation is async, so this must not assume a live span object).

## 4. Optional: Evaluator Spans (experiment)

Goal: provide deeper visibility when metrics/logs are insufficient (debugging, cost attribution, latency breakdown).

Proposal (behind an opt-in env var):
- Create a span around each *judge call* (best) or around each *evaluator execution* (fallback).
- Parent the span to the original invocation span context when available.

Suggested shape:
- Span name: `gen_ai.evaluation.client` (judge call) or `gen_ai.evaluation` (evaluator execution).
- Attributes: reuse the same low-cardinality attributes planned for the evaluation client metrics (`gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.evaluation.name`, `gen_ai.evaluation.evaluator.name`, `error.type`).

Open question to validate in review:
- Whether these spans add enough value to justify added trace volume; keep disabled by default unless a clear customer use case emerges.

## 5. Implementation Plan (phased)

### Phase 1: Queue health metrics + enqueue errors (in `opentelemetry-util-genai-evals`)

- Add instruments in the evaluation manager (`util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/manager.py`):
  - `gen_ai.evaluation.client.queue.size` (observable preferred).
  - `gen_ai.evaluation.client.enqueue.errors`.
- Wire increments/observations:
  - Observe queue size via callback (or maintain a counter with enqueue/dequeue bookkeeping).
  - Increment enqueue errors on exceptions in `Manager.offer()`.
- Ensure logs exist for enqueue errors (warn/error), but avoid log storms.

### Phase 2: Common “judge client telemetry” helper (in `opentelemetry-util-genai-evals`)

- Add a small helper API (new module) for evaluator implementations to record:
  - `gen_ai.evaluation.client.operation.duration`
  - `gen_ai.evaluation.client.token.usage` (when known)
- Keep the helper dependency-light and generic (works with any judge client library).

### Phase 3: Adopt helper in evaluators (starting with Deepeval integration)

- In `util/opentelemetry-util-genai-evals-deepeval`, wrap the outer judge call path as a first pass:
  - Record duration around the call that triggers LLM-as-a-judge behavior.
  - Attempt to extract provider/model and token usage when available; otherwise omit token usage emission.
- Revisit later if Deepeval exposes structured token usage (or if we add an optional offline token counter).

### Phase 4: Optional spans (opt-in) + validation

- Add an opt-in flag (environment variable) to enable evaluator spans.
- Validate overhead and trace volume in an example app; decide whether to keep/ship.

## 6. Testing / Validation Plan

- Unit tests in `util/opentelemetry-util-genai-evals/tests/`:
  - Enqueue error counter increments on forced enqueue failure.
  - Queue size callback does not crash and reports non-negative values.
- Manual validation:
  - Run an example with `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=...` enabled and verify metrics appear alongside existing GenAI metrics.

## 7. Open Questions

- Exact attribute set for the new evaluation-prefixed metrics: do we strictly mirror GenAI client metric attributes, or add evaluator-specific attributes (keeping cardinality low)?
- For evaluator frameworks (e.g., Deepeval) that encapsulate LLM calls: what token usage fields are reliably available, and what is the “correct” operation boundary to time?
- Should evaluator spans be enabled only for troubleshooting (opt-in), or supported as a first-class feature?

---

## 8. Current Implementation (executed)

The plan above is implemented (minus the optional evaluator spans experiment).

### 8.1 Code Changes (summary)

- Added evaluator monitoring instruments + helper APIs in `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/monitoring.py`.
  - Metrics created:
    - `gen_ai.evaluation.client.operation.duration` (Histogram, `s`)
    - `gen_ai.evaluation.client.token.usage` (Histogram, `{token}`)
    - `gen_ai.evaluation.client.queue.size` (UpDownCounter, `1`) used as a live gauge via `+1/-1` bookkeeping
    - `gen_ai.evaluation.client.enqueue.errors` (Counter, `1`)
  - Helper functions:
    - `time_client_operation(...)` (duration timing helper)
    - `record_client_token_usage(...)` (token usage emission; only when values are known)
  - Emission gating: metrics are only emitted when `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`.

- Wired queue/backpressure monitoring into the evaluation manager in `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/manager.py`.
  - On enqueue success: increments `gen_ai.evaluation.client.queue.size` by `+1`.
  - On dequeue (worker gets an item): decrements `gen_ai.evaluation.client.queue.size` by `-1`.
  - On enqueue failure: increments `gen_ai.evaluation.client.enqueue.errors` and emits a warning log with exception info.

- Enabled evaluators to use the handler’s meter provider via a lightweight binding hook:
  - `Evaluator.bind_handler(handler)` in `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/base.py`.
  - Manager calls `bind_handler()` when instantiating evaluators (best-effort).

- Instrumented Deepeval evaluator execution to emit evaluator-side client telemetry in `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py`.
  - Records `gen_ai.evaluation.client.operation.duration` around the Deepeval “judge” call.
  - Attempts best-effort token usage extraction from Deepeval results (only emits when integer token counts are available; otherwise no token metric is recorded).

### 8.2 Notes / Design Decisions

- `gen_ai.evaluation.client.queue.size` is implemented as an UpDownCounter updated on enqueue/dequeue. This behaves like a gauge in backends that support non-monotonic sums; it avoids relying on `queue.qsize()` approximation.
- Token usage is intentionally best-effort and conservative: emission only occurs when values are clearly present as non-negative integers.
- Evaluator spans remain unimplemented (still optional/experimental).

## 9. PR Documentation Template (tests + telemetry proof)

Use this section verbatim in the pull request description.

### 9.1 Summary

Adds evaluator-side monitoring metrics for the async evaluation pipeline and instruments the Deepeval evaluator to emit evaluation-client telemetry (duration + best-effort token usage), providing visibility into evaluator performance, backpressure, and enqueue failures.

### 9.2 Metrics / Telemetry Added

- `gen_ai.evaluation.client.operation.duration` (Histogram, seconds): duration of LLM-as-a-judge calls.
- `gen_ai.evaluation.client.token.usage` (Histogram, `{token}`): input/output token usage for LLM-as-a-judge calls (only when known).
- `gen_ai.evaluation.client.queue.size` (UpDownCounter, `1`): current evaluation queue size.
- `gen_ai.evaluation.client.enqueue.errors` (Counter, `1`): enqueue failures.
- Logs: warning on enqueue failure (`Manager.offer()`), including exception info.
- Emission gating: evaluator monitoring metrics are emitted only when `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`.

### 9.3 Tests Added

- `util/opentelemetry-util-genai-evals/tests/test_monitoring_metrics.py`
  - Verifies queue size returns to `0` after processing.
  - Verifies enqueue error counter increments on forced enqueue failure.
- `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_evaluator.py`
  - Verifies Deepeval evaluator emits `gen_ai.evaluation.client.operation.duration` (in-memory metrics).

### 9.4 Proof: Tests Run

Executed locally:

```bash
pytest -q util/opentelemetry-util-genai-evals/tests/test_monitoring_metrics.py
```

Result:
- `2 passed`

```bash
pytest -q util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_evaluator.py::test_deepeval_emits_evaluation_client_duration_metric
```

Result:
- `1 passed`

### 9.5 Proof: Telemetry Confirmed

Telemetry was validated via unit tests using the OpenTelemetry SDK’s `InMemoryMetricReader`:

- `gen_ai.evaluation.client.queue.size` confirmed by asserting the recorded value returns to `0` after enqueue + worker dequeue processing.
- `gen_ai.evaluation.client.enqueue.errors` confirmed by forcing an enqueue exception and asserting the counter increments to `1`.
- `gen_ai.evaluation.client.operation.duration` confirmed by executing the Deepeval evaluator and asserting the metric exists in collected in-memory metrics.

---

## 10. Change Summary (thread + implementation)

This section summarizes what changed across the planning + implementation work so far.

### 10.1 Docs

- `README.eval-monitoring.md` added as the design/plan doc, then updated with:
  - the executed implementation summary,
  - PR-ready “tests + telemetry proof” template,
  - explicit emission gating via `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`.

### 10.2 Core evaluation plumbing (`util/opentelemetry-util-genai-evals`)

- Added evaluator monitoring module `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/monitoring.py`:
  - creates the 4 metrics instruments,
  - provides helper APIs (`time_client_operation`, `record_client_token_usage`),
  - gates emission behind `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING` (default disabled).
- Updated evaluation manager `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/manager.py`:
  - increments/decrements queue size (`+1/-1`) on enqueue/dequeue,
  - increments enqueue error counter on enqueue failure,
  - logs a warning on enqueue failure (with exception info),
  - best-effort binds evaluators to the handler so evaluators can use the handler’s `meter_provider`.
- Extended evaluator base `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/base.py`:
  - adds `Evaluator.bind_handler(handler)` hook storing the handler’s meter provider (best-effort; safe default).

### 10.3 Deepeval integration (`util/opentelemetry-util-genai-evals-deepeval`)

- Updated `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py`:
  - emits `gen_ai.evaluation.client.operation.duration` around the Deepeval judge call,
  - attempts best-effort extraction of token counts and emits `gen_ai.evaluation.client.token.usage` only when integer token counts are present,
  - all of the above is gated behind `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`.

### 10.4 Environment variables (`util/opentelemetry-util-genai`)

- Added env var constant + docs for `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING` in `util/opentelemetry-util-genai/src/opentelemetry/util/genai/environment_variables.py`.

### 10.5 Tests

- Added `util/opentelemetry-util-genai-evals/tests/test_monitoring_metrics.py` (queue size and enqueue errors) and updated tests to set `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`.
- Updated `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_evaluator.py` to assert the duration metric exists, with monitoring enabled.

### 10.6 Example app tweaks (manual validation helpers)

These are not part of the core monitoring feature but help with manual validation:

- `instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot/main.py`
  - adds a `--wait-after-completion` option to allow time for async evaluations to finish before process exit.
- `instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot/runbook_search.py`
  - prefers `OpenAIEmbeddings()` when `OPENAI_API_KEY` is set, otherwise uses `AzureOpenAIEmbeddings` (simplifies local runs).

## 11. Status / Next Work

### 11.1 Current status

- Implemented: evaluator monitoring metrics, queue/enqueue instrumentation, Deepeval duration + best-effort token usage.
- Implemented: emission gating via `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true` (default off).
- Implemented: unit-test validation using `InMemoryMetricReader`.
- Not implemented: optional evaluator spans.
- Needs follow-up: architectural migration + backend visibility troubleshooting (below).

### 11.2 TODO: migrate monitoring metrics to the Emitter design (util-genai)

Current state: monitoring metrics are emitted directly from the evaluation manager / evaluator integration using the handler’s meter provider.

Target state: monitoring metrics follow the same “emitter pipeline” model as the rest of `opentelemetry-util-genai` so that:
- metrics instruments live in one place (core util-genai metrics emitter),
- enable/disable/override uses the existing emitter selection and configuration model,
- vendor packages can replace/augment evaluation monitoring emission consistently.

Proposed next steps (design work needed):
- Decide the “event boundary” that feeds the emitter pipeline:
  - Option A: add a dedicated `CompositeEmitter.on_evaluation_monitoring(...)` hook and call it from the eval manager/evaluators.
  - Option B: model evaluation monitoring as a first-class GenAI type (e.g., `EvaluationClientOperation`) and reuse `on_start/on_end` dispatch.
- Add a metrics emitter in `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/` responsible for:
  - creating the 4 instruments,
  - mapping low-cardinality attributes,
  - applying the existing emitter config/filtering/ordering mechanisms.
- Replace direct metric emission from `opentelemetry-util-genai-evals` with emitter dispatch calls.

### 11.3 TODO: troubleshoot “only 1 of 4 metrics shows up in the backend”

Observed: metrics appear to be reported in the OpenTelemetry Collector, but only one of the 4 is visible in the backend UI.

Likely causes (and what to check):
- Emission gating not enabled in the app runtime: ensure `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true` is set in the actual workload environment.
- `gen_ai.evaluation.client.token.usage` may legitimately be absent:
  - current Deepeval integration only emits token usage when integer token counts are present in Deepeval results; often this data is not exposed.
- `gen_ai.evaluation.client.enqueue.errors` may legitimately be absent:
  - it only increments on enqueue failure; in healthy runs it stays at 0 and some UIs hide never-nonzero time series.
- `gen_ai.evaluation.client.queue.size` may be hard to “see” depending on backend semantics:
  - it is implemented as an `UpDownCounter` with `+1/-1` bookkeeping and will often be `0` at scrape/export time; some backends or UIs hide constant-zero sums.

Concrete troubleshooting steps:
- In the backend metric explorer, search by prefix `gen_ai.evaluation.client.` (not by dashboard widgets).
- Confirm the time series is not being dropped due to dimensionality:
  - check `service.name` and other resource attributes match what you’re filtering on.
- Confirm temporality handling:
  - some backends/exporters treat non-monotonic sums differently than histograms; verify the OTLP exporter/backend supports `UpDownCounter` and non-monotonic sum presentation.
- If the backend/UI still hides `queue.size`, consider switching `queue.size` to an observable instrument (`ObservableUpDownCounter` / gauge-style callback) so a point is emitted every collection cycle even when the value is 0.
