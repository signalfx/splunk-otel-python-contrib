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

