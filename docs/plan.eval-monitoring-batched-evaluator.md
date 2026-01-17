# Plan: Eval Monitoring + Batched Template Evaluator

## Feature Description

This work updates the evaluator-side monitoring feature described in `README.eval-monitoring.md` to use a different evaluator approach:

- Keep the existing evaluation monitoring metrics and gating:
  - `gen_ai.evaluation.client.operation.duration`
  - `gen_ai.evaluation.client.token.usage`
  - `gen_ai.evaluation.client.queue.size`
  - `gen_ai.evaluation.client.enqueue.errors`
  - Enabled only when `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`
- Replace the Deepeval integration strategy:
  - Do **not** call Deepeval’s evaluation runner / metric classes
  - Use Deepeval prompt templates as rubric text only
  - Perform a **single batched** LLM-as-a-judge call per invocation to evaluate multiple metrics at once
  - Capture judge-call telemetry directly from the LLM client response (tokens, duration)

## Iterative Implementation Plan

1. Review existing eval monitoring + Deepeval evaluator integration
2. Design a batched judge prompt + JSON response schema
3. Implement a template-driven evaluator that calls an LLM directly (OpenAI SDK)
4. Update docs to reflect the new approach and remove Deepeval-runner dependency
5. Update unit tests to validate:
   - metric gating
   - duration/token usage metrics emitted from the evaluator side
   - evaluator result parsing for batched metrics
6. Run tests using `.venv-codex`

## Changelog

- Updated feature documentation for the new approach:
  - `README.eval-monitoring.md`
- Updated Deepeval evaluator package documentation:
  - `util/opentelemetry-util-genai-evals-deepeval/README.rst`
- Updated evaluator implementation to batch metrics in one LLM call:
  - `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py`
- Updated Deepeval evaluator unit tests for the new prompt-driven approach:
  - `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_evaluator.py`
  - `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_metric_name_variants.py`
  - `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_sentiment_metric.py`

## Pull Request Template

### Summary

Switches the Deepeval evaluator integration from “call Deepeval runner/metrics” to a **batched, template-driven** judge call that:

- reuses Deepeval prompt templates as rubrics only
- evaluates multiple metrics in a single prompt/response
- emits evaluator monitoring metrics (duration + token usage) using direct LLM response telemetry

### Monitoring Metrics

- `gen_ai.evaluation.client.operation.duration` (Histogram, `s`)
- `gen_ai.evaluation.client.token.usage` (Histogram, `{token}`)
- `gen_ai.evaluation.client.queue.size` (UpDownCounter, `1`)
- `gen_ai.evaluation.client.enqueue.errors` (Counter, `1`)
- Gated by `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`

### Tests

Run (local, `.venv-codex`):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-codex/bin/python -m pytest -q util/opentelemetry-util-genai-evals-deepeval/tests
```

### Files Changed

- `README.eval-monitoring.md`
- `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py`
- `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_evaluator.py`
- `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_metric_name_variants.py`
- `util/opentelemetry-util-genai-evals-deepeval/tests/test_deepeval_sentiment_metric.py`
