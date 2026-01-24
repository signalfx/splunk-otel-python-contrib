# Feature: Evaluation Monitoring Metrics

**Status**: ✅ Completed  
**Created**: 2026-01-23  
**Completed**: 2026-01-24  

## Overview

This feature adds monitoring metrics for the evaluation subsystem to provide customers visibility into evaluator performance. The metrics follow OpenTelemetry semantic conventions for GenAI metrics.

## Problem Statement

Currently, the evaluation system (LLM-as-a-judge) runs asynchronously and customers have limited visibility into:
- How long evaluations take (LLM calls to evaluation provider)
- Token usage for evaluation LLM calls
- Evaluation queue health (size, enqueue failures)
- Evaluator errors

## Proposed Metrics

| Metric Name | Type | Unit | Description |
|------------|------|------|-------------|
| `gen_ai.evaluation.client.operation.duration` | Histogram | `s` | Duration of calls to LLM-as-a-judge |
| `gen_ai.evaluation.client.token.usage` | Histogram | `{token}` | Token usage of calls to LLM-as-a-judge |
| `gen_ai.evaluation.client.queue.size` | UpDownCounter | `{invocation}` | Current evaluation queue size |
| `gen_ai.evaluation.client.enqueue.errors` | Counter | `{error}` | Count of sampled spans that failed to enqueue |

### Metric Attributes

Following semantic conventions from `semantic-conventions/docs/gen-ai/gen-ai-metrics.md`:

- `gen_ai.operation.name`: "evaluate"
- `gen_ai.request.model`: Model used for evaluation (e.g., "gpt-4o-mini")
- `gen_ai.system`: "openai" (or the evaluation model provider)
- `gen_ai.evaluation.name`: The evaluation metric being run (e.g., "bias", "toxicity")
- `error.type`: Error type when applicable

## Configuration

### Environment Variable

```
OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true|false
```

Default: `false` (opt-in to avoid additional overhead if not needed)

When enabled:
1. Duration and token usage metrics are recorded for each evaluation LLM call
2. Queue size is reported as an observable gauge
3. Enqueue errors are counted

## Implementation Plan

### Phase 1: Core Monitoring Infrastructure

1. **Add environment variable** to `environment_variables.py`:
   - `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING`

2. **Create monitoring module** in `util/opentelemetry-util-genai-evals/`:
   - `monitoring.py` - Contains `EvaluationMonitor` class with metric instruments

3. **Integrate with Manager**:
   - Initialize `EvaluationMonitor` when monitoring is enabled
   - Record queue size changes on enqueue/dequeue
   - Record enqueue errors

### Phase 2: Evaluator-Level Metrics

4. **Instrument deepeval evaluator**:
   - Wrap evaluation calls to capture duration
   - Capture token usage from deepeval metrics (if available)
   - Pass evaluation context to monitoring

5. **Add optional spans** (experimental):
   - Create spans for evaluation operations to provide trace context
   - Can be enabled/disabled separately

### Phase 3: Testing & Documentation

6. **Add tests**:
   - Unit tests for `EvaluationMonitor`
   - Integration tests with mocked evaluators
   - Test with real OpenAI API using existing example

7. **Update documentation**:
   - Update README with monitoring configuration
   - Update CHANGELOG.md

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/monitoring.py` | EvaluationMonitor class with metric instruments |
| `util/opentelemetry-util-genai-evals/tests/test_monitoring.py` | Tests for monitoring |

### Modified Files

| File | Changes |
|------|---------|
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/environment_variables.py` | Add `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING` |
| `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/manager.py` | Initialize and use EvaluationMonitor |
| `util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/env.py` | Add helper to read monitoring flag |
| `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py` | Capture evaluation duration/tokens |

## Sequence Diagram

```
┌───────────┐   ┌─────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────┐
│ Invocation│   │ Manager │   │EvalMonitor    │   │  Evaluator   │   │ LLM Judge │
└─────┬─────┘   └────┬────┘   └───────┬───────┘   └──────┬───────┘   └─────┬─────┘
      │              │                │                   │                 │
      │ on_completion│                │                   │                 │
      │─────────────>│                │                   │                 │
      │              │                │                   │                 │
      │              │ on_enqueue()   │                   │                 │
      │              │───────────────>│                   │                 │
      │              │                │ queue.size++      │                 │
      │              │                │                   │                 │
      │              │                │                   │                 │
      │              ├────────────────┼───────────────────┤                 │
      │              │   (async worker thread)            │                 │
      │              │                │                   │                 │
      │              │ on_eval_start()│                   │                 │
      │              │───────────────>│                   │                 │
      │              │                │                   │                 │
      │              │                │  evaluate()       │                 │
      │              │                │──────────────────>│                 │
      │              │                │                   │ LLM API call    │
      │              │                │                   │────────────────>│
      │              │                │                   │<────────────────│
      │              │                │<──────────────────│                 │
      │              │                │                   │                 │
      │              │ on_eval_end()  │                   │                 │
      │              │───────────────>│                   │                 │
      │              │                │ record duration   │                 │
      │              │                │ record tokens     │                 │
      │              │                │ queue.size--      │                 │
      │              │                │                   │                 │
```

## Testing Strategy

1. **Unit Tests**: Test `EvaluationMonitor` class with mocked meter
2. **Integration Tests**: Test full flow with real Manager and mocked evaluators
3. **Manual Tests**: Run `invocation_example.py` and verify metrics in console output

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Performance overhead from metrics | Opt-in via environment variable; use efficient metric recording |
| Deepeval token usage not exposed | Capture from evaluation_cost attribute if available |
| Queue size metric overhead | Use ObservableGauge with callback instead of increment/decrement |

## Success Criteria

- [ ] Metrics are emitted when `OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING=true`
- [ ] Duration histogram captures evaluation LLM call duration
- [ ] Token histogram captures evaluation token usage (when available)
- [ ] Queue size is observable
- [ ] Enqueue errors are counted
- [ ] All tests pass
- [ ] Documentation updated
