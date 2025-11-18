from __future__ import annotations

from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.types import EvaluationResult, LLMInvocation


class _StubHandler:  # Minimal handler stub; not a TelemetryHandler (type: ignore[override])
    def __init__(self) -> None:
        self.calls: list[list[EvaluationResult]] = []

    def evaluation_results(
        self, invocation: LLMInvocation, results: list[EvaluationResult]
    ):
        self.calls.append(list(results))


def test_dynamic_aggregation_env_toggle(monkeypatch):  # type: ignore[no-untyped-def]
    # Start with aggregation disabled via explicit false / absence
    monkeypatch.delenv(  # type: ignore[attr-defined]
        "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION", raising=False
    )
    monkeypatch.setenv(  # type: ignore[attr-defined]
        "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
    )
    handler = _StubHandler()
    manager = Manager(handler)  # type: ignore[arg-type]
    # Simulate evaluators present manually
    manager._evaluators = {"LLMInvocation": []}
    # Force buckets emission path
    invocation = LLMInvocation(request_model="dyn-agg-model")
    buckets = [
        [EvaluationResult(metric_name="bias", score=0.1)],
        [EvaluationResult(metric_name="toxicity", score=0.3)],
    ]
    # Set internal aggregate flag to None to enable dynamic env var reading
    manager._aggregate_results = None
    manager._publish_results(invocation, buckets)
    assert len(handler.calls) == 2  # two separate batches (env var not set)

    # Now enable aggregation via env and emit again -> should aggregate
    monkeypatch.setenv(  # type: ignore[attr-defined]
        "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION", "true"
    )
    handler.calls.clear()
    manager._publish_results(invocation, buckets)
    assert len(handler.calls) == 1
    assert [r.metric_name for r in handler.calls[0]] == ["bias", "toxicity"]
