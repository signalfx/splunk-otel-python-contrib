#!/usr/bin/env python3
"""Integration test for the LLM-as-a-judge batched evaluator with real OpenAI API.

This test requires OPENAI_API_KEY to be set. It is skipped if the key is not available.
Run with: pytest -v -s tests/test_real_openai_integration.py

This test uses DeepevalBatchedEvaluator which does NOT require the deepeval package.
"""

import os

import pytest

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.util.evaluator.deepeval_batched import (
    DeepevalBatchedEvaluator,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING,
)
from opentelemetry.util.genai.evals.monitoring import (
    EVAL_CLIENT_OPERATION_DURATION,
    EVAL_CLIENT_TOKEN_USAGE,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


def _get_api_key() -> str | None:
    """Get OpenAI API key from environment or ~/.cr/.cr.openai."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        from pathlib import Path

        path = Path.home() / ".cr" / ".cr.openai"
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if "=" in stripped and "OPENAI_API_KEY" in stripped:
                _, value = stripped.split("=", 1)
                return value.strip().strip("'\"")
    except Exception:
        pass
    return None


def _collect_metric_names(reader: InMemoryMetricReader) -> set[str]:
    metrics_data = reader.get_metrics_data()
    metrics = []
    for rm in getattr(metrics_data, "resource_metrics", []) or []:
        for scope_metrics in getattr(rm, "scope_metrics", []) or []:
            metrics.extend(getattr(scope_metrics, "metrics", []) or [])
    return {m.name for m in metrics}


@pytest.mark.skipif(not _get_api_key(), reason="OPENAI_API_KEY not available")
def test_real_openai_evaluation(monkeypatch):
    """Test the evaluator with a real OpenAI API call."""
    api_key = _get_api_key()
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING, "true")

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handler:
        _meter_provider = provider

    # Create a test invocation
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    invocation.input_messages.append(
        InputMessage(role="user", parts=[Text(content="What is 2 + 2?")])
    )
    invocation.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="2 + 2 equals 4.")],
            finish_reason="stop",
        )
    )

    # Create evaluator and run evaluation
    evaluator = DeepevalBatchedEvaluator(
        metrics=["bias", "toxicity", "answer_relevancy"],
        invocation_type="LLMInvocation",
    )
    evaluator.bind_handler(_Handler())

    results = evaluator.evaluate(invocation)

    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    result_by_name = {r.metric_name: r for r in results}
    print("\n=== Evaluation Results ===")
    for name, result in result_by_name.items():
        print(f"  {name}:")
        print(f"    score: {result.score}")
        print(f"    label: {result.label}")
        print(f"    explanation: {result.explanation}")
        print(
            f"    passed: {result.attributes.get('gen_ai.evaluation.passed')}"
        )
        assert result.error is None, (
            f"Unexpected error for {name}: {result.error}"
        )
        assert result.score is not None, f"Missing score for {name}"

    # Verify metrics were emitted
    try:
        provider.force_flush()
    except Exception:
        pass
    try:
        reader.collect()
    except Exception:
        pass

    names = _collect_metric_names(reader)
    print("\n=== Metrics Emitted ===")
    for name in sorted(names):
        print(f"  {name}")

    assert EVAL_CLIENT_OPERATION_DURATION in names, (
        "Duration metric not emitted"
    )
    assert EVAL_CLIENT_TOKEN_USAGE in names, "Token usage metric not emitted"

    print("\n✅ Real OpenAI integration test passed!")


if __name__ == "__main__":
    import sys

    # Allow running directly
    if _get_api_key():
        os.environ["OPENAI_API_KEY"] = _get_api_key()
        os.environ[OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING] = "true"

        class MockMonkeypatch:
            @staticmethod
            def setenv(key, value):
                os.environ[key] = value

        test_real_openai_evaluation(MockMonkeypatch())
    else:
        print("OPENAI_API_KEY not available, skipping test")
        sys.exit(1)
