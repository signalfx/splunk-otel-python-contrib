"""Tests for DeepevalBatchedEvaluator (batched LLM-as-a-judge evaluator)."""

import importlib
import json
from types import SimpleNamespace

import pytest

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.util.evaluator import deepeval_batched as batched_plugin
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING,
)
from opentelemetry.util.genai.evals.monitoring import (
    EVAL_CLIENT_OPERATION_DURATION,
    EVAL_CLIENT_TOKEN_USAGE,
)
from opentelemetry.util.genai.evals.registry import (
    clear_registry,
    get_evaluator,
    list_evaluators,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


def _restore_builtin_evaluators() -> None:
    try:
        from opentelemetry.util.genai.evals import builtins as _builtins

        importlib.reload(_builtins)
    except Exception:
        return


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    _restore_builtin_evaluators()
    importlib.reload(batched_plugin)
    batched_plugin.register()
    yield
    clear_registry()
    _restore_builtin_evaluators()


def _build_invocation() -> LLMInvocation:
    invocation = LLMInvocation(request_model="test-model")
    invocation.input_messages.append(
        InputMessage(role="user", parts=[Text(content="hello")])
    )
    invocation.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="hi there")],
            finish_reason="stop",
        )
    )
    return invocation


def test_registration_adds_deepeval_batched() -> None:
    names = list_evaluators()
    assert "deepeval_batched" in names


def _patch_openai(monkeypatch, *, content: str) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )
    stub_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    monkeypatch.setattr(
        batched_plugin.openai, "OpenAI", lambda **_kwargs: stub_client
    )


def _collect_metric_names(reader: InMemoryMetricReader) -> set[str]:
    metrics_data = reader.get_metrics_data()
    metrics = []
    for rm in getattr(metrics_data, "resource_metrics", []) or []:
        for scope_metrics in getattr(rm, "scope_metrics", []) or []:
            metrics.extend(getattr(scope_metrics, "metrics", []) or [])
    return {m.name for m in metrics}


def test_batched_emits_evaluation_client_metrics(monkeypatch) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handler:
        _meter_provider = provider

    evaluator = get_evaluator(
        "deepeval_batched",
        metrics=["bias"],
        invocation_type="LLMInvocation",
    )
    evaluator.bind_handler(_Handler())
    _patch_openai(
        monkeypatch,
        content=json.dumps(
            {"results": {"bias": {"score": 0.1, "reason": "ok"}}}
        ),
    )
    with monkeypatch.context() as m:
        m.setenv(OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING, "true")
        m.setenv("OPENAI_API_KEY", "test-key")
        evaluator.evaluate(_build_invocation())

    try:
        provider.force_flush()
    except Exception:
        pass
    try:
        reader.collect()
    except Exception:
        pass
    names = _collect_metric_names(reader)
    assert EVAL_CLIENT_OPERATION_DURATION in names
    assert EVAL_CLIENT_TOKEN_USAGE in names


def test_batched_default_metrics_covered() -> None:
    evaluator = get_evaluator("deepeval_batched")
    assert set(m.lower() for m in evaluator.metrics) == {
        "bias",
        "toxicity",
        "answer_relevancy",
        "hallucination",
        "sentiment",
    }


def test_batched_evaluator_parses_judge_results(monkeypatch) -> None:
    invocation = _build_invocation()
    evaluator = get_evaluator(
        "deepeval_batched",
        metrics=["bias", "answer_relevancy", "sentiment"],
        invocation_type="LLMInvocation",
    )
    _patch_openai(
        monkeypatch,
        content=json.dumps(
            {
                "results": {
                    "bias": {"score": 0.2, "reason": "not biased"},
                    "answer_relevancy": {
                        "score": 0.9,
                        "reason": "answers the question",
                    },
                    "sentiment": {"score": 0.9, "reason": "positive"},
                }
            }
        ),
    )
    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)
    assert {r.metric_name for r in results} == {
        "bias",
        "answer_relevancy",
        "sentiment",
    }
    by_name = {r.metric_name: r for r in results}
    assert by_name["bias"].label == "Not Biased"
    assert by_name["answer_relevancy"].label == "Relevant"
    assert by_name["sentiment"].label == "Positive"


def test_batched_metric_threshold_option_affects_label(monkeypatch) -> None:
    invocation = _build_invocation()
    evaluator = batched_plugin.DeepevalBatchedEvaluator(
        ("toxicity",),
        invocation_type="LLMInvocation",
        options={"toxicity": {"threshold": "0.1"}},
    )
    _patch_openai(
        monkeypatch,
        content=json.dumps(
            {"results": {"toxicity": {"score": 0.2, "reason": "toxic"}}}
        ),
    )
    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)
    assert len(results) == 1
    assert results[0].metric_name == "toxicity"
    assert results[0].label == "Toxic"


def test_batched_evaluator_missing_output():
    invocation = LLMInvocation(request_model="abc")
    evaluator = batched_plugin.DeepevalBatchedEvaluator(
        ("bias",), invocation_type="LLMInvocation"
    )
    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    assert results[0].error is not None


def test_batched_faithfulness_skipped_without_retrieval_context():
    invocation = _build_invocation()
    evaluator = batched_plugin.DeepevalBatchedEvaluator(
        ("faithfulness",),
        invocation_type="LLMInvocation",
    )
    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    result = results[0]
    assert result.label == "skipped"
    assert result.error is not None
    assert "retrieval_context" in (result.explanation or "")
    assert result.attributes.get("deepeval.skipped") is True
