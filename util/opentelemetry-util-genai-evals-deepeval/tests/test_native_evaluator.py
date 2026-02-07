"""Tests for NativeEvaluator (LLM-as-a-judge evaluator)."""

import importlib
import json
from types import SimpleNamespace

import pytest

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.util.evaluator import native as native_plugin
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
    importlib.reload(native_plugin)
    native_plugin.register()
    yield
    clear_registry()
    _restore_builtin_evaluators()


@pytest.fixture
def reset_mode_env(monkeypatch):
    """Ensure mode environment variables are unset for clean tests."""
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", raising=False
    )
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_CUSTOM_RUBRICS", raising=False
    )
    yield


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


def test_registration_adds_native() -> None:
    names = list_evaluators()
    assert "native" in names


def _patch_openai(monkeypatch, *, content: str) -> None:
    """Patch OpenAI client to return specified content."""
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
        native_plugin.openai, "OpenAI", lambda **_kwargs: stub_client
    )


def _collect_metric_names(reader: InMemoryMetricReader) -> set[str]:
    metrics_data = reader.get_metrics_data()
    metrics = []
    for rm in getattr(metrics_data, "resource_metrics", []) or []:
        for scope_metrics in getattr(rm, "scope_metrics", []) or []:
            metrics.extend(getattr(scope_metrics, "metrics", []) or [])
    return {m.name for m in metrics}


def test_native_emits_evaluation_client_metrics(
    monkeypatch, reset_mode_env
) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handler:
        _meter_provider = provider

    evaluator = get_evaluator(
        "native",
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


def test_native_default_metrics_covered() -> None:
    evaluator = get_evaluator("native")
    assert set(m.lower() for m in evaluator.metrics) == {
        "bias",
        "toxicity",
        "answer_relevancy",
        "hallucination",
        "sentiment",
    }


def test_native_batched_mode_parses_results(
    monkeypatch, reset_mode_env
) -> None:
    """Test batched mode (default) parses batched JSON response."""
    invocation = _build_invocation()
    evaluator = get_evaluator(
        "native",
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


def test_native_non_batched_mode_parses_results(
    monkeypatch, reset_mode_env
) -> None:
    """Test non-batched mode parses individual JSON responses."""
    invocation = _build_invocation()

    # Create evaluator with batched=False
    evaluator = native_plugin.NativeEvaluator(
        ["bias", "toxicity"],
        invocation_type="LLMInvocation",
        batched=False,
    )

    # Track call count
    call_count = [0]

    def mock_create(**_kwargs):
        call_count[0] += 1
        # Return single-metric format for non-batched
        content = json.dumps(
            {"score": 0.1, "reason": f"result {call_count[0]}"}
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    stub_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    monkeypatch.setattr(
        native_plugin.openai, "OpenAI", lambda **_kwargs: stub_client
    )

    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)

    # Should make 2 separate calls (one per metric)
    assert call_count[0] == 2
    assert len(results) == 2
    assert {r.metric_name for r in results} == {"bias", "toxicity"}


def test_native_non_batched_env_var(monkeypatch, reset_mode_env) -> None:
    """Test that OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE=non-batched enables non-batched."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "non-batched"
    )

    # Need to reload module to pick up env var
    importlib.reload(native_plugin)

    evaluator = native_plugin.NativeEvaluator(["bias"])
    assert evaluator._batched is False


def test_native_batched_env_var(monkeypatch, reset_mode_env) -> None:
    """Test that OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE=batched enables batched."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "batched"
    )

    importlib.reload(native_plugin)

    evaluator = native_plugin.NativeEvaluator(["bias"])
    assert evaluator._batched is True


def test_native_threshold_option_affects_label(
    monkeypatch, reset_mode_env
) -> None:
    invocation = _build_invocation()
    evaluator = native_plugin.NativeEvaluator(
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


def test_native_custom_rubrics_constructor(
    monkeypatch, reset_mode_env
) -> None:
    """Test custom rubrics passed via constructor."""
    invocation = _build_invocation()

    custom_rubrics = {
        "helpfulness": {
            "rubric": "Evaluate helpfulness. Score 1=helpful, 0=not helpful.",
            "score_direction": "higher_is_better",
            "threshold": 0.5,
            "labels": {"pass": "Helpful", "fail": "Unhelpful"},
        }
    }

    evaluator = native_plugin.NativeEvaluator(
        ["helpfulness"],
        invocation_type="LLMInvocation",
        custom_rubrics=custom_rubrics,
    )

    _patch_openai(
        monkeypatch,
        content=json.dumps(
            {
                "results": {
                    "helpfulness": {"score": 0.8, "reason": "very helpful"}
                }
            }
        ),
    )

    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)

    assert len(results) == 1
    assert results[0].metric_name == "helpfulness"
    assert results[0].score == 0.8
    assert results[0].label == "Helpful"


def test_native_custom_rubrics_env_var(monkeypatch, reset_mode_env) -> None:
    """Test custom rubrics loaded from environment variable."""
    custom_rubrics_json = json.dumps(
        {
            "code_quality": {
                "rubric": "Evaluate code quality. Score 1=excellent, 0=poor.",
                "score_direction": "higher_is_better",
                "threshold": 0.6,
                "labels": {"pass": "Good Code", "fail": "Poor Code"},
            }
        }
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_CUSTOM_RUBRICS", custom_rubrics_json
    )

    importlib.reload(native_plugin)

    invocation = _build_invocation()
    evaluator = native_plugin.NativeEvaluator(
        ["code_quality"],
        invocation_type="LLMInvocation",
    )

    _patch_openai(
        monkeypatch,
        content=json.dumps(
            {
                "results": {
                    "code_quality": {"score": 0.7, "reason": "clean code"}
                }
            }
        ),
    )

    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)

    assert len(results) == 1
    assert results[0].metric_name == "code_quality"
    assert results[0].label == "Good Code"


def test_native_flexible_json_parsing(monkeypatch, reset_mode_env) -> None:
    """Test that evaluator accepts direct numeric scores without score/reason wrapper."""
    invocation = _build_invocation()
    evaluator = native_plugin.NativeEvaluator(
        ["bias"],
        invocation_type="LLMInvocation",
    )

    # Return direct numeric value instead of {"score": X, "reason": Y}
    _patch_openai(
        monkeypatch,
        content=json.dumps({"results": {"bias": 0.1}}),
    )

    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)

    assert len(results) == 1
    assert results[0].score == 0.1
    assert results[0].label == "Not Biased"


def test_native_handles_missing_output(reset_mode_env) -> None:
    """Test that evaluator handles invocations without output text."""
    invocation = LLMInvocation(request_model="test-model")
    invocation.input_messages.append(
        InputMessage(role="user", parts=[Text(content="hello")])
    )
    # No output messages

    evaluator = native_plugin.NativeEvaluator(["bias"])
    results = evaluator.evaluate(invocation)

    assert len(results) == 1
    assert results[0].error is not None
    assert "output text" in results[0].explanation.lower()


def test_native_handles_missing_api_key(monkeypatch, reset_mode_env) -> None:
    """Test that evaluator handles missing API key gracefully."""
    invocation = _build_invocation()
    evaluator = native_plugin.NativeEvaluator(["bias"])

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GENAI_OPENAI_API_KEY", raising=False)

    results = evaluator.evaluate(invocation)

    assert len(results) == 1
    assert results[0].error is not None
    assert "api key" in results[0].explanation.lower()


def test_native_attributes_include_evaluator_name(
    monkeypatch, reset_mode_env
) -> None:
    """Test that results include gen_ai.evaluation.evaluator.name attribute."""
    invocation = _build_invocation()
    evaluator = native_plugin.NativeEvaluator(["bias"])

    _patch_openai(
        monkeypatch,
        content=json.dumps(
            {"results": {"bias": {"score": 0.1, "reason": "ok"}}}
        ),
    )

    with monkeypatch.context() as m:
        m.setenv("OPENAI_API_KEY", "test-key")
        results = evaluator.evaluate(invocation)

    assert len(results) == 1
    assert (
        results[0].attributes.get("gen_ai.evaluation.evaluator.name")
        == "native"
    )
