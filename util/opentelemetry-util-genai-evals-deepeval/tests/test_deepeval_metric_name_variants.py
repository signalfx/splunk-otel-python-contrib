import importlib
from unittest.mock import patch

import pytest

from opentelemetry.util.evaluator import deepeval as plugin
from opentelemetry.util.genai.evals.registry import clear_registry
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    importlib.reload(plugin)
    plugin.register()
    yield
    clear_registry()


def _build_invocation():
    inv = LLMInvocation(request_model="variant-model")
    inv.input_messages.append(
        InputMessage(role="user", parts=[Text(content="question")])
    )
    inv.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="answer")],
            finish_reason="stop",
        )
    )
    return inv


@pytest.mark.parametrize(
    "variant,expected_key",
    [
        ("answer relevancy", "answer_relevancy"),  # spaces -> underscore
        ("answer_relevance", "answer_relevance"),  # underscore preserved
        ("relevance", "relevance"),  # direct synonym accepted
        ("answer_relevancy", "answer_relevancy"),  # canonical form
    ],
)
def test_answer_relevancy_variants_normalize(variant, expected_key):
    captured = {}

    def fake_instantiate(specs, test_case, model):
        # capture the normalized internal spec names
        captured["spec_names"] = [s.name for s in specs]
        # return a dummy metric instance so evaluation proceeds to conversion path (which will produce no data)
        return [object()], []

    with (
        patch(
            "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
            fake_instantiate,
        ),
        patch(
            "opentelemetry.util.evaluator.deepeval._build_llm_test_case",
            lambda inv: object(),
        ),
        patch(
            "opentelemetry.util.evaluator.deepeval._run_deepeval",
            lambda case, metrics, debug_log: type(
                "_DummyEval", (), {"test_results": []}
            )(),
        ),
    ):
        evaluator = plugin.DeepevalEvaluator(
            (variant,), invocation_type="LLMInvocation"
        )
        evaluator.evaluate(_build_invocation())

    assert captured["spec_names"] == [expected_key]


def test_unknown_metric_produces_error():
    # Provide metric that shouldn't resolve even after normalization
    invalid = "nonexistent-metric"

    # Patch _instantiate_metrics to raise the same ValueError pattern used by evaluator for unknown metric registry key
    def fake_instantiate(specs, test_case, model):
        raise ValueError(f"Unknown Deepeval metric '{invalid}'")

    with (
        patch(
            "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
            fake_instantiate,
        ),
        patch(
            "opentelemetry.util.evaluator.deepeval._build_llm_test_case",
            lambda inv: object(),
        ),
    ):
        evaluator = plugin.DeepevalEvaluator(
            (invalid,), invocation_type="LLMInvocation"
        )
        results = evaluator.evaluate(_build_invocation())

    # Expect one error result with the provided metric name
    assert len(results) == 1
    err = results[0]
    assert err.metric_name == invalid
    assert err.error is not None
    assert "Unknown Deepeval metric" in err.error.message
