import importlib
import json
from types import SimpleNamespace

import pytest

from opentelemetry.util.evaluator import deepeval as plugin
from opentelemetry.util.genai.evals.registry import clear_registry
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
    importlib.reload(plugin)
    plugin.register()
    yield
    clear_registry()
    _restore_builtin_evaluators()


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
    "variant",
    [
        "answer relevancy",  # spaces -> underscore
        "answer_relevance",  # alias -> canonical
        "relevance",  # alias -> canonical
        "answer_relevancy",  # canonical form
    ],
)
def test_answer_relevancy_variants_normalize_to_canonical(
    monkeypatch, variant
):
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "results": {
                                "answer_relevancy": {
                                    "score": 0.9,
                                    "reason": "ok",
                                }
                            }
                        }
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    stub_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    monkeypatch.setattr(plugin.openai, "OpenAI", lambda **_kwargs: stub_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    evaluator = plugin.DeepevalEvaluator(
        (variant,), invocation_type="LLMInvocation"
    )
    results = evaluator.evaluate(_build_invocation())

    assert len(results) == 1
    assert results[0].metric_name == "answer_relevancy"


def test_unknown_metric_produces_error(monkeypatch):
    # Provide metric that shouldn't resolve even after normalization
    invalid = "nonexistent-metric"

    # Expect one error result with the provided metric name
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = plugin.DeepevalEvaluator(
        (invalid,), invocation_type="LLMInvocation"
    )
    results = evaluator.evaluate(_build_invocation())
    assert len(results) == 1
    err = results[0]
    assert err.metric_name == invalid
    assert err.error is not None
    assert (
        "Unknown Deepeval metric" in err.error.message
        or "Unknown Deepeval metric(s)" in err.error.message
    )
