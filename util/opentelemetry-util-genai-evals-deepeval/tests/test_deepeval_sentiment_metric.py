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


def _build_invocation() -> LLMInvocation:
    inv = LLMInvocation(request_model="sentiment-model")
    inv.input_messages.append(
        InputMessage(role="user", parts=[Text(content="I love sunny days")])
    )
    inv.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="Today is wonderful and bright!")],
            finish_reason="stop",
        )
    )
    return inv


@pytest.mark.parametrize(
    "score,expected_label",
    [
        (0.9, "Positive"),
        (0.5, "Neutral"),
        (0.1, "Negative"),
    ],
)
def test_sentiment_label_mapping(monkeypatch, score, expected_label) -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "results": {
                                "sentiment": {
                                    "score": score,
                                    "reason": "stub",
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
        ("sentiment",), invocation_type="LLMInvocation"
    )
    results = evaluator.evaluate(_build_invocation())

    assert len(results) == 1
    assert results[0].metric_name == "sentiment"
    assert results[0].label == expected_label
