"""Tests for the deepeval mode switching functionality.

Tests that OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE env var correctly
switches between DeepevalEvaluator (default) and DeepevalBatchedEvaluator.
"""

import importlib
import json
from types import SimpleNamespace

from opentelemetry.util.genai.evals.registry import (
    clear_registry,
    get_evaluator,
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


def test_default_mode_uses_deepeval_evaluator(monkeypatch) -> None:
    """When mode env var is not set, factory returns DeepevalEvaluator."""
    clear_registry()
    _restore_builtin_evaluators()

    # Ensure env var is not set
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", raising=False
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    # DeepevalEvaluator (not batched) should be returned
    assert type(evaluator).__name__ == "DeepevalEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_deepeval_mode_uses_deepeval_evaluator(monkeypatch) -> None:
    """When mode is 'deepeval', factory returns DeepevalEvaluator."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "deepeval"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    assert type(evaluator).__name__ == "DeepevalEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_batched_mode_uses_batched_evaluator(monkeypatch) -> None:
    """When mode is 'batched', factory returns LLMJudgeEvaluator (backward compat)."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "batched"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    # LLMJudgeEvaluator should be returned (batched is alias for llmjudge)
    assert type(evaluator).__name__ == "LLMJudgeEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_batched_mode_evaluates_with_openai(monkeypatch) -> None:
    """Batched mode should use OpenAI directly without deepeval package."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "batched"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from opentelemetry.util.evaluator import deepeval as plugin
    from opentelemetry.util.evaluator import llmjudge

    importlib.reload(plugin)
    plugin.register()

    # Mock OpenAI for the llmjudge evaluator
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {"results": {"bias": {"score": 0.1, "reason": "ok"}}}
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    stub_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    monkeypatch.setattr(
        llmjudge.openai, "OpenAI", lambda **_kwargs: stub_client
    )

    evaluator = get_evaluator("deepeval", metrics=["bias"])
    results = evaluator.evaluate(_build_invocation())

    assert len(results) == 1
    assert results[0].metric_name == "bias"
    assert results[0].score == 0.1
    assert results[0].label == "Not Biased"

    clear_registry()
    _restore_builtin_evaluators()


def test_mode_case_insensitive(monkeypatch) -> None:
    """Mode value should be case-insensitive."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "BATCHED"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    assert type(evaluator).__name__ == "LLMJudgeEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_llmjudge_mode_uses_llmjudge_evaluator(monkeypatch) -> None:
    """When mode is 'llmjudge', factory returns LLMJudgeEvaluator."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "llmjudge"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    assert type(evaluator).__name__ == "LLMJudgeEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_invalid_mode_defaults_to_deepeval(monkeypatch) -> None:
    """Invalid mode value should default to deepeval."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "invalid_mode"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    # Should fallback to DeepevalEvaluator
    assert type(evaluator).__name__ == "DeepevalEvaluator"

    clear_registry()
    _restore_builtin_evaluators()
