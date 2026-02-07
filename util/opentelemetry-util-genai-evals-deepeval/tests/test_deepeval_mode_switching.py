"""Tests for the deepeval implementation switching functionality.

Tests that OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION env var correctly
switches between NativeEvaluator (default) and DeepevalEvaluator.
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


def test_default_implementation_uses_deepeval_evaluator(monkeypatch) -> None:
    """When implementation env var is not set, factory returns DeepevalEvaluator for backward compatibility."""
    clear_registry()
    _restore_builtin_evaluators()

    # Ensure env var is not set
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION",
        raising=False,
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    # DeepevalEvaluator should be returned by default for backward compatibility
    assert type(evaluator).__name__ == "DeepevalEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_native_implementation_uses_native_evaluator(monkeypatch) -> None:
    """When implementation is 'native', factory returns NativeEvaluator."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION", "native"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    assert type(evaluator).__name__ == "NativeEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_deepeval_implementation_uses_deepeval_evaluator(monkeypatch) -> None:
    """When implementation is 'deepeval', factory returns DeepevalEvaluator."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION", "deepeval"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    assert type(evaluator).__name__ == "DeepevalEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_native_implementation_evaluates_with_openai(monkeypatch) -> None:
    """Native implementation should use OpenAI directly without deepeval package."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION", "native"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from opentelemetry.util.evaluator import deepeval as plugin
    from opentelemetry.util.evaluator import native

    importlib.reload(plugin)
    plugin.register()

    # Mock OpenAI for the native evaluator
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
    monkeypatch.setattr(native.openai, "OpenAI", lambda **_kwargs: stub_client)

    evaluator = get_evaluator("deepeval", metrics=["bias"])
    results = evaluator.evaluate(_build_invocation())

    assert len(results) == 1
    assert results[0].metric_name == "bias"
    assert results[0].score == 0.1
    assert results[0].label == "Not Biased"

    clear_registry()
    _restore_builtin_evaluators()


def test_implementation_case_insensitive(monkeypatch) -> None:
    """Implementation value should be case-insensitive."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION", "NATIVE"
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    assert type(evaluator).__name__ == "NativeEvaluator"

    clear_registry()
    _restore_builtin_evaluators()


def test_invalid_implementation_defaults_to_deepeval(monkeypatch) -> None:
    """Invalid implementation value should default to deepeval (for safety)."""
    clear_registry()
    _restore_builtin_evaluators()

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION",
        "invalid_impl",
    )

    from opentelemetry.util.evaluator import deepeval as plugin

    importlib.reload(plugin)
    plugin.register()

    evaluator = get_evaluator("deepeval", metrics=["bias"])

    # Should fallback to DeepevalEvaluator for unknown values
    assert type(evaluator).__name__ == "DeepevalEvaluator"

    clear_registry()
    _restore_builtin_evaluators()
