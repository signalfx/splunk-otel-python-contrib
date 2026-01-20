"""Tests for deepeval_runner.py async/concurrent evaluation support."""

# ruff: noqa: E402

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Stub Setup (reuse pattern from test_deepeval_evaluator.py)
# =============================================================================


def _install_deepeval_stubs():
    """Install minimal deepeval stubs if not available."""
    if "deepeval" in sys.modules:
        return
    try:
        __import__("deepeval")
        return
    except Exception:
        pass

    root = types.ModuleType("deepeval")
    eval_cfg_mod = types.ModuleType("deepeval.evaluate.configs")

    class AsyncConfig:
        def __init__(self, run_async=False):
            self.run_async = run_async

    class DisplayConfig:
        def __init__(self, show_indicator=False, print_results=False):
            self.show_indicator = show_indicator
            self.print_results = print_results

    eval_cfg_mod.AsyncConfig = AsyncConfig
    eval_cfg_mod.DisplayConfig = DisplayConfig

    class _EvalResult:
        test_results = []

    def evaluate(test_cases, metrics, async_config=None, display_config=None):
        return _EvalResult()

    root.evaluate = evaluate

    sys.modules["deepeval"] = root
    sys.modules["deepeval.evaluate"] = root
    sys.modules["deepeval.evaluate.configs"] = eval_cfg_mod


_install_deepeval_stubs()


from opentelemetry.util.evaluator.deepeval_runner import (
    _is_async_mode_enabled,
    run_evaluation,
    run_evaluation_async,
)

# =============================================================================
# Tests for _is_async_mode_enabled
# =============================================================================


class TestIsAsyncModeEnabled:
    """Tests for _is_async_mode_enabled helper function."""

    def test_returns_false_when_not_set(self, monkeypatch):
        """Returns False when env var not set."""
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", raising=False
        )
        assert _is_async_mode_enabled() is False

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
    def test_returns_true_for_truthy_values(self, monkeypatch, value):
        """Returns True for various truthy values."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", value)
        assert _is_async_mode_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_returns_false_for_falsy_values(self, monkeypatch, value):
        """Returns False for various falsy values."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", value)
        assert _is_async_mode_enabled() is False


# =============================================================================
# Tests for run_evaluation
# =============================================================================


class TestRunEvaluation:
    """Tests for run_evaluation function."""

    def test_calls_deepeval_evaluate(self, monkeypatch):
        """Calls deepeval.evaluate with test case and metrics."""
        mock_evaluate = MagicMock()

        class _Result:
            test_results = []

        mock_evaluate.return_value = _Result()
        # Must patch at module level where it's imported
        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            mock_evaluate,
        )

        test_case = MagicMock()
        metrics = [MagicMock()]
        run_evaluation(test_case, metrics)

        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        assert test_case in call_args[0][0]  # test_cases list
        assert metrics[0] in call_args[0][1]  # metrics list

    def test_uses_sync_mode_by_default(self, monkeypatch):
        """Uses AsyncConfig(run_async=False) by default."""
        captured_config = {}

        def capture_evaluate(test_cases, metrics, async_config=None, display_config=None):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", raising=False
        )

        run_evaluation(MagicMock(), [MagicMock()])

        assert captured_config["async_config"].run_async is False

    def test_uses_async_mode_when_env_enabled(self, monkeypatch):
        """Uses AsyncConfig(run_async=True) when env var enabled."""
        captured_config = {}

        def capture_evaluate(test_cases, metrics, async_config=None, display_config=None):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "true")

        run_evaluation(MagicMock(), [MagicMock()])

        assert captured_config["async_config"].run_async is True

    def test_use_async_parameter_overrides_env(self, monkeypatch):
        """use_async parameter overrides environment variable."""
        captured_config = {}

        def capture_evaluate(test_cases, metrics, async_config=None, display_config=None):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "false")

        # Override with use_async=True
        run_evaluation(MagicMock(), [MagicMock()], use_async=True)

        assert captured_config["async_config"].run_async is True

    def test_debug_log_called_for_stdout(self, monkeypatch):
        """Calls debug_log for captured stdout."""
        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            lambda *a, **kw: type("R", (), {"test_results": []})(),
        )

        debug_log = MagicMock()
        # Run evaluation with debug_log - should not crash
        run_evaluation(MagicMock(), [MagicMock()], debug_log=debug_log)

        # debug_log may or may not be called depending on captured output
        # Just verify it doesn't crash


# =============================================================================
# Tests for run_evaluation_async
# =============================================================================


class TestRunEvaluationAsync:
    """Tests for run_evaluation_async function."""

    def test_is_coroutine_function(self):
        """run_evaluation_async is a coroutine function."""
        assert asyncio.iscoroutinefunction(run_evaluation_async)

    def test_calls_deepeval_with_async_config(self, monkeypatch):
        """Calls deepeval.evaluate with AsyncConfig(run_async=True)."""
        captured_config = {}

        def capture_evaluate(test_cases, metrics, async_config=None, display_config=None):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )

        async def run_test():
            return await run_evaluation_async(MagicMock(), [MagicMock()])

        asyncio.run(run_test())

        assert captured_config["async_config"].run_async is True

    def test_returns_evaluation_result(self, monkeypatch):
        """Returns the evaluation result from deepeval."""

        class _ExpectedResult:
            test_results = [{"metric": "test"}]

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            lambda *a, **kw: _ExpectedResult(),
        )

        async def run_test():
            return await run_evaluation_async(MagicMock(), [MagicMock()])

        result = asyncio.run(run_test())
        assert result.test_results == [{"metric": "test"}]

    def test_handles_debug_log(self, monkeypatch):
        """Handles debug_log parameter without errors."""
        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            lambda *a, **kw: type("R", (), {"test_results": []})(),
        )

        debug_log = MagicMock()

        async def run_test():
            return await run_evaluation_async(
                MagicMock(), [MagicMock()], debug_log=debug_log
            )

        # Should not raise
        asyncio.run(run_test())

    def test_can_run_multiple_concurrent_evaluations(self, monkeypatch):
        """Can run multiple evaluations concurrently."""
        call_count = {"count": 0}

        def counting_evaluate(test_cases, metrics, async_config=None, display_config=None):
            call_count["count"] += 1

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            counting_evaluate,
        )

        async def run_multiple():
            tasks = [
                run_evaluation_async(MagicMock(), [MagicMock()])
                for _ in range(3)
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_multiple())

        assert len(results) == 3
        assert call_count["count"] == 3

