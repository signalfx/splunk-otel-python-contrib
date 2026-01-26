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
    """Install deepeval stubs if not available.

    This stub must include all submodules that the package imports:
    - deepeval.metrics (for metric classes)
    - deepeval.test_case (for LLMTestCase)
    - deepeval.evaluate.configs (for AsyncConfig, DisplayConfig)
    """
    if "deepeval" in sys.modules:
        return
    try:
        __import__("deepeval")
        return
    except Exception:
        pass

    root = types.ModuleType("deepeval")
    metrics_mod = types.ModuleType("deepeval.metrics")
    test_case_mod = types.ModuleType("deepeval.test_case")
    eval_cfg_mod = types.ModuleType("deepeval.evaluate.configs")

    # Stub metric classes
    class BiasMetric:
        _required_params = []

        def __init__(self, **kwargs):
            self.name = "bias"
            self.score = 0.5
            self.success = True
            self.threshold = kwargs.get("threshold", 0.5)
            self.reason = "stub"

    class ToxicityMetric(BiasMetric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "toxicity"

    class AnswerRelevancyMetric(BiasMetric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "answer_relevancy"

    class FaithfulnessMetric(BiasMetric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "faithfulness"

    class GEval:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "geval")
            self.score = 0.0
            self.threshold = 0.0
            self.success = True
            self.reason = None

    metrics_mod.BiasMetric = BiasMetric
    metrics_mod.ToxicityMetric = ToxicityMetric
    metrics_mod.AnswerRelevancyMetric = AnswerRelevancyMetric
    metrics_mod.FaithfulnessMetric = FaithfulnessMetric
    metrics_mod.GEval = GEval

    # Stub test case classes
    class LLMTestCaseParams:
        INPUT_OUTPUT = "io"
        INPUT = "input"
        ACTUAL_OUTPUT = "actual_output"

    class LLMTestCase:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.retrieval_context = kwargs.get("retrieval_context")

    test_case_mod.LLMTestCaseParams = LLMTestCaseParams
    test_case_mod.LLMTestCase = LLMTestCase

    # Stub config classes
    class AsyncConfig:
        def __init__(self, run_async=False, max_concurrent=None):
            self.run_async = run_async
            self.max_concurrent = max_concurrent

    class DisplayConfig:
        def __init__(self, show_indicator=False, print_results=False):
            self.show_indicator = show_indicator
            self.print_results = print_results

    eval_cfg_mod.AsyncConfig = AsyncConfig
    eval_cfg_mod.DisplayConfig = DisplayConfig

    # Stub evaluate function
    class _EvalResult:
        test_results = []

    def evaluate(test_cases, metrics, async_config=None, display_config=None):
        return _EvalResult()

    root.evaluate = evaluate

    # Register all modules
    sys.modules["deepeval"] = root
    sys.modules["deepeval.metrics"] = metrics_mod
    sys.modules["deepeval.test_case"] = test_case_mod
    sys.modules["deepeval.evaluate"] = root
    sys.modules["deepeval.evaluate.configs"] = eval_cfg_mod


_install_deepeval_stubs()


from opentelemetry.util.evaluator.deepeval_runner import (
    run_evaluation,
    run_evaluation_async,
)
from opentelemetry.util.genai.evals.env import (
    read_concurrent_flag,
    read_max_concurrent,
)

# =============================================================================
# Tests for read_concurrent_flag (moved to env module)
# =============================================================================


class TestReadConcurrentFlag:
    """Tests for read_concurrent_flag helper function."""

    def test_returns_false_when_not_set(self, monkeypatch):
        """Returns False when env var not set."""
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", raising=False
        )
        assert read_concurrent_flag() is False

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
    def test_returns_true_for_truthy_values(self, monkeypatch, value):
        """Returns True for various truthy values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", value
        )
        assert read_concurrent_flag() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_returns_false_for_falsy_values(self, monkeypatch, value):
        """Returns False for various falsy values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", value
        )
        assert read_concurrent_flag() is False


# =============================================================================
# Tests for read_max_concurrent
# =============================================================================


class TestReadMaxConcurrent:
    """Tests for read_max_concurrent helper function."""

    def test_returns_default_when_not_set(self, monkeypatch):
        """Returns default (10) when env var not set."""
        monkeypatch.delenv("DEEPEVAL_MAX_CONCURRENT", raising=False)
        assert read_max_concurrent() == 10

    def test_returns_custom_value(self, monkeypatch):
        """Returns custom value when set."""
        monkeypatch.setenv("DEEPEVAL_MAX_CONCURRENT", "20")
        assert read_max_concurrent() == 20

    def test_clamps_to_minimum(self, monkeypatch):
        """Clamps to minimum of 1."""
        monkeypatch.setenv("DEEPEVAL_MAX_CONCURRENT", "0")
        assert read_max_concurrent() == 1

    def test_clamps_to_maximum(self, monkeypatch):
        """Clamps to maximum of 50."""
        monkeypatch.setenv("DEEPEVAL_MAX_CONCURRENT", "100")
        assert read_max_concurrent() == 50

    def test_returns_default_for_invalid(self, monkeypatch):
        """Returns default for invalid values."""
        monkeypatch.setenv("DEEPEVAL_MAX_CONCURRENT", "invalid")
        assert read_max_concurrent() == 10


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

        def capture_evaluate(
            test_cases, metrics, async_config=None, display_config=None
        ):
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

    def test_always_uses_sync_mode(self, monkeypatch):
        """run_evaluation always uses AsyncConfig(run_async=False).

        The sync function is for sequential evaluation path and should
        never use DeepEval's internal async mode, regardless of env var.
        """
        captured_config = {}

        def capture_evaluate(
            test_cases, metrics, async_config=None, display_config=None
        ):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )
        # Even with concurrent mode enabled, sync run_evaluation should use run_async=False
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "true"
        )

        run_evaluation(MagicMock(), [MagicMock()])

        # Sync path always uses sequential mode
        assert captured_config["async_config"].run_async is False


# =============================================================================
# Tests for run_evaluation_async
# =============================================================================


class TestRunEvaluationAsync:
    """Tests for run_evaluation_async function."""

    def test_is_coroutine_function(self):
        """run_evaluation_async is a coroutine function."""
        assert asyncio.iscoroutinefunction(run_evaluation_async)

    def test_calls_deepeval_with_async_config(self, monkeypatch):
        """Calls deepeval.evaluate with AsyncConfig(run_async=True) when concurrent mode enabled."""
        captured_config = {}

        def capture_evaluate(
            test_cases, metrics, async_config=None, display_config=None
        ):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )
        # Enable concurrent mode to activate DeepEval's async mode
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "true"
        )

        async def run_test():
            return await run_evaluation_async(MagicMock(), [MagicMock()])

        asyncio.run(run_test())

        assert captured_config["async_config"].run_async is True

    def test_async_disabled_when_concurrent_mode_off(self, monkeypatch):
        """AsyncConfig.run_async is False when concurrent mode is disabled."""
        captured_config = {}

        def capture_evaluate(
            test_cases, metrics, async_config=None, display_config=None
        ):
            captured_config["async_config"] = async_config

            class _Result:
                test_results = []

            return _Result()

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval_runner.deepeval_evaluate",
            capture_evaluate,
        )
        # Explicitly disable concurrent mode
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "false"
        )

        async def run_test():
            return await run_evaluation_async(MagicMock(), [MagicMock()])

        asyncio.run(run_test())

        assert captured_config["async_config"].run_async is False

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

    def test_can_run_multiple_concurrent_evaluations(self, monkeypatch):
        """Can run multiple evaluations concurrently."""
        call_count = {"count": 0}

        def counting_evaluate(
            test_cases, metrics, async_config=None, display_config=None
        ):
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
