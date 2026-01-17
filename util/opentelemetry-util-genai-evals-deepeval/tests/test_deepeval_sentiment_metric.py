"""Sentiment metric tests with stubbed deepeval modules.

Dynamic stub injection precedes some imports (violating E402), so we apply a
file-level ignore to prefer clarity over strict ordering.
"""

# ruff: noqa: E402

import importlib

import pytest


class MetricData:  # lightweight stub
    def __init__(
        self,
        *,
        name: str,
        threshold=None,
        success=None,
        score=None,
        reason=None,
        evaluation_model=None,
        evaluation_cost=None,
        verbose_logs=None,
        strict_mode=None,
        error=None,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.success = success
        self.score = score
        self.reason = reason
        self.evaluation_model = evaluation_model
        self.evaluation_cost = evaluation_cost
        self.verbose_logs = verbose_logs
        self.strict_mode = strict_mode
        self.error = error


class TestResult:  # lightweight stub
    def __init__(
        self,
        *,
        name: str,
        success: bool | None,
        metrics_data: list[MetricData],
        conversational: bool = False,
    ) -> None:
        self.name = name
        self.success = success
        self.metrics_data = metrics_data
        self.conversational = conversational


class DeeEvaluationResult:  # stub container
    def __init__(self, *, test_results: list[TestResult], confident_link=None):
        self.test_results = test_results
        self.confident_link = confident_link


# Install deepeval stubs if dependency absent (reuse logic similar to main evaluator tests)
def _install_deepeval_stubs():
    import sys as _sys
    import types

    if "deepeval" in _sys.modules:
        return
    root = types.ModuleType("deepeval")
    metrics_mod = types.ModuleType("deepeval.metrics")
    test_case_mod = types.ModuleType("deepeval.test_case")
    eval_cfg_mod = types.ModuleType("deepeval.evaluate.configs")

    class GEval:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "geval")

    class LLMTestCaseParams:
        ACTUAL_OUTPUT = "actual_output"

    class LLMTestCase:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    metrics_mod.GEval = GEval
    test_case_mod.LLMTestCaseParams = LLMTestCaseParams
    test_case_mod.LLMTestCase = LLMTestCase

    class AsyncConfig:  # noqa: D401
        def __init__(self, run_async=False):
            self.run_async = run_async

    class DisplayConfig:
        def __init__(self, show_indicator=False, print_results=False):
            pass

    eval_cfg_mod.AsyncConfig = AsyncConfig
    eval_cfg_mod.DisplayConfig = DisplayConfig

    def evaluate(
        test_cases,
        metrics,
        async_config=None,
        display_config=None,
    ):
        class _Eval:
            test_results = []

        return _Eval()

    root.evaluate = evaluate
    _sys.modules["deepeval"] = root
    _sys.modules["deepeval.metrics"] = metrics_mod
    _sys.modules["deepeval.test_case"] = test_case_mod
    _sys.modules["deepeval.evaluate"] = root
    _sys.modules["deepeval.evaluate.configs"] = eval_cfg_mod


_install_deepeval_stubs()

from opentelemetry.util.evaluator import deepeval as plugin
from opentelemetry.util.genai.evals.registry import (
    clear_registry,
)
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


def test_sentiment_metric_result_attributes(monkeypatch):
    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("sentiment [geval] [GEval]",), invocation_type="LLMInvocation"
    )

    # Fake deepeval result with a sentiment compound score 0.6 (will be mapped to (0.6+1)/2=0.8 recorded score)
    fake_result = DeeEvaluationResult(
        test_results=[
            TestResult(
                name="case",
                success=True,
                metrics_data=[
                    MetricData(
                        name="sentiment [geval] [GEval]",
                        threshold=0.0,
                        success=True,
                        score=0.6,
                        reason="Positive tone",
                        evaluation_model="gpt-4o-mini",
                        evaluation_cost=0.001,
                    )
                ],
                conversational=False,
            )
        ],
        confident_link=None,
    )

    # Bypass instantiation logic to avoid real deepeval dependency usage
    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
        lambda specs, test_case, model: ([object()], []),
    )
    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._run_deepeval",
        lambda case, metrics, debug_log: fake_result,
    )

    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    res = results[0]
    assert res.metric_name == "sentiment [geval] [GEval]"
    # Recorded score should be mapped to [0,1]
    assert res.score == pytest.approx((0.6 + 1) / 2, rel=1e-6)
    # Distribution attributes should be present
    assert "deepeval.sentiment.neg" in res.attributes
    assert "deepeval.sentiment.neu" in res.attributes
    assert "deepeval.sentiment.pos" in res.attributes
    assert "deepeval.sentiment.compound" in res.attributes
    # Values within valid ranges
    assert 0.0 <= res.attributes["deepeval.sentiment.neg"] <= 1.0
    assert 0.0 <= res.attributes["deepeval.sentiment.neu"] <= 1.0
    assert 0.0 <= res.attributes["deepeval.sentiment.pos"] <= 1.0
    compound = res.attributes["deepeval.sentiment.compound"]
    assert compound == pytest.approx(0.6, rel=1e-6)
    # Normalization check (allow tiny float drift)
    total = (
        res.attributes["deepeval.sentiment.neg"]
        + res.attributes["deepeval.sentiment.neu"]
        + res.attributes["deepeval.sentiment.pos"]
    )
    assert abs(total - 1.0) < 1e-6
