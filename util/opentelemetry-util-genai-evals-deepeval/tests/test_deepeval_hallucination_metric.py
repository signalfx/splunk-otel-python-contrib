"""Hallucination metric tests with stubbed deepeval modules.

Dynamic stub injection precedes some imports (violating E402), so we apply a
file-level ignore to prefer clarity over strict ordering.
"""

# ruff: noqa: E402

import importlib
import sys
import types

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


class FakeTestResult:  # lightweight stub (renamed from TestResult to avoid pytest collection)
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
    def __init__(
        self, *, test_results: list[FakeTestResult], confident_link=None
    ):
        self.test_results = test_results
        self.confident_link = confident_link


# Install deepeval stubs if dependency absent (reuse logic similar to main evaluator tests)
def _install_deepeval_stubs():
    if "deepeval" in sys.modules:
        return
    root = types.ModuleType("deepeval")
    metrics_mod = types.ModuleType("deepeval.metrics")
    test_case_mod = types.ModuleType("deepeval.test_case")
    eval_cfg_mod = types.ModuleType("deepeval.evaluate.configs")

    class GEval:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "geval")

    class LLMTestCaseParams:
        INPUT_OUTPUT = "io"
        INPUT = "input"
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
    sys.modules["deepeval"] = root
    sys.modules["deepeval.metrics"] = metrics_mod
    sys.modules["deepeval.test_case"] = test_case_mod
    sys.modules["deepeval.evaluate"] = root
    sys.modules["deepeval.evaluate.configs"] = eval_cfg_mod


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
    inv = LLMInvocation(request_model="hallucination-model")
    inv.input_messages.append(
        InputMessage(
            role="user", parts=[Text(content="What is the capital of France?")]
        )
    )
    inv.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="The capital of France is Paris.")],
            finish_reason="stop",
        )
    )
    return inv


def test_hallucination_metric_score_inversion(monkeypatch):
    """Test that hallucination GEval scores are inverted correctly.

    GEval uses higher=better (1.0=no hallucination), but final score should be
    lower=better (0.0=no hallucination) to match industry standard.
    """
    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("hallucination",), invocation_type="LLMInvocation"
    )

    test_cases = [
        # (GEval score, expected final score, expected label)
        (1.0, 0.0, "Not Hallucinated"),  # Perfect: GEval 1.0 → final 0.0
        (0.9, 0.1, "Not Hallucinated"),  # Very good: GEval 0.9 → final 0.1
        (0.7, 0.3, "Not Hallucinated"),  # Good: GEval 0.7 → final 0.3
        (
            0.5,
            0.5,
            None,
        ),  # Moderate: GEval 0.5 → final 0.5 (label determined by success)
        (0.3, 0.7, "Hallucinated"),  # Bad: GEval 0.3 → final 0.7
        (0.0, 1.0, "Hallucinated"),  # Worst: GEval 0.0 → final 1.0
    ]

    for geval_score, expected_final_score, expected_label in test_cases:
        # Determine success based on threshold (0.7): GEval score >= 0.7 passes
        success = geval_score >= 0.7

        fake_result = DeeEvaluationResult(
            test_results=[
                FakeTestResult(
                    name="case",
                    success=success,
                    metrics_data=[
                        MetricData(
                            name="hallucination",
                            threshold=0.7,
                            success=success,
                            score=geval_score,
                            reason=f"GEval score {geval_score}",
                            evaluation_model="gpt-4o-mini",
                            evaluation_cost=0.001,
                        )
                    ],
                    conversational=False,
                )
            ],
            confident_link=None,
        )

        # Use a closure to capture the current fake_result
        def make_fake_runner(result):
            def fake_runner(case, metrics):
                return result

            return fake_runner

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
            lambda specs, test_case, model: ([object()], []),
        )
        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval._run_deepeval",
            make_fake_runner(fake_result),
        )

        results = evaluator.evaluate(invocation)
        assert len(results) == 1
        res = results[0]
        assert res.metric_name == "hallucination"
        # Verify score inversion: GEval score 1.0 → final score 0.0
        assert res.score == pytest.approx(expected_final_score, rel=1e-6), (
            f"GEval score {geval_score} should invert to final score {expected_final_score}"
        )
        # Verify original GEval score is stored in attributes
        assert "deepeval.hallucination.geval_score" in res.attributes
        assert res.attributes[
            "deepeval.hallucination.geval_score"
        ] == pytest.approx(geval_score, rel=1e-6)
        # Verify label if expected
        if expected_label is not None:
            assert res.label == expected_label, (
                f"Final score {expected_final_score} (from GEval {geval_score}) "
                f"should be labeled '{expected_label}'"
            )
        # Verify threshold is stored
        assert res.attributes["deepeval.threshold"] == 0.7
        assert res.attributes["deepeval.success"] == success


def test_hallucination_metric_name_variants(monkeypatch):
    """Test that hallucination post-processing works with different name variants."""
    invocation = _build_invocation()

    name_variants = [
        "hallucination",
        "hallucination [geval]",
        "hallucination [geval] [GEval]",
    ]

    for name_variant in name_variants:
        evaluator = plugin.DeepevalEvaluator(
            (name_variant,), invocation_type="LLMInvocation"
        )

        # GEval score 0.9 should invert to final score 0.1
        fake_result = DeeEvaluationResult(
            test_results=[
                FakeTestResult(
                    name="case",
                    success=True,
                    metrics_data=[
                        MetricData(
                            name=name_variant,
                            threshold=0.7,
                            success=True,
                            score=0.9,
                            reason="Test name variant",
                        )
                    ],
                    conversational=False,
                )
            ],
            confident_link=None,
        )

        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
            lambda specs, test_case, model: ([object()], []),
        )
        monkeypatch.setattr(
            "opentelemetry.util.evaluator.deepeval._run_deepeval",
            lambda case, metrics: fake_result,
        )

        results = evaluator.evaluate(invocation)
        assert len(results) == 1
        res = results[0]
        assert res.metric_name == name_variant
        # Verify inversion works for all name variants
        assert res.score == pytest.approx(0.1, rel=1e-6), (
            f"Name variant '{name_variant}' should invert GEval score 0.9 to final score 0.1"
        )
        assert res.attributes[
            "deepeval.hallucination.geval_score"
        ] == pytest.approx(0.9, rel=1e-6)
