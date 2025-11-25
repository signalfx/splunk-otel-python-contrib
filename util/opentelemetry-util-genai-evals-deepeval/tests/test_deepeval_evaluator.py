"""Tests for Deepeval evaluator (uses local stubs when dependency absent).

We intentionally perform dynamic stub installation before importing the plugin
module, which violates Ruff's E402 (module level import not at top). A file-
level ignore is used to keep the logical setup order clear.
"""

# ruff: noqa: E402

import importlib
import os
import sys
from unittest.mock import patch

import pytest


# Provide stub 'deepeval' package structure if dependency is unavailable.
def _install_deepeval_stubs():
    if os.getenv("OTEL_TEST_USE_REAL_DEEPEVAL") == "1":
        try:
            import importlib as _il  # noqa: F401

            __import__("deepeval")  # pragma: no cover
            return
        except Exception:
            pass
    # Ensure any previously imported real modules are cleared so our stubs take effect.
    for name in list(sys.modules):
        if name == "deepeval" or name.startswith("deepeval."):
            sys.modules.pop(name, None)
    import types

    root = types.ModuleType("deepeval")
    metrics_mod = types.ModuleType("deepeval.metrics")
    test_case_mod = types.ModuleType("deepeval.test_case")
    eval_cfg_mod = types.ModuleType("deepeval.evaluate.configs")
    models_root_mod = types.ModuleType("deepeval.models")
    models_base_mod = types.ModuleType("deepeval.models.base_model")

    class _ReqParam:
        def __init__(self, value):
            self.value = value

    class GEval:  # minimal constructor compatibility
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "geval")
            self.score = kwargs.get("score", 0.0)
            self.threshold = kwargs.get("threshold", 0.0)
            self.success = True
            self.reason = None

    class BiasMetric:
        _required_params = []

        def __init__(self, **kwargs):
            self.name = "bias"
            self.score = 0.5
            self.success = True
            self.threshold = kwargs.get("threshold", 0.5)
            self.reason = "stub bias"

    class ToxicityMetric(BiasMetric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "toxicity"
            self.reason = "stub toxicity"

    class AnswerRelevancyMetric(BiasMetric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "answer_relevancy"
            self.reason = "stub answer relevancy"

    class FaithfulnessMetric(BiasMetric):
        _required_params = [_ReqParam("retrieval_context")]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "faithfulness"
            self.reason = "stub faithfulness"

    metrics_mod.GEval = GEval
    metrics_mod.BiasMetric = BiasMetric
    metrics_mod.ToxicityMetric = ToxicityMetric
    metrics_mod.AnswerRelevancyMetric = AnswerRelevancyMetric
    metrics_mod.FaithfulnessMetric = FaithfulnessMetric

    class DeepEvalBaseLLM:  # minimal behaviour for registry tests
        def load_model(self):
            return self

        def generate(self, prompt: str):  # pragma: no cover - unused
            raise NotImplementedError

        async def a_generate(self, prompt: str):  # pragma: no cover - unused
            raise NotImplementedError

        def get_model_name(self):  # pragma: no cover - unused
            return "stub"

    models_base_mod.DeepEvalBaseLLM = DeepEvalBaseLLM
    models_root_mod.base_model = models_base_mod

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

    class AsyncConfig:
        def __init__(self, run_async=False):
            self.run_async = run_async

    class DisplayConfig:
        def __init__(self, show_indicator=False, print_results=False):
            self.show_indicator = show_indicator
            self.print_results = print_results

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
    sys.modules["deepeval.evaluate"] = root  # simplify
    sys.modules["deepeval.evaluate.configs"] = eval_cfg_mod
    sys.modules["deepeval.models"] = models_root_mod
    sys.modules["deepeval.models.base_model"] = models_base_mod


_install_deepeval_stubs()


# Local lightweight stand-ins to avoid requiring the real 'deepeval' package
class MetricData:  # type: ignore[override]
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


class TestResult:  # type: ignore[override]
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


class DeeEvaluationResult:  # type: ignore[override]
    def __init__(self, *, test_results: list[TestResult], confident_link=None):
        self.test_results = test_results
        self.confident_link = confident_link


from opentelemetry.util.evaluator import deepeval as plugin
from opentelemetry.util.evaluator.deepeval_models import (
    clear_models as clear_model_registry,
)
from opentelemetry.util.evaluator.deepeval_models import (
    register_model,
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


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    importlib.reload(plugin)
    plugin.register()
    clear_model_registry()
    yield
    clear_registry()
    clear_model_registry()


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


def test_registration_adds_deepeval() -> None:
    names = list_evaluators()
    assert "deepeval" in names


def test_default_metrics_covered() -> None:
    evaluator = get_evaluator("deepeval")
    assert set(m.lower() for m in evaluator.metrics) == {
        "bias",
        "toxicity",
        "answer_relevancy",
        "hallucination",
        "sentiment",
    }


def test_evaluator_converts_results(monkeypatch):
    invocation = _build_invocation()
    evaluator = get_evaluator(
        "deepeval",
        ("bias",),
        invocation_type="LLMInvocation",
    )

    fake_result = DeeEvaluationResult(
        test_results=[
            TestResult(
                name="case",
                success=True,
                metrics_data=[
                    MetricData(
                        name="bias",
                        threshold=0.7,
                        success=True,
                        score=0.8,
                        reason="looks good",
                        evaluation_model="gpt-4o-mini",
                        evaluation_cost=0.01,
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
        lambda case, metrics, debug_log: fake_result,
    )

    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    result = results[0]
    assert result.metric_name == "bias"
    assert result.score == 0.8
    assert result.label == "Not Biased"
    assert result.explanation == "looks good"
    assert result.attributes["deepeval.threshold"] == 0.7
    assert result.attributes["deepeval.success"] is True


def test_metric_options_coercion(monkeypatch):
    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("bias",),
        invocation_type="LLMInvocation",
        options={"bias": {"threshold": "0.9", "strict_mode": "true"}},
    )

    captured = {}

    def fake_instantiate(specs, test_case, model):
        captured.update(specs[0].options)
        return [object()], []

    fake_result = DeeEvaluationResult(
        test_results=[
            TestResult(
                name="case",
                success=False,
                metrics_data=[
                    MetricData(
                        name="bias",
                        threshold=0.9,
                        success=False,
                        score=0.1,
                        reason="too biased",
                    )
                ],
                conversational=False,
            )
        ],
        confident_link=None,
    )

    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
        fake_instantiate,
    )
    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._run_deepeval",
        lambda case, metrics, debug_log: fake_result,
    )

    results = evaluator.evaluate(invocation)
    assert captured["threshold"] == 0.9
    assert captured["strict_mode"] is True
    assert captured.get("model", evaluator._default_model()) == "gpt-4o-mini"
    assert results[0].label == "Biased"


def test_evaluator_handles_instantiation_error(monkeypatch):
    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("bias",), invocation_type="LLMInvocation"
    )

    def boom(specs, test_case, model):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._instantiate_metrics", boom
    )

    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    assert results[0].error is not None


def test_custom_deepeval_model_registry(monkeypatch):
    import sys

    models_mod = sys.modules["deepeval.models.base_model"]

    class DummyModel(models_mod.DeepEvalBaseLLM):  # type: ignore[attr-defined]
        def __init__(self):
            self._loaded = False

        def load_model(self):
            self._loaded = True
            return self

        def generate(self, prompt: str) -> str:
            self.load_model()
            return "stubbed"

        async def a_generate(
            self, prompt: str
        ) -> str:  # pragma: no cover - sync used
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return "dummy"

    register_model("custom-circuit", DummyModel)

    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("bias",), invocation_type="LLMInvocation"
    )

    captured = {}

    def fake_instantiate(specs, test_case, model):
        captured["model"] = model
        return [object()], []

    monkeypatch.setenv("DEEPEVAL_MODEL", "custom-circuit")
    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
        fake_instantiate,
    )
    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._run_deepeval",
        lambda case, metrics, debug_log: DeeEvaluationResult(test_results=[]),
    )

    evaluator.evaluate(invocation)
    assert "model" in captured
    assert isinstance(captured["model"], DummyModel)


def test_evaluator_missing_output(monkeypatch):
    invocation = LLMInvocation(request_model="abc")
    evaluator = plugin.DeepevalEvaluator(
        ("bias",), invocation_type="LLMInvocation"
    )
    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    assert results[0].error is not None


def test_dependency_missing(monkeypatch):
    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("bias",), invocation_type="LLMInvocation"
    )
    # Patch all deepeval modules to ensure import_module("deepeval.metrics") fails
    # The test file installs stub modules, so we must patch submodules too
    with patch.dict(
        sys.modules,
        {
            "deepeval": None,
            "deepeval.metrics": None,
            "deepeval.test_case": None,
        },
    ):
        results = evaluator.evaluate(invocation)
    assert len(results) == 1
    assert results[0].error is not None


def test_faithfulness_skipped_without_retrieval_context():
    invocation = _build_invocation()
    evaluator = plugin.DeepevalEvaluator(
        ("faithfulness",),
        invocation_type="LLMInvocation",
    )
    results = evaluator.evaluate(invocation)
    assert len(results) == 1
    result = results[0]
    assert result.label == "skipped"
    assert result.error is not None
    assert "retrieval_context" in (result.explanation or "")
    assert result.attributes.get("deepeval.skipped") is True


def test_retrieval_context_extracted_from_attributes(monkeypatch):
    invocation = _build_invocation()
    invocation.attributes["retrieval_context"] = [
        {"content": "doc1"},
        "doc2",
    ]
    evaluator = plugin.DeepevalEvaluator(
        ("faithfulness",),
        invocation_type="LLMInvocation",
    )

    captured = {}

    def fake_instantiate(specs, test_case, model):
        captured["retrieval_context"] = getattr(
            test_case, "retrieval_context", None
        )
        return ([object()], [])

    fake_result = DeeEvaluationResult(
        test_results=[
            TestResult(
                name="case",
                success=True,
                metrics_data=[
                    MetricData(
                        name="faithfulness",
                        threshold=0.5,
                        success=True,
                        score=0.95,
                        reason="faithful",
                    )
                ],
                conversational=False,
            )
        ],
        confident_link=None,
    )

    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._instantiate_metrics",
        fake_instantiate,
    )
    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval._run_deepeval",
        lambda case, metrics, debug_log: fake_result,
    )

    results = evaluator.evaluate(invocation)
    assert captured["retrieval_context"] == ["doc1", "doc2"]
    assert results[0].metric_name == "faithfulness"
