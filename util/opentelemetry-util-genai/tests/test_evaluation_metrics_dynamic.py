from __future__ import annotations

from typing import Any, Dict, List

from opentelemetry.util.genai.emitters.evaluation import (
    EvaluationMetricsEmitter,
)
from opentelemetry.util.genai.types import (
    Error,
    EvaluationResult,
    LLMInvocation,
)


class _RecordingHistogram:
    def __init__(self, name: str) -> None:
        self.name = name
        self.points: List[tuple[float, Dict[str, Any], Any]] = []

    def record(
        self, value: float, *, attributes: Dict[str, Any], context: Any = None
    ):
        self.points.append((value, attributes, context))


class _HistogramFactory:
    """Factory that creates separate histograms per metric (legacy mode)."""

    def __init__(self) -> None:
        self.created: Dict[str, _RecordingHistogram] = {}

    def __call__(self, metric_name: str):
        # Canonical instruments now: gen_ai.evaluation.<metric>
        full = f"gen_ai.evaluation.{metric_name}"
        if full not in self.created:
            self.created[full] = _RecordingHistogram(full)
        return self.created[full]


class _SingleHistogramFactory:
    """Factory that returns a single histogram for all metrics (single metric mode)."""

    def __init__(self) -> None:
        self.created: Dict[str, _RecordingHistogram] = {}
        self._single_histogram: _RecordingHistogram | None = None

    def __call__(self, metric_name: str):
        # Return the same histogram for all metrics
        if self._single_histogram is None:
            self._single_histogram = _RecordingHistogram(
                "gen_ai.evaluation.score"
            )
            self.created["gen_ai.evaluation.score"] = self._single_histogram
        return self._single_histogram


def test_dynamic_metric_histograms_created_per_metric():
    factory = _HistogramFactory()
    emitter = EvaluationMetricsEmitter(factory)
    invocation = LLMInvocation(request_model="gpt-test")
    results = [
        EvaluationResult(metric_name="bias", score=0.5),
        EvaluationResult(metric_name="toxicity", score=0.1),
        EvaluationResult(metric_name="bias", score=0.75, label="medium"),
    ]

    emitter.on_evaluation_results(results, invocation)

    # Ensure two canonical histograms were created
    assert set(factory.created.keys()) == {
        "gen_ai.evaluation.bias",
        "gen_ai.evaluation.toxicity",
    }

    bias_hist = factory.created["gen_ai.evaluation.bias"]
    tox_hist = factory.created["gen_ai.evaluation.toxicity"]

    # Bias scores recorded twice
    bias_points = [p[0] for p in bias_hist.points]
    assert bias_points == [0.5, 0.75]

    # Toxicity once
    tox_points = [p[0] for p in tox_hist.points]
    assert tox_points == [0.1]

    # Attribute propagation
    for _, attrs, _ in bias_hist.points + tox_hist.points:
        assert attrs["gen_ai.operation.name"] == "evaluation"
        assert attrs["gen_ai.evaluation.name"] in {"bias", "toxicity"}
    # label only present for second bias result
    labels = [
        attrs.get("gen_ai.evaluation.score.label")
        for _, attrs, _ in bias_hist.points
    ]
    assert labels == [None, "medium"]
    # gen_ai.evaluation.passed derivation only when label clearly indicates pass/fail; 'medium' should not set it
    passed_vals = [
        attrs.get("gen_ai.evaluation.passed")
        for _, attrs, _ in bias_hist.points
    ]
    assert passed_vals == [None, None]
    # Units should be set for each point
    for _, attrs, _ in bias_hist.points + tox_hist.points:
        assert attrs.get("gen_ai.evaluation.score.units") == "score"


def test_single_metric_mode_all_evaluations_to_one_histogram():
    """Test that single metric mode uses one histogram with gen_ai.evaluation.name attribute."""
    factory = _SingleHistogramFactory()
    emitter = EvaluationMetricsEmitter(factory)
    invocation = LLMInvocation(request_model="gpt-test")
    results = [
        EvaluationResult(metric_name="bias", score=0.5),
        EvaluationResult(metric_name="toxicity", score=0.1),
        EvaluationResult(metric_name="bias", score=0.75, label="medium"),
        EvaluationResult(metric_name="sentiment", score=0.8),
    ]

    emitter.on_evaluation_results(results, invocation)

    # Ensure only one histogram was created
    assert set(factory.created.keys()) == {"gen_ai.evaluation.score"}

    score_hist = factory.created["gen_ai.evaluation.score"]

    # All scores should be recorded to the same histogram
    assert len(score_hist.points) == 4
    scores = [p[0] for p in score_hist.points]
    assert scores == [0.5, 0.1, 0.75, 0.8]

    # Check that gen_ai.evaluation.name attribute distinguishes the evaluation types
    eval_names = [p[1]["gen_ai.evaluation.name"] for p in score_hist.points]
    assert eval_names == ["bias", "toxicity", "bias", "sentiment"]

    # Attribute propagation - single metric mode should NOT have gen_ai.operation.name
    for _, attrs, _ in score_hist.points:
        assert (
            "gen_ai.operation.name" not in attrs
        )  # Should not be present in single metric mode
        assert "gen_ai.evaluation.name" in attrs
        assert (
            "gen_ai.evaluation.score.units" not in attrs
        )  # Should not be present in single metric mode

    # Check label propagation
    labels = [
        p[1].get("gen_ai.evaluation.score.label") for p in score_hist.points
    ]
    assert labels == [None, None, "medium", None]


def test_evaluation_metrics_emitter_error_event_attributes():
    factory = _HistogramFactory()
    emitter = EvaluationMetricsEmitter(factory)
    invocation = LLMInvocation(request_model="gpt-test")
    # Simulate an error result
    error_result = EvaluationResult(
        metric_name="toxicity",
        score=0.0,
        error=Error(
            message="Model failed to evaluate toxicity",
            type=RuntimeError,
        ),
    )

    emitter.on_evaluation_results([error_result], invocation)

    tox_hist = factory.created["gen_ai.evaluation.toxicity"]
    assert len(tox_hist.points) == 1
    _, attrs, _ = tox_hist.points[0]

    # Check required error attributes per semantic conventions
    assert attrs["gen_ai.operation.name"] == "evaluation"
    assert attrs["gen_ai.evaluation.name"] == "toxicity"
    assert attrs["error.message"] == "Model failed to evaluate toxicity"
    assert attrs["error.type"] == "RuntimeError"
    # Units should still be set
    assert attrs.get("gen_ai.evaluation.score.units") == "score"
    # Request model should be present
    assert attrs.get("gen_ai.request.model") == "gpt-test"
