from __future__ import annotations

from typing import Any, Dict, List

from opentelemetry.util.genai.emitters.evaluation import (
    EvaluationEventsEmitter,
    EvaluationMetricsEmitter,
)
from opentelemetry.util.genai.types import EvaluationResult, LLMInvocation


class _RecordingHistogram:
    def __init__(self) -> None:
        self.records: List[tuple[float, Dict[str, Any], Any]] = []

    def record(
        self,
        value: float,
        *,
        attributes: Dict[str, Any],
        context: Any = None,
    ):  # type: ignore[override]
        self.records.append((value, attributes, context))


class _HistogramFactory:
    def __init__(self) -> None:
        self.created: Dict[str, _RecordingHistogram] = {}

    def __call__(self, metric_name: str):
        if metric_name not in self.created:
            self.created[metric_name] = _RecordingHistogram()
        return self.created[metric_name]


class _RecordingEventLogger:
    def __init__(self) -> None:
        self.records: List[Any] = []

    def emit(self, event: Any) -> None:
        self.records.append(event)


def test_metric_name_canonicalization_variants():
    inv = LLMInvocation(request_model="gpt-test")
    factory = _HistogramFactory()
    metrics_emitter = EvaluationMetricsEmitter(factory)
    events_logger = _RecordingEventLogger()
    events_emitter = EvaluationEventsEmitter(events_logger)

    # Provide several variant names including spaces, underscores, brackets, suffixes
    results = [
        EvaluationResult(metric_name="answer relevancy", score=1.0),
        EvaluationResult(metric_name="answer_relevancy", score=0.9),
        EvaluationResult(
            metric_name="faithfulness", score=0.8
        ),  # legacy synonym -> hallucination
        EvaluationResult(metric_name="hallucination [geval]", score=0.95),
        EvaluationResult(metric_name="Hallucination-GEval", score=0.96),
        EvaluationResult(metric_name="bias", score=0.5),
    ]

    metrics_emitter.on_evaluation_results(results, inv)
    events_emitter.on_evaluation_results(results, inv)

    # Histograms created only for canonical names
    assert set(factory.created.keys()) == {
        "relevance",
        "hallucination",
        "bias",
    }

    # Relevance histogram should have two points
    rel_points = factory.created["relevance"].records
    assert len(rel_points) == 2
    # Hallucination histogram should have three points (faithfulness + 2 variants)
    hall_points = factory.created["hallucination"].records
    assert len(hall_points) == 3
    bias_points = factory.created["bias"].records
    assert len(bias_points) == 1

    # Events: ensure canonical names appear
    names = [
        e.attributes["gen_ai.evaluation.name"] for e in events_logger.records
    ]
    # We should have 6 events total
    assert len(names) == 6
    assert names.count("relevance") == 2
    assert names.count("hallucination") == 3
    assert names.count("bias") == 1
