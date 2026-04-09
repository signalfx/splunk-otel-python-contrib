"""Tests for evaluation monitoring metrics (duration + cost histograms)."""

from __future__ import annotations

import time

from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.util.genai.emitters.evaluation import (
    EvaluationMonitoringEmitter,
)
from opentelemetry.util.genai.evals.base import Evaluator
from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.instruments import Instruments
from opentelemetry.util.genai.types import (
    EvaluationResult,
    GenAI,
    LLMInvocation,
)

_DURATION_METRIC = "gen_ai.evaluation.client.operation.duration"
_COST_METRIC = "gen_ai.evaluation.client.usage.cost"


class _StubHandler:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def evaluation_results(self, invocation, results) -> None:
        self.calls.append((invocation, list(results)))


def _make_meter_and_reader() -> tuple[SDKMeterProvider, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = SDKMeterProvider(metric_readers=[reader])
    return provider, reader


def _get_metric_data(reader: InMemoryMetricReader, metric_name: str):
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return None
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == metric_name:
                    return metric
    return None


# ============================================================================
# Unit: EvaluationMonitoringEmitter records duration from EvaluationResult
# ============================================================================


def _make_emitter(meter):
    instruments = Instruments(meter)
    return EvaluationMonitoringEmitter(
        duration_histogram=instruments.evaluation_client_operation_duration,
        cost_histogram=instruments.evaluation_client_usage_cost,
    )


def test_emitter_records_duration():
    """EvaluationMonitoringEmitter reads duration_s from EvaluationResult."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    results = [
        EvaluationResult(
            metric_name="bias",
            score=0.1,
            duration_s=1.5,
            evaluator_name="MyEvaluator",
        ),
    ]
    emitter.on_evaluation_results(results, None)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _DURATION_METRIC)

    assert metric is not None
    assert metric.unit == "s"
    data_points = list(metric.data.data_points)
    assert len(data_points) == 1
    assert data_points[0].sum == 1.5
    assert data_points[0].count == 1
    assert (
        dict(data_points[0].attributes)["gen_ai.evaluation.evaluator"]
        == "MyEvaluator"
    )


def test_emitter_skips_results_without_duration():
    """Results without duration_s are ignored by the monitoring emitter."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    results = [
        EvaluationResult(metric_name="bias", score=0.1),
    ]
    emitter.on_evaluation_results(results, None)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _DURATION_METRIC)
    assert metric is None


def test_emitter_multiple_evaluators():
    """Each result with different evaluator_name produces separate data points."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    results = [
        EvaluationResult(
            metric_name="bias",
            score=0.5,
            duration_s=0.5,
            evaluator_name="EvalA",
        ),
        EvaluationResult(
            metric_name="toxicity",
            score=0.3,
            duration_s=0.8,
            evaluator_name="EvalB",
        ),
    ]
    emitter.on_evaluation_results(results, None)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _DURATION_METRIC)

    assert metric is not None
    data_points = list(metric.data.data_points)
    assert len(data_points) == 2

    evaluator_names = {
        dict(dp.attributes)["gen_ai.evaluation.evaluator"]
        for dp in data_points
    }
    assert evaluator_names == {"EvalA", "EvalB"}


# ============================================================================
# Unit: EvaluationMonitoringEmitter records evaluation cost
# ============================================================================


def test_emitter_records_cost():
    """EvaluationMonitoringEmitter records evaluation_cost from EvaluationResult."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    results = [
        EvaluationResult(
            metric_name="bias",
            score=0.1,
            duration_s=1.5,
            evaluator_name="MyEvaluator",
            evaluation_cost=0.0023,
        ),
    ]
    emitter.on_evaluation_results(results, None)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _COST_METRIC)

    assert metric is not None
    assert metric.unit == "{usd}"
    data_points = list(metric.data.data_points)
    assert len(data_points) == 1
    assert data_points[0].sum >= 0.002  # float comparison
    assert data_points[0].count == 1
    assert (
        dict(data_points[0].attributes)["gen_ai.evaluation.evaluator"]
        == "MyEvaluator"
    )
    assert dict(data_points[0].attributes)["gen_ai.evaluation.name"] == "bias"


def test_emitter_skips_results_without_cost():
    """Results without evaluation_cost do not produce cost metric data points."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    results = [
        EvaluationResult(
            metric_name="bias",
            score=0.1,
            duration_s=1.5,
            evaluator_name="MyEvaluator",
        ),
    ]
    emitter.on_evaluation_results(results, None)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _COST_METRIC)
    assert metric is None


def test_emitter_records_cost_with_invocation_attrs():
    """Cost metric carries shared attributes from invocation."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    invocation = LLMInvocation(request_model="gpt-4o", provider="azure")
    invocation.agent_name = "coordinator"

    results = [
        EvaluationResult(
            metric_name="toxicity",
            score=0.0,
            evaluator_name="DeepevalEvaluator",
            evaluation_cost=0.005,
        ),
    ]
    emitter.on_evaluation_results(results, invocation)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _COST_METRIC)
    assert metric is not None
    data_points = list(metric.data.data_points)
    assert len(data_points) == 1

    attrs = dict(data_points[0].attributes)
    assert attrs["gen_ai.agent.name"] == "coordinator"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.provider.name"] == "azure"
    assert attrs["gen_ai.evaluation.name"] == "toxicity"


def test_emitter_records_zero_cost():
    """Zero cost is a valid value and should be recorded."""
    meter_provider, reader = _make_meter_and_reader()
    meter = meter_provider.get_meter("test")
    emitter = _make_emitter(meter)

    results = [
        EvaluationResult(
            metric_name="bias",
            score=0.1,
            evaluator_name="MyEval",
            evaluation_cost=0,
        ),
    ]
    emitter.on_evaluation_results(results, None)

    meter_provider.force_flush()
    metric = _get_metric_data(reader, _COST_METRIC)
    assert metric is not None
    data_points = list(metric.data.data_points)
    assert len(data_points) == 1
    assert data_points[0].sum == 0
    assert data_points[0].count == 1


# ============================================================================
# Integration: Manager stamps duration_s/evaluator_name on EvaluationResult
# ============================================================================


class _SlowEvaluator(Evaluator):
    def __init__(self, delay: float = 0.05):
        self._delay = delay

    def evaluate(self, invocation: GenAI) -> list[EvaluationResult]:
        time.sleep(self._delay)
        return [
            EvaluationResult(metric_name="bias", score=0.1),
            EvaluationResult(metric_name="toxicity", score=0.2),
        ]


def test_manager_stamps_duration_and_evaluator_name(monkeypatch):
    """Manager populates duration_s and evaluator_name on each EvaluationResult."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")

    handler = _StubHandler()
    manager = Manager(handler)

    evaluator = _SlowEvaluator(delay=0.05)
    manager._evaluators = {"LLMInvocation": [evaluator]}

    invocation = LLMInvocation(request_model="gpt-4o")
    manager._process_invocation(invocation)

    assert len(handler.calls) == 1
    _, results = handler.calls[0]
    assert len(results) == 2
    for res in results:
        assert res.duration_s is not None
        assert res.duration_s >= 0.04
        assert res.evaluator_name == "_SlowEvaluator"
