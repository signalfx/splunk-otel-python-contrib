import os
import time
import unittest
from unittest.mock import patch

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
    OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING,
)
from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.evals.monitoring import (
    EVAL_CLIENT_ENQUEUE_ERRORS,
    EVAL_CLIENT_QUEUE_SIZE,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


class _Handler:
    def __init__(self, meter_provider: MeterProvider) -> None:
        self._meter_provider = meter_provider
        self.calls = 0

    def evaluation_results(self, invocation, results) -> None:
        self.calls += 1


def _collect_metrics(reader: InMemoryMetricReader):
    try:
        reader.collect()
    except Exception:
        pass
    try:
        return reader.get_metrics_data()
    except Exception:
        return None


def _iter_metric_points(metrics_data):
    if metrics_data is None:
        return []
    points = []
    for rm in getattr(metrics_data, "resource_metrics", []) or []:
        for scope_metrics in getattr(rm, "scope_metrics", []) or []:
            for metric in getattr(scope_metrics, "metrics", []) or []:
                points.append(metric)
    return points


def _build_invocation() -> LLMInvocation:
    inv = LLMInvocation(request_model="m")
    inv.input_messages.append(
        InputMessage(role="user", parts=[Text(content="hi")])
    )
    inv.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="ok")],
            finish_reason="stop",
        )
    )
    return inv


class TestEvaluatorMonitoringMetrics(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "length",
            OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING: "true",
        },
        clear=True,
    )
    def test_queue_size_returns_to_zero_after_processing(self) -> None:
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        handler = _Handler(provider)
        manager = Manager(handler)
        try:
            manager.offer(_build_invocation())
            manager.wait_for_all(timeout=2.0)
        finally:
            manager.shutdown()

        try:
            provider.force_flush()
        except Exception:
            pass
        time.sleep(0.01)
        metrics_data = _collect_metrics(reader)
        metrics = _iter_metric_points(metrics_data)

        queue_metrics = [
            m for m in metrics if m.name == EVAL_CLIENT_QUEUE_SIZE
        ]
        self.assertTrue(queue_metrics, "queue size metric missing")
        data = getattr(queue_metrics[0], "data", None)
        self.assertIsNotNone(data)
        points = getattr(data, "data_points", []) or []
        self.assertTrue(points, "queue size metric has no points")
        # UpDownCounter is reported as a non-monotonic Sum; the value should be 0 after +1/-1.
        self.assertEqual(getattr(points[0], "value", None), 0)

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "length",
            OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING: "true",
        },
        clear=True,
    )
    def test_enqueue_errors_counter_increments_on_failure(self) -> None:
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        handler = _Handler(provider)
        manager = Manager(handler)
        try:
            manager._queue.put_nowait = (  # type: ignore[method-assign]
                lambda _inv: (_ for _ in ()).throw(RuntimeError("boom"))
            )
            manager.offer(_build_invocation())
        finally:
            manager.shutdown()

        try:
            provider.force_flush()
        except Exception:
            pass
        time.sleep(0.01)
        metrics_data = _collect_metrics(reader)
        metrics = _iter_metric_points(metrics_data)

        err_metrics = [
            m for m in metrics if m.name == EVAL_CLIENT_ENQUEUE_ERRORS
        ]
        self.assertTrue(err_metrics, "enqueue error counter metric missing")
        data = getattr(err_metrics[0], "data", None)
        self.assertIsNotNone(data)
        points = getattr(data, "data_points", []) or []
        self.assertTrue(points, "enqueue error counter has no points")
        # Counter should have incremented once for the RuntimeError.
        self.assertEqual(getattr(points[0], "value", None), 1)
