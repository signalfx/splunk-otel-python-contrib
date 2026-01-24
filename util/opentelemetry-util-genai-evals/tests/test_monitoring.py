# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for evaluation monitoring metrics."""

from __future__ import annotations

import time
from unittest.mock import MagicMock


class TestEvaluationMonitor:
    """Tests for the EvaluationMonitor class."""

    def test_monitor_initialization(self, monkeypatch):
        """Test that monitor initializes with correct metric instruments."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        # Reset the global monitor
        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_observable_gauge.return_value = MagicMock()

        _monitor = EvaluationMonitor(meter=mock_meter)

        # Verify instruments were created
        assert mock_meter.create_histogram.call_count == 2
        assert mock_meter.create_counter.call_count == 1
        assert mock_meter.create_observable_gauge.call_count == 1

        # Verify histogram names
        histogram_calls = mock_meter.create_histogram.call_args_list
        histogram_names = [call.kwargs.get("name") for call in histogram_calls]
        assert "gen_ai.evaluation.client.operation.duration" in histogram_names
        assert "gen_ai.evaluation.client.token.usage" in histogram_names

        # Verify counter name
        counter_call = mock_meter.create_counter.call_args
        assert (
            counter_call.kwargs.get("name")
            == "gen_ai.evaluation.client.enqueue.errors"
        )

    def test_on_enqueue_increments_queue_size(self, monkeypatch):
        """Test that on_enqueue increments the queue size."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_meter.create_histogram.return_value = MagicMock()
        mock_meter.create_counter.return_value = MagicMock()
        mock_meter.create_observable_gauge.return_value = MagicMock()

        monitor = EvaluationMonitor(meter=mock_meter)

        assert monitor._queue_size == 0
        monitor.on_enqueue()
        assert monitor._queue_size == 1
        monitor.on_enqueue()
        assert monitor._queue_size == 2

    def test_on_dequeue_decrements_queue_size(self, monkeypatch):
        """Test that on_dequeue decrements the queue size."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_meter.create_histogram.return_value = MagicMock()
        mock_meter.create_counter.return_value = MagicMock()
        mock_meter.create_observable_gauge.return_value = MagicMock()

        monitor = EvaluationMonitor(meter=mock_meter)

        monitor.on_enqueue()
        monitor.on_enqueue()
        assert monitor._queue_size == 2

        monitor.on_dequeue()
        assert monitor._queue_size == 1

        monitor.on_dequeue()
        assert monitor._queue_size == 0

        # Should not go below 0
        monitor.on_dequeue()
        assert monitor._queue_size == 0

    def test_on_enqueue_error_records_counter(self, monkeypatch):
        """Test that on_enqueue_error records to the counter."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_observable_gauge.return_value = MagicMock()

        monitor = EvaluationMonitor(meter=mock_meter)

        monitor.on_enqueue_error("queue_full")

        mock_counter.add.assert_called_once_with(
            1, {"error.type": "queue_full"}
        )

    def test_on_enqueue_error_with_custom_attributes(self, monkeypatch):
        """Test that on_enqueue_error accepts additional attributes."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_histogram.return_value = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_observable_gauge.return_value = MagicMock()

        monitor = EvaluationMonitor(meter=mock_meter)

        monitor.on_enqueue_error("queue_error", {"custom.attribute": "value"})

        mock_counter.add.assert_called_once()
        call_args = mock_counter.add.call_args
        assert call_args[0][0] == 1
        assert call_args[0][1]["error.type"] == "queue_error"
        assert call_args[0][1]["custom.attribute"] == "value"


class TestEvaluationContext:
    """Tests for the EvaluationContext dataclass."""

    def test_context_creation(self):
        """Test that EvaluationContext is created with correct defaults."""
        from opentelemetry.util.genai.evals.monitoring import EvaluationContext

        ctx = EvaluationContext(
            evaluation_name="bias",
            evaluator_name="deepeval",
        )

        assert ctx.evaluation_name == "bias"
        assert ctx.evaluator_name == "deepeval"
        assert ctx.request_model is None
        assert ctx.provider is None
        assert ctx.input_tokens is None
        assert ctx.output_tokens is None
        assert ctx.error_type is None
        assert isinstance(ctx.attributes, dict)
        assert ctx.start_time > 0

    def test_context_elapsed_seconds(self):
        """Test that elapsed_seconds returns correct duration."""
        from opentelemetry.util.genai.evals.monitoring import EvaluationContext

        ctx = EvaluationContext(evaluation_name="bias")

        time.sleep(0.1)  # Sleep for 100ms

        elapsed = ctx.elapsed_seconds()
        assert elapsed >= 0.1
        assert elapsed < 0.5  # Should be less than 500ms


class TestRecordEvaluation:
    """Tests for recording evaluation metrics."""

    def test_record_evaluation_with_duration(self, monkeypatch):
        """Test that record_evaluation records duration histogram."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationContext,
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_duration_histogram = MagicMock()
        mock_token_histogram = MagicMock()

        def create_histogram_side_effect(**kwargs):
            if "duration" in kwargs.get("name", ""):
                return mock_duration_histogram
            return mock_token_histogram

        mock_meter.create_histogram.side_effect = create_histogram_side_effect
        mock_meter.create_counter.return_value = MagicMock()
        mock_meter.create_observable_gauge.return_value = MagicMock()

        monitor = EvaluationMonitor(meter=mock_meter)

        ctx = EvaluationContext(
            evaluation_name="bias",
            evaluator_name="deepeval",
            request_model="gpt-4o-mini",
            provider="openai",
        )

        time.sleep(0.05)  # Small delay
        monitor.record_evaluation(ctx)

        # Verify duration was recorded
        mock_duration_histogram.record.assert_called_once()
        call_args = mock_duration_histogram.record.call_args
        duration = call_args[0][0]
        attrs = call_args[0][1]

        assert duration >= 0.05
        assert attrs["gen_ai.operation.name"] == "evaluate"
        assert attrs["gen_ai.evaluation.name"] == "bias"
        assert attrs["gen_ai.evaluation.evaluator"] == "deepeval"
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert attrs["gen_ai.system"] == "openai"

    def test_record_evaluation_with_tokens(self, monkeypatch):
        """Test that record_evaluation records token usage."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )

        from opentelemetry.util.genai.evals.monitoring import (
            EvaluationContext,
            EvaluationMonitor,
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        mock_meter = MagicMock()
        mock_duration_histogram = MagicMock()
        mock_token_histogram = MagicMock()

        def create_histogram_side_effect(**kwargs):
            if "duration" in kwargs.get("name", ""):
                return mock_duration_histogram
            return mock_token_histogram

        mock_meter.create_histogram.side_effect = create_histogram_side_effect
        mock_meter.create_counter.return_value = MagicMock()
        mock_meter.create_observable_gauge.return_value = MagicMock()

        monitor = EvaluationMonitor(meter=mock_meter)

        ctx = EvaluationContext(
            evaluation_name="toxicity",
            evaluator_name="deepeval",
            input_tokens=500,
            output_tokens=100,
        )

        monitor.record_evaluation(ctx)

        # Verify tokens were recorded (2 calls: input and output)
        assert mock_token_histogram.record.call_count == 2

        calls = mock_token_histogram.record.call_args_list
        input_call = next(
            c for c in calls if c[0][1].get("gen_ai.token.type") == "input"
        )
        output_call = next(
            c for c in calls if c[0][1].get("gen_ai.token.type") == "output"
        )

        assert input_call[0][0] == 500
        assert output_call[0][0] == 100


class TestManagerIntegration:
    """Tests for Manager integration with monitoring."""

    def test_manager_creates_monitor_when_enabled(self, monkeypatch):
        """Test that Manager creates a monitor when monitoring is enabled."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        from opentelemetry.util.genai.evals.manager import Manager
        from opentelemetry.util.genai.evals.monitoring import (
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        handler = MagicMock()
        manager = Manager(handler)

        assert manager.monitoring_enabled is True
        assert manager.monitor is not None
        manager.shutdown()

    def test_manager_no_monitor_when_disabled(self, monkeypatch):
        """Test that Manager does not create a monitor when disabled."""
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", raising=False
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        from opentelemetry.util.genai.evals.manager import Manager
        from opentelemetry.util.genai.evals.monitoring import (
            reset_evaluation_monitor,
        )

        reset_evaluation_monitor()

        handler = MagicMock()
        manager = Manager(handler)

        assert manager.monitoring_enabled is False
        assert manager.monitor is None
        manager.shutdown()

    def test_manager_records_enqueue_metrics(self, monkeypatch):
        """Test that Manager records enqueue metrics."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        from opentelemetry.util.genai.evals.manager import Manager
        from opentelemetry.util.genai.evals.monitoring import (
            reset_evaluation_monitor,
        )
        from opentelemetry.util.genai.types import LLMInvocation

        reset_evaluation_monitor()

        handler = MagicMock()
        manager = Manager(handler)
        # Force has_evaluators to be True to test queue behavior
        manager._evaluators = {"LLMInvocation": [MagicMock()]}

        # Record initial queue size
        initial_size = manager.monitor._queue_size

        invocation = LLMInvocation(request_model="test-model")
        manager.offer(invocation)

        # Queue size should have increased
        assert manager.monitor._queue_size == initial_size + 1

        manager.shutdown()

    def test_manager_records_enqueue_error_on_full_queue(self, monkeypatch):
        """Test that Manager records error when queue is full."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "1")

        from opentelemetry.util.genai.evals.manager import Manager
        from opentelemetry.util.genai.evals.monitoring import (
            reset_evaluation_monitor,
        )
        from opentelemetry.util.genai.types import LLMInvocation

        reset_evaluation_monitor()

        handler = MagicMock()
        manager = Manager(handler)
        manager._evaluators = {"LLMInvocation": [MagicMock()]}

        # Mock the error counter
        mock_counter = MagicMock()
        manager.monitor._enqueue_error_counter = mock_counter

        # Fill the queue
        invocation1 = LLMInvocation(request_model="test-model-1")
        manager.offer(invocation1)

        # This should fail and record error
        invocation2 = LLMInvocation(request_model="test-model-2")
        manager.offer(invocation2)

        mock_counter.add.assert_called_once_with(
            1, {"error.type": "queue_full"}
        )
        assert invocation2.evaluation_error == "client_evaluation_queue_full"

        manager.shutdown()


class TestReadMonitoringFlag:
    """Tests for the read_monitoring_flag env helper."""

    def test_monitoring_flag_enabled(self, monkeypatch):
        """Test that truthy values enable monitoring."""
        from opentelemetry.util.genai.evals.env import read_monitoring_flag

        for value in ["true", "TRUE", "1", "yes", "on"]:
            monkeypatch.setenv(
                "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", value
            )
            assert read_monitoring_flag() is True

    def test_monitoring_flag_disabled(self, monkeypatch):
        """Test that falsy values disable monitoring."""
        from opentelemetry.util.genai.evals.env import read_monitoring_flag

        for value in ["false", "FALSE", "0", "no", "off"]:
            monkeypatch.setenv(
                "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", value
            )
            assert read_monitoring_flag() is False

    def test_monitoring_flag_not_set(self, monkeypatch):
        """Test that unset env var defaults to False."""
        from opentelemetry.util.genai.evals.env import read_monitoring_flag

        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", raising=False
        )
        assert read_monitoring_flag() is False
