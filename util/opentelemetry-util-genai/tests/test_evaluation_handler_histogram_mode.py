"""Tests for handler histogram creation based on environment variable."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from opentelemetry.sdk.metrics import InMemoryMetricReader, MeterProvider
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC,
)
from opentelemetry.util.genai.handler import get_telemetry_handler


class TestHandlerHistogramMode(unittest.TestCase):
    def setUp(self):
        # Reset handler singleton before each test
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")

    def tearDown(self):
        # Clean up handler singleton after each test
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")

    @patch.dict(os.environ, {}, clear=True)
    def test_handler_creates_multiple_histograms_when_env_var_unset(self):
        """Test that handler creates separate histograms when env var is not set."""
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        handler = get_telemetry_handler(meter_provider=meter_provider)

        # Get histogram factory
        histogram_factory = handler._get_eval_histogram

        # Request histograms for different metrics
        bias_hist = histogram_factory("bias")
        toxicity_hist = histogram_factory("toxicity")
        bias_hist_2 = histogram_factory("bias")  # Should return same as first

        # In multiple histogram mode, each metric gets its own histogram
        assert bias_hist is not None
        assert toxicity_hist is not None
        assert (
            bias_hist is bias_hist_2
        )  # Same metric should return same histogram
        assert (
            bias_hist is not toxicity_hist
        )  # Different metrics should have different histograms

        # Verify that handler has separate histograms stored
        assert hasattr(handler, "_evaluation_histograms")
        assert "gen_ai.evaluation.bias" in handler._evaluation_histograms
        assert "gen_ai.evaluation.toxicity" in handler._evaluation_histograms

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC: "true"},
        clear=True,
    )
    def test_handler_creates_single_histogram_when_env_var_true(self):
        """Test that handler creates single histogram when env var is set to true."""
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        handler = get_telemetry_handler(meter_provider=meter_provider)

        # Get histogram factory
        histogram_factory = handler._get_eval_histogram

        # Request histograms for different metrics
        bias_hist = histogram_factory("bias")
        toxicity_hist = histogram_factory("toxicity")
        sentiment_hist = histogram_factory("sentiment")

        # In single metric mode, all metrics should return the same histogram
        assert bias_hist is not None
        assert toxicity_hist is not None
        assert sentiment_hist is not None
        assert bias_hist is toxicity_hist  # All should be the same histogram
        assert bias_hist is sentiment_hist

        # Verify that handler has single histogram stored
        assert hasattr(handler, "_evaluation_histogram")
        assert handler._evaluation_histogram is not None
        assert handler._evaluation_histogram is bias_hist

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC: "false"},
        clear=True,
    )
    def test_handler_creates_multiple_histograms_when_env_var_false(self):
        """Test that handler creates separate histograms when env var is set to false."""
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        handler = get_telemetry_handler(meter_provider=meter_provider)

        # Get histogram factory
        histogram_factory = handler._get_eval_histogram

        # Request histograms for different metrics
        bias_hist = histogram_factory("bias")
        toxicity_hist = histogram_factory("toxicity")

        # In multiple histogram mode, each metric gets its own histogram
        assert bias_hist is not None
        assert toxicity_hist is not None
        assert bias_hist is not toxicity_hist

        # Verify that handler has separate histograms stored
        assert hasattr(handler, "_evaluation_histograms")
        assert "gen_ai.evaluation.bias" in handler._evaluation_histograms
        assert "gen_ai.evaluation.toxicity" in handler._evaluation_histograms

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC: "1"},
        clear=True,
    )
    def test_handler_creates_single_histogram_when_env_var_is_one(self):
        """Test that handler creates single histogram when env var is set to '1'."""
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        handler = get_telemetry_handler(meter_provider=meter_provider)

        # Get histogram factory
        histogram_factory = handler._get_eval_histogram

        # Request histograms for different metrics
        bias_hist = histogram_factory("bias")
        toxicity_hist = histogram_factory("toxicity")

        # In single metric mode, all metrics should return the same histogram
        assert bias_hist is not None
        assert toxicity_hist is not None
        assert bias_hist is toxicity_hist

        # Verify that handler has single histogram stored
        assert hasattr(handler, "_evaluation_histogram")
        assert handler._evaluation_histogram is not None
        assert handler._evaluation_histogram is bias_hist


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
