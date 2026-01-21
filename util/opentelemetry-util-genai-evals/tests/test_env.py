"""Tests for environment variable reader functions in env.py."""

from __future__ import annotations

import pytest

from opentelemetry.util.genai.evals.env import (
    read_aggregation_flag,
    read_concurrent_flag,
    read_interval,
    read_queue_size,
    read_raw_evaluators,
    read_worker_count,
)

# =============================================================================
# Shared Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all OTEL_INSTRUMENTATION_GENAI_EVALS_* env vars."""
    env_vars = [
        "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# Tests for read_concurrent_flag
# =============================================================================


class TestReadConcurrentFlag:
    """Tests for read_concurrent_flag function."""

    def test_returns_false_when_not_set(self, clean_env):
        """Default is False when env var not set."""
        assert read_concurrent_flag() is False

    @pytest.mark.parametrize(
        "value", ["true", "True", "TRUE", "1", "yes", "on"]
    )
    def test_returns_true_for_truthy_values(
        self, clean_env, monkeypatch, value
    ):
        """Returns True for various truthy values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", value
        )
        assert read_concurrent_flag() is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
    def test_returns_false_for_falsy_values(
        self, clean_env, monkeypatch, value
    ):
        """Returns False for various falsy/empty values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", value
        )
        assert read_concurrent_flag() is False

    def test_whitespace_handling(self, clean_env, monkeypatch):
        """Handles whitespace in value."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "  true  "
        )
        assert read_concurrent_flag() is True

    def test_custom_env_mapping(self, clean_env):
        """Works with custom environment mapping."""
        env = {"OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT": "true"}
        assert read_concurrent_flag(env) is True


# =============================================================================
# Tests for read_worker_count
# =============================================================================


class TestReadWorkerCount:
    """Tests for read_worker_count function."""

    def test_default_value(self, clean_env):
        """Returns default (4) when env var not set."""
        assert read_worker_count() == 4

    def test_custom_default(self, clean_env):
        """Accepts custom default value."""
        assert read_worker_count(default=8) == 8

    def test_reads_from_env(self, clean_env, monkeypatch):
        """Reads value from environment variable."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "6")
        assert read_worker_count() == 6

    def test_clamps_minimum_to_1(self, clean_env, monkeypatch):
        """Clamps minimum value to 1."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "0")
        assert read_worker_count() == 1

        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "-5")
        assert read_worker_count() == 1

    def test_clamps_maximum_to_16(self, clean_env, monkeypatch):
        """Clamps maximum value to 16."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "100")
        assert read_worker_count() == 16

    def test_invalid_value_returns_default(self, clean_env, monkeypatch):
        """Returns default for invalid (non-integer) values."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "abc")
        assert read_worker_count() == 4

    def test_empty_string_returns_default(self, clean_env, monkeypatch):
        """Returns default for empty string."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "")
        assert read_worker_count() == 4

    def test_whitespace_handling(self, clean_env, monkeypatch):
        """Handles whitespace in value."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", "  8  ")
        assert read_worker_count() == 8


# =============================================================================
# Tests for read_queue_size
# =============================================================================


class TestReadQueueSize:
    """Tests for read_queue_size function."""

    def test_default_value(self, clean_env):
        """Returns default (100) when env var not set."""
        assert read_queue_size() == 100

    def test_custom_default(self, clean_env):
        """Accepts custom default value."""
        assert read_queue_size(default=50) == 50

    def test_reads_from_env(self, clean_env, monkeypatch):
        """Reads value from environment variable."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "500"
        )
        assert read_queue_size() == 500

    def test_fallback_to_legacy_env_var(self, clean_env, monkeypatch):
        """Falls back to legacy OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE", "200"
        )
        assert read_queue_size() == 200

    def test_new_env_var_takes_precedence(self, clean_env, monkeypatch):
        """New env var takes precedence over legacy."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "300"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE", "200"
        )
        assert read_queue_size() == 300

    def test_non_positive_returns_default(self, clean_env, monkeypatch):
        """Returns default for non-positive values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "-10"
        )
        assert read_queue_size() == 100

    def test_zero_returns_default(self, clean_env, monkeypatch):
        """Returns default for zero value."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "0")
        assert read_queue_size() == 100

    def test_invalid_value_returns_default(self, clean_env, monkeypatch):
        """Returns default for invalid (non-integer) values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "invalid"
        )
        assert read_queue_size() == 100

    def test_empty_string_returns_default(self, clean_env, monkeypatch):
        """Returns default for empty string."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "")
        assert read_queue_size() == 100


# =============================================================================
# Tests for read_interval
# =============================================================================


class TestReadInterval:
    """Tests for read_interval function."""

    def test_default_value(self, clean_env):
        """Returns default (5.0) when env var not set."""
        assert read_interval() == 5.0

    def test_custom_default(self, clean_env):
        """Accepts custom default value."""
        assert read_interval(default=10.0) == 10.0

    def test_reads_from_env(self, clean_env, monkeypatch):
        """Reads value from environment variable."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL", "2.5")
        assert read_interval() == 2.5

    def test_invalid_value_returns_default(self, clean_env, monkeypatch):
        """Returns default for invalid (non-float) values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL", "not-a-float"
        )
        assert read_interval() == 5.0


# =============================================================================
# Tests for read_aggregation_flag
# =============================================================================


class TestReadAggregationFlag:
    """Tests for read_aggregation_flag function."""

    def test_returns_none_when_not_set(self, clean_env):
        """Returns None when env var not set."""
        assert read_aggregation_flag() is None

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
    def test_returns_true_for_truthy_values(
        self, clean_env, monkeypatch, value
    ):
        """Returns True for various truthy values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION", value
        )
        assert read_aggregation_flag() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off"])
    def test_returns_false_for_falsy_values(
        self, clean_env, monkeypatch, value
    ):
        """Returns False for various falsy values."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION", value
        )
        assert read_aggregation_flag() is False


# =============================================================================
# Tests for read_raw_evaluators
# =============================================================================


class TestReadRawEvaluators:
    """Tests for read_raw_evaluators function."""

    def test_returns_none_when_not_set(self, clean_env):
        """Returns None when env var not set."""
        assert read_raw_evaluators() is None

    def test_reads_from_env(self, clean_env, monkeypatch):
        """Reads value from environment variable."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "deepeval"
        )
        assert read_raw_evaluators() == "deepeval"

    def test_complex_config(self, clean_env, monkeypatch):
        """Reads complex config string."""
        config = "deepeval(LLMInvocation(bias,toxicity))"
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", config
        )
        assert read_raw_evaluators() == config
