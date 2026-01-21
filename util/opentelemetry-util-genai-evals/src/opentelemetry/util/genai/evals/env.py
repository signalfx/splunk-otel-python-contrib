"""Environment helpers for evaluation configuration."""

from __future__ import annotations

import os
from typing import Mapping

from opentelemetry.util.genai.environment_variables import (
    DEEPEVAL_MAX_CONCURRENT,
    OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT,
    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
    OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL,
    OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE,
    OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION,
    OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE,
)

_TRUTHY = {"1", "true", "yes", "on"}


def _get_env(name: str, source: Mapping[str, str] | None = None) -> str | None:
    env = source if source is not None else os.environ
    return env.get(name)


def read_raw_evaluators(
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return raw evaluator configuration text."""

    return _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS, env)


def read_interval(
    env: Mapping[str, str] | None = None,
    *,
    default: float | None = 5.0,
) -> float | None:
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def read_aggregation_flag(
    env: Mapping[str, str] | None = None,
) -> bool | None:
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION, env)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def read_concurrent_flag(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Read the concurrent evaluation mode flag from environment.

    Args:
        env: Optional environment mapping (defaults to os.environ)

    Returns:
        True if concurrent mode is enabled, False otherwise.
    """
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT, env)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def read_worker_count(
    env: Mapping[str, str] | None = None,
    *,
    default: int = 4,
) -> int:
    """Read the number of evaluation worker threads from environment.

    Args:
        env: Optional environment mapping (defaults to os.environ)
        default: Default number of workers (4)

    Returns:
        Number of workers as integer. Minimum 1, maximum 16.
    """
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        count = int(raw.strip())
        return max(1, min(16, count))  # Clamp between 1 and 16
    except ValueError:
        return default


def read_queue_size(
    env: Mapping[str, str] | None = None,
    *,
    default: int = 100,
) -> int:
    """Read the evaluation queue size from environment.

    Checks OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE first, then falls back
    to legacy OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE for backward
    compatibility.

    Args:
        env: Optional environment mapping (defaults to os.environ)
        default: Default value when not set (100)

    Returns:
        Queue size as integer. Must be positive.
    """
    # Check new env var first
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE, env)
    if raw is None or raw.strip() == "":
        # Fall back to legacy env var
        raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        size = int(raw.strip())
        return size if size > 0 else default
    except ValueError:
        return default


def read_max_concurrent(
    env: Mapping[str, str] | None = None,
    *,
    default: int = 10,
) -> int:
    """Read the max concurrent metrics per test case from DEEPEVAL_MAX_CONCURRENT.

    This controls DeepEval's internal parallelism for metric evaluation.

    Args:
        env: Optional environment mapping (defaults to os.environ)
        default: Default value (10)

    Returns:
        Max concurrent value. Minimum 1, maximum 50.
    """
    raw = _get_env(DEEPEVAL_MAX_CONCURRENT, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
        return max(1, min(50, value))  # Clamp between 1 and 50
    except ValueError:
        return default


__all__ = [
    "read_raw_evaluators",
    "read_interval",
    "read_aggregation_flag",
    "read_queue_size",
    "read_concurrent_flag",
    "read_worker_count",
    "read_max_concurrent",
]
