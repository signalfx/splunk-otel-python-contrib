"""Environment helpers for evaluation configuration."""

from __future__ import annotations

import logging
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
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_ENABLE,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS,
)

_TRUTHY = {"1", "true", "yes", "on"}
_LOGGER = logging.getLogger(__name__)


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
    except ValueError as e:
        _LOGGER.warning(
            "Failed to parse %s: %s (error: %s), using default %s",
            OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL,
            raw,
            e,
            default,
        )
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
        if count < 1 or count > 16:
            _LOGGER.warning(
                "Value for %s: %s is outside valid range [1, 16], clamping to range",
                OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS,
                raw,
            )
        return max(1, min(16, count))  # Clamp between 1 and 16
    except ValueError as e:
        _LOGGER.warning(
            "Failed to parse %s: %s (error: %s), using default %d",
            OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS,
            raw,
            e,
            default,
        )
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
        if size <= 0:
            _LOGGER.warning(
                "Invalid value for queue size: %s (must be positive integer), using default %d",
                raw,
                default,
            )
            return default
        return size
    except ValueError as e:
        _LOGGER.warning(
            "Failed to parse queue size: %s (error: %s), using default %d",
            raw,
            e,
            default,
        )
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
        if value < 1 or value > 50:
            _LOGGER.warning(
                "Value for %s: %s is outside valid range [1, 50], clamping to range",
                DEEPEVAL_MAX_CONCURRENT,
                raw,
            )
        return max(1, min(50, value))  # Clamp between 1 and 50
    except ValueError as e:
        _LOGGER.warning(
            "Failed to parse %s: %s (error: %s), using default %d",
            DEEPEVAL_MAX_CONCURRENT,
            raw,
            e,
            default,
        )
        return default


def read_evaluation_rate_limit_enable(
    env: Mapping[str, str] | None = None,
    *,
    default: bool = True,
) -> bool:
    """
    Enable evaluation rate limiting.
    """
    raw = _get_env(
        OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_ENABLE, env
    )
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUTHY


def read_evaluation_rate_limit_rps(
    env: Mapping[str, str] | None = None,
    *,
    default: int = 0,
) -> int:
    """
    Per-process proactive admission rate (invocations per second) for evaluation queue.
    Set to <= 0 to disable rate limiting.
    """
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        _LOGGER.warning(
            "Failed to parse %s: %s (error: %s), using default %d",
            OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS,
            raw,
            e,
            default,
        )
        return default


def read_evaluation_rate_limit_burst(
    env: Mapping[str, str] | None = None,
    *,
    default: int = 4,
) -> int:
    """
    Burst capacity for token bucket limiter.
    """
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
        if value <= 0:
            _LOGGER.warning(
                "Invalid value for %s: %s (must be positive integer), using default %d",
                OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST,
                raw,
                default,
            )
            return default
        return value
    except (ValueError, TypeError) as e:
        _LOGGER.warning(
            "Failed to parse %s: %s (error: %s), using default %d",
            OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST,
            raw,
            e,
            default,
        )
        return default


__all__ = [
    "read_raw_evaluators",
    "read_interval",
    "read_aggregation_flag",
    "read_queue_size",
    "read_concurrent_flag",
    "read_worker_count",
    "read_max_concurrent",
    "read_evaluation_rate_limit_enable",
    "read_evaluation_rate_limit_rps",
    "read_evaluation_rate_limit_burst",
]
