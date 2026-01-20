"""Environment helpers for evaluation configuration."""

from __future__ import annotations

import os
from typing import Mapping

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
    OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL,
    OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION,
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


def read_evaluation_queue_size() -> int:
    evaluation_queue_size = _get_env(
        OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE
    )
    default_queue_size = 100
    try:
        queue_size = (
            int(evaluation_queue_size)
            if evaluation_queue_size and evaluation_queue_size.strip()
            else default_queue_size
        )
        evaluation_queue_size = (
            queue_size if queue_size > 0 else default_queue_size
        )
    except (ValueError, TypeError):
        evaluation_queue_size = default_queue_size

    return evaluation_queue_size


__all__ = [
    "read_raw_evaluators",
    "read_interval",
    "read_aggregation_flag",
    "read_evaluation_queue_size",
]
