# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""DeepEval evaluation runner.

This module provides sync and async evaluation functions that wrap DeepEval.
The core evaluation logic is centralized in _execute_evaluation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

from opentelemetry.util.genai.evals.env import (
    read_concurrent_flag,
    read_max_concurrent,
)


def _execute_evaluation(
    test_case: Any,
    metrics: Sequence[Any],
    run_async: bool,
) -> Any:
    """Core evaluation logic - runs DeepEval evaluation.

    This is the single source of truth for running DeepEval evaluations.
    Both sync and async public APIs delegate to this function.

    Args:
        test_case: The test case to evaluate.
        metrics: List of metrics to run.
        run_async: Whether to enable DeepEval's internal parallel metric evaluation.

    Returns:
        Evaluation result from DeepEval.
    """
    display_config = DisplayConfig(show_indicator=False, print_results=False)
    max_concurrent = read_max_concurrent() if run_async else 1
    async_config = AsyncConfig(
        run_async=run_async, max_concurrent=max_concurrent
    )

    return deepeval_evaluate(
        [test_case],
        list(metrics),
        async_config=async_config,
        display_config=display_config,
    )


def run_evaluation(
    test_case: Any,
    metrics: Sequence[Any],
) -> Any:
    """Run DeepEval evaluation synchronously (sequential mode).

    This function is called from the sequential evaluation path where
    evaluations are processed one at a time. DeepEval's internal async
    mode is disabled to maintain sequential behavior.

    Args:
        test_case: The test case to evaluate.
        metrics: List of metrics to run.

    Returns:
        Evaluation result from DeepEval.
    """
    return _execute_evaluation(test_case, metrics, run_async=False)


async def run_evaluation_async(
    test_case: Any,
    metrics: Sequence[Any],
) -> Any:
    """Run DeepEval evaluation asynchronously with parallel metrics.

    This function is called from the concurrent evaluation path. It runs
    DeepEval in a thread pool to avoid blocking the event loop.

    When OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT is enabled, DeepEval's
    internal async mode is also enabled for parallel metric evaluation within
    each test case. The number of parallel metrics is controlled by
    DEEPEVAL_MAX_CONCURRENT (default: 10).

    Note: When DeepEval's run_async=True, it spawns internal threads for
    concurrent metric evaluation. The caller should add a buffer wait time
    after queue completion to allow these threads to finish.

    Args:
        test_case: The test case to evaluate.
        metrics: List of metrics to run.

    Returns:
        Evaluation result from DeepEval.
    """
    use_parallel = read_concurrent_flag()
    return await asyncio.to_thread(
        _execute_evaluation, test_case, metrics, use_parallel
    )


__all__ = ["run_evaluation", "run_evaluation_async"]
