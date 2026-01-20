# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Sequence

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

# Environment variable to control async mode
_ASYNC_MODE_ENV = "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT"


def _is_async_mode_enabled() -> bool:
    """Check if concurrent/async mode is enabled via environment variable."""
    raw = os.environ.get(_ASYNC_MODE_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def run_evaluation(
    test_case: Any,
    metrics: Sequence[Any],
    debug_log: Callable[..., None] | None = None,
    *,
    use_async: bool | None = None,
) -> Any:
    """Run DeepEval evaluation synchronously or asynchronously.

    Args:
        test_case: The test case to evaluate.
        metrics: List of metrics to run.
        debug_log: Optional debug logging function.
        use_async: Override async mode. If None, uses env var setting.

    Returns:
        Evaluation result from DeepEval.
    """
    # Determine whether to use async based on parameter or env var
    async_enabled = use_async if use_async is not None else _is_async_mode_enabled()

    display_config = DisplayConfig(show_indicator=False, print_results=False)
    async_config = AsyncConfig(run_async=async_enabled)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        result = deepeval_evaluate(
            [test_case],
            list(metrics),
            async_config=async_config,
            display_config=display_config,
        )

    _handle_debug_output(debug_log, stdout_buffer, stderr_buffer)
    return result


async def run_evaluation_async(
    test_case: Any,
    metrics: Sequence[Any],
    debug_log: Callable[..., None] | None = None,
) -> Any:
    """Run DeepEval evaluation asynchronously in a thread pool.

    This function runs DeepEval in a thread pool to avoid blocking the event
    loop. When concurrent mode is enabled via OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT,
    DeepEval's internal async mode is also enabled for concurrent metric evaluation.

    Note: When DeepEval's run_async=True, it spawns internal threads for concurrent
    metric evaluation. The caller should add a buffer wait time after queue completion
    to allow these threads to finish.

    Args:
        test_case: The test case to evaluate.
        metrics: List of metrics to run.
        debug_log: Optional debug logging function.

    Returns:
        Evaluation result from DeepEval.
    """
    # Check if concurrent mode is enabled - if so, use DeepEval's async mode too
    use_deepeval_async = _is_async_mode_enabled()

    def _run_sync() -> Any:
        display_config = DisplayConfig(show_indicator=False, print_results=False)
        # When concurrent mode enabled, use DeepEval's internal async for metric parallelism
        # When disabled, run metrics sequentially within each invocation
        async_config = AsyncConfig(run_async=use_deepeval_async, max_concurrent=10)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = deepeval_evaluate(
                [test_case],
                list(metrics),
                async_config=async_config,
                display_config=display_config,
            )

        _handle_debug_output(debug_log, stdout_buffer, stderr_buffer)
        return result

    # Run in thread pool to not block event loop with I/O redirection
    return await asyncio.to_thread(_run_sync)


def _handle_debug_output(
    debug_log: Callable[..., None] | None,
    stdout_buffer: io.StringIO,
    stderr_buffer: io.StringIO,
) -> None:
    """Handle debug logging for captured stdout/stderr."""
    if debug_log is None:
        return

    out = stdout_buffer.getvalue().strip()
    err = stderr_buffer.getvalue().strip()
    if out:
        try:
            debug_log("evaluator.deepeval.stdout", None, output=out)
        except Exception:
            pass
    if err:
        try:
            debug_log("evaluator.deepeval.stderr", None, output=err)
        except Exception:
            pass


__all__ = ["run_evaluation", "run_evaluation_async"]
