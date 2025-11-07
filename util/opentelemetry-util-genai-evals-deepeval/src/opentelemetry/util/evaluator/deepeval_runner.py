# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Sequence

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

# CacheConfig is only available in deepeval >= 3.7.0
try:
    from deepeval.evaluate.configs import CacheConfig
    HAS_CACHE_CONFIG = True
except ImportError:
    HAS_CACHE_CONFIG = False


def run_evaluation(
    test_case: Any,
    metrics: Sequence[Any],
    debug_log: Callable[..., None] | None = None,
) -> Any:
    display_config = DisplayConfig(show_indicator=False, print_results=False)
    async_config = AsyncConfig(run_async=False)
    
    # Prepare kwargs for deepeval_evaluate
    eval_kwargs = {
        "async_config": async_config,
        "display_config": display_config,
    }
    
    # Only add cache_config if available (deepeval >= 3.7.0)
    if HAS_CACHE_CONFIG:
        cache_config = CacheConfig(write_cache=False, use_cache=False)
        eval_kwargs["cache_config"] = cache_config
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        result = deepeval_evaluate(
            [test_case],
            list(metrics),
            **eval_kwargs,
        )
    if debug_log is not None:
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
    return result


__all__ = ["run_evaluation"]
