# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from inspect import signature
from typing import Any, Callable, Dict, Sequence

from deepeval import evaluate as deepeval_evaluate  # type: ignore[import]

_configs_module = import_module("deepeval.evaluate.configs")
AsyncConfig = getattr(_configs_module, "AsyncConfig")
DisplayConfig = getattr(_configs_module, "DisplayConfig")
CacheConfig = getattr(_configs_module, "CacheConfig", None)

_evaluate_params = set(signature(deepeval_evaluate).parameters)
_supports_async_config = "async_config" in _evaluate_params
_supports_display_config = "display_config" in _evaluate_params
_supports_cache_config = (
    "cache_config" in _evaluate_params and CacheConfig is not None
)


def run_evaluation(
    test_case: Any,
    metrics: Sequence[Any],
    debug_log: Callable[..., None] | None = None,
) -> Any:
    call_kwargs: Dict[str, Any] = {}
    if _supports_display_config:
        display_config = DisplayConfig(
            show_indicator=False, print_results=False
        )
        call_kwargs["display_config"] = display_config
    if _supports_async_config:
        async_config = AsyncConfig(run_async=False)
        call_kwargs["async_config"] = async_config
    if _supports_cache_config and CacheConfig is not None:
        cache_config = CacheConfig(write_cache=False, use_cache=False)
        call_kwargs["cache_config"] = cache_config
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        result = deepeval_evaluate(
            [test_case],
            list(metrics),
            **call_kwargs,
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
