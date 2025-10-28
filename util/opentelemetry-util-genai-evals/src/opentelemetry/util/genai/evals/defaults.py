"""Builtin evaluator registrations exposed via entry points."""

from __future__ import annotations

from typing import Mapping, Sequence

from .builtins import LengthEvaluator
from .registry import EvaluatorRegistration


def _factory_wrapper(
    metrics: Sequence[str] | None = None,
    *,
    invocation_type: str | None = None,
    options: Mapping[str, str] | None = None,
) -> LengthEvaluator:
    return LengthEvaluator(
        metrics=metrics,
        invocation_type=invocation_type,
        options=options,
    )


def _default_metrics() -> Mapping[str, Sequence[str]]:
    instance = LengthEvaluator()
    return {"LLMInvocation": tuple(instance.default_metrics())}


def length_registration() -> EvaluatorRegistration:
    """Return the entry-point registration for builtin length evaluator."""

    return EvaluatorRegistration(
        factory=_factory_wrapper,
        default_metrics_factory=_default_metrics,
    )


__all__ = ["length_registration"]
