# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from importlib import import_module
from typing import Any, Mapping, Sequence

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from opentelemetry.util.genai.types import Error, EvaluationResult

# Canonical registry of metric names to Deepeval metric class names or sentinels
METRIC_REGISTRY: Mapping[str, str] = {
    "bias": "BiasMetric",
    "toxicity": "ToxicityMetric",
    "answer_relevancy": "AnswerRelevancyMetric",
    "answer_relevance": "AnswerRelevancyMetric",
    "relevance": "AnswerRelevancyMetric",
    "faithfulness": "FaithfulnessMetric",
    # custom metrics implemented via GEval
    "hallucination": "__custom_hallucination__",
    "sentiment": "__custom_sentiment__",
}


def coerce_option(value: Any) -> Any:
    # Best-effort recursive coercion to primitives
    if isinstance(value, MappingABC):
        out: dict[Any, Any] = {}
        for k, v in value.items():  # type: ignore[assignment]
            out[k] = coerce_option(v)
        return out
    if isinstance(value, (int, float, bool)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _missing_required_params(metric_cls: Any, test_case: Any) -> list[str]:
    required = getattr(metric_cls, "_required_params", [])
    missing: list[str] = []
    for param in required:
        attr_name = getattr(param, "value", str(param))
        value = getattr(test_case, attr_name, None)
        if value is None:
            missing.append(attr_name)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(attr_name)
            continue
        if isinstance(value, SequenceABC) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            # treat empty or whitespace-only sequences as missing
            flattened = []
            for item in value:
                try:
                    text = str(item).strip()
                except Exception:
                    text = ""
                if text:
                    flattened.append(text)
            if not flattened:
                missing.append(attr_name)
    return missing


def build_hallucination_metric(
    options: Mapping[str, Any], default_model: str | None
) -> Any:
    criteria = (
        "Assess if the output hallucinates by introducing facts, details, or claims not directly supported "
        "or implied by the input. Score 1 for fully grounded outputs (no fabrications) and 0 for severe hallucination."
    )
    steps = [
        "Review the input to extract all key facts and scope.",
        "Scan the output for any unsubstantiated additions, contradictions, or extrapolations beyond the input.",
        "Rate factual alignment: 1 = no hallucination, 0 = high hallucination risk.",
    ]
    if hasattr(LLMTestCaseParams, "INPUT_OUTPUT"):
        params = getattr(LLMTestCaseParams, "INPUT_OUTPUT")
    else:
        params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    threshold = options.get("threshold") if options else None
    model_override = options.get("model") if options else None
    strict_mode = options.get("strict_mode") if options else None
    kwargs: dict[str, Any] = {
        "name": "hallucination",
        "criteria": criteria,
        "evaluation_params": params,
        "threshold": threshold if isinstance(threshold, (int, float)) else 0.7,
        "model": model_override or default_model or "gpt-4o-mini",
    }
    if strict_mode is not None:
        kwargs["strict_mode"] = bool(strict_mode)
    try:
        return GEval(evaluation_steps=steps, **kwargs)
    except TypeError:
        return GEval(steps=steps, **kwargs)


def build_sentiment_metric(
    options: Mapping[str, Any], default_model: str | None
) -> Any:
    criteria = "Determine the overall sentiment polarity of the output text: -1 very negative, 0 neutral, +1 very positive."
    steps = [
        "Read the text and note words/phrases indicating sentiment.",
        "Judge if overall tone is negative, neutral, or positive.",
        "Assign a numeric polarity in [-1,1] capturing intensity.",
    ]
    if hasattr(LLMTestCaseParams, "ACTUAL_OUTPUT"):
        params = [LLMTestCaseParams.ACTUAL_OUTPUT]
    else:
        params = [LLMTestCaseParams.INPUT_OUTPUT]
    model_override = options.get("model") if options else None
    threshold = options.get("threshold") if options else None
    kwargs: dict[str, Any] = {
        "name": "sentiment [geval]",
        "criteria": criteria,
        "evaluation_params": params,
        "threshold": threshold if isinstance(threshold, (int, float)) else 0.0,
        "model": model_override or default_model or "gpt-4o-mini",
    }
    try:
        return GEval(evaluation_steps=steps, **kwargs)
    except TypeError:
        return GEval(steps=steps, **kwargs)


def instantiate_metrics(
    specs: Sequence[Any], test_case: Any, default_model: str | None
) -> tuple[Sequence[Any], Sequence[EvaluationResult]]:
    metrics_module = import_module("deepeval.metrics")
    registry = METRIC_REGISTRY
    instances: list[Any] = []
    skipped: list[EvaluationResult] = []
    for spec in specs:
        options = getattr(spec, "options", {}) or {}
        name = getattr(spec, "name", None)
        if name is None:
            continue
        if "__error__" in options:
            raise ValueError(options["__error__"])
        metric_class_name = registry[name]
        if metric_class_name == "__custom_hallucination__":
            instances.append(
                build_hallucination_metric(options, default_model)
            )
            continue
        if metric_class_name == "__custom_sentiment__":
            instances.append(build_sentiment_metric(options, default_model))
            continue
        metric_cls = getattr(metrics_module, metric_class_name, None)
        if metric_cls is None:
            raise RuntimeError(
                f"Deepeval metric class '{metric_class_name}' not found"
            )
        missing = _missing_required_params(metric_cls, test_case)
        if missing:
            message = (
                "Missing required Deepeval test case fields "
                f"{', '.join(missing)} for metric '{name}'."
            )
            skipped.append(
                EvaluationResult(
                    metric_name=name,
                    label="skipped",
                    explanation=message,
                    error=Error(message=message, type=ValueError),
                    attributes={
                        "deepeval.error": message,
                        "deepeval.skipped": True,
                        "deepeval.missing_params": missing,
                    },
                )
            )
            continue
        kwargs = dict(options)
        if default_model and "model" not in kwargs:
            kwargs["model"] = default_model
        instances.append(metric_cls(**kwargs))
    return instances, skipped


__all__ = [
    "METRIC_REGISTRY",
    "coerce_option",
    "instantiate_metrics",
    "build_hallucination_metric",
    "build_sentiment_metric",
]
