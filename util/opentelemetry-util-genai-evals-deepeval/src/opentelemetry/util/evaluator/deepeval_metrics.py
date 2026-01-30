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
        "Detect hallucinations: factual claims in the output that contradict the input or introduce "
        "specific details (names, dates, numbers, quotes, statistics) that cannot be reasonably inferred "
        "from the input. Distinguish between valid logical inference (acceptable) and fabrication (hallucination). "
        "Score 1.0 for outputs fully grounded in the input (no hallucinations), 0.0 for outputs with clear fabrications "
        "or contradictions (maximum hallucination). Higher scores indicate less hallucination (better quality). "
        "Be conservative: only flag as hallucination if there is clear evidence of fabrication or contradiction, "
        "not reasonable inference or interpretation."
    )
    steps = [
        "Extract all explicit facts, claims, and details from the input. Note what information is provided and the scope.",
        "Identify each factual claim in the output. For each claim, categorize it as: (a) explicitly stated in input, "
        "(b) reasonable logical inference from input (e.g., 'it's sunny' → 'good weather'), or (c) introduces new "
        "specifics not derivable from input (e.g., specific names, dates, numbers, quotes not mentioned).",
        "Distinguish inference from fabrication: Inference = claim logically follows from input using common knowledge. "
        "Fabrication = introduces new factual specifics (names, dates, statistics, quotes) that cannot be derived. "
        "For domain-specific claims, consider if they're reasonable extensions of the input rather than arbitrary additions.",
        "Check for direct contradictions: does the output state something that contradicts information in the input?",
        "Self-verify: For each flagged item, ask 'Could a reasonable person infer this from the input?' If yes, it's inference, "
        "not fabrication. Only mark as hallucination if confident there is clear fabrication (new specifics) or contradiction. "
        "When uncertain between inference and fabrication, always favor inference (assign higher score) to minimize false positives.",
        "Assign score: 1.0 = no hallucinations (all claims grounded or reasonably inferred), 0.8-0.9 = minor unwarranted "
        "specifics or edge cases, 0.5-0.7 = some fabrications present, 0.0-0.4 = significant fabrications or contradictions. "
        "Higher scores indicate less hallucination (better quality).",
    ]
    if hasattr(LLMTestCaseParams, "INPUT_OUTPUT"):
        params = getattr(LLMTestCaseParams, "INPUT_OUTPUT")
    else:
        params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    threshold = options.get("threshold") if options else None
    model_override = options.get("model") if options else None
    strict_mode = options.get("strict_mode") if options else None
    kwargs: dict[str, Any] = {
        "name": "hallucination [geval]",
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
    criteria = (
        "Rate the overall sentiment of the output text on a scale from 0 to 1. "
        "0 = strongly negative, 0.5 = neutral, 1 = strongly positive. "
        "Use intermediate values to capture intensity and mixed sentiments."
    )
    steps = [
        "Identify emotion-carrying words, phrases, and overall tone in the text.",
        "Consider context: sarcasm, irony, or mixed sentiment should be judged by net effect on overall tone.",
        "Assess intensity: is the sentiment strongly expressed (near 0 or 1) or mild (near 0.5)?",
        "Assign score: 0.0-0.35 = negative sentiment, 0.35-0.65 = neutral sentiment, 0.65-1.0 = positive sentiment. "
        "Use decimal values within these ranges to capture intensity (e.g., 0.2 = moderately negative, 0.8 = strongly positive).",
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
