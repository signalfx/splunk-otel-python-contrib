# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the Deepeval evaluator plugin."""

from __future__ import annotations

import logging
import os
import re as _re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import openai

from opentelemetry.util.genai.evals.base import Evaluator
from opentelemetry.util.genai.evals.registry import (
    EvaluatorRegistration,
    register_evaluator,
)
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Error,
    EvaluationResult,
    GenAI,
    LLMInvocation,
)

from .deepeval_adapter import build_llm_test_case as _build_llm_test_case
from .deepeval_metrics import (
    METRIC_REGISTRY as _DEEPEVAL_METRIC_REGISTRY,
)
from .deepeval_metrics import (
    coerce_option as _coerce_option,
)
from .deepeval_metrics import (
    instantiate_metrics as _instantiate_metrics,
)
from .deepeval_model import create_eval_model as _create_eval_model
from .deepeval_runner import run_evaluation as _run_deepeval

try:  # Optional debug logging import
    from opentelemetry.util.genai.debug import genai_debug_log
except Exception:  # pragma: no cover

    def genai_debug_log(*_a: Any, **_k: Any) -> None:  # type: ignore
        return None


_DEFAULT_METRICS: Mapping[str, Sequence[str]] = {
    "LLMInvocation": (
        "bias",
        "toxicity",
        "answer_relevancy",
        "hallucination",
        "sentiment",
    ),
    "AgentInvocation": (
        "bias",
        "toxicity",
        "answer_relevancy",
        "hallucination",
        "sentiment",
    ),
}


_LOGGER = logging.getLogger(__name__)


# Disable Deepeval's internal telemetry (Posthog/New Relic) by default so that
# it does not emit extra spans or events when running inside the GenAI
# instrumentation stack. Users can re-enable it by explicitly setting
# ``DEEPEVAL_TELEMETRY_OPT_OUT`` to ``0`` before importing this module.
# "YES" works with deepeval>=3.3.9,<3.8.0
if os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") is None:
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    options: Mapping[str, Any]


@dataclass
class _MetricContext:
    name: str
    key: str
    raw_score: Any
    score: float | None
    raw_threshold: Any
    threshold: float | None
    success: Any
    reason: str | None
    error: Error | None
    attributes: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _build_metric_context(metric: Any, test: Any) -> _MetricContext:
    name = getattr(metric, "name", "deepeval")
    key = str(name).lower()
    raw_score = getattr(metric, "score", None)
    score = _safe_float(raw_score)
    raw_threshold = getattr(metric, "threshold", None)
    threshold = _safe_float(raw_threshold)
    success = getattr(metric, "success", None)
    reason = getattr(metric, "reason", None)
    evaluation_model = getattr(metric, "evaluation_model", None)
    evaluation_cost = getattr(metric, "evaluation_cost", None)
    error_msg = getattr(metric, "error", None)

    attributes: dict[str, Any] = {"deepeval.success": success}
    if raw_threshold is not None:
        attributes["deepeval.threshold"] = raw_threshold
    if evaluation_model:
        attributes["deepeval.evaluation_model"] = evaluation_model
    if evaluation_cost is not None:
        attributes["deepeval.evaluation_cost"] = evaluation_cost
    if getattr(test, "name", None):
        attributes.setdefault("deepeval.test_case", getattr(test, "name"))
    if getattr(test, "success", None) is not None:
        attributes.setdefault(
            "deepeval.test_success", getattr(test, "success")
        )

    error = (
        Error(message=str(error_msg), type=RuntimeError) if error_msg else None
    )
    return _MetricContext(
        name=name,
        key=key,
        raw_score=raw_score,
        score=score,
        raw_threshold=raw_threshold,
        threshold=threshold,
        success=success,
        reason=reason,
        error=error,
        attributes=attributes,
    )


def _determine_label(ctx: _MetricContext) -> str | None:
    key = ctx.key
    success = ctx.success
    score = ctx.score
    threshold = ctx.threshold

    if key in {"relevance", "answer_relevancy", "answer_relevance"}:
        if success is True:
            return "Relevant"
        if success is False:
            return "Irrelevant"
        if isinstance(score, (int, float)):
            effective_threshold = (
                float(threshold)
                if isinstance(threshold, (int, float))
                else 0.5
            )
            return "Relevant" if score >= effective_threshold else "Irrelevant"
        return "Irrelevant"

    if key.startswith("hallucination") or key == "faithfulness":
        if success is True:
            return "Not Hallucinated"
        if success is False:
            return "Hallucinated"
        return None

    if key == "toxicity":
        if success is True:
            return "Not Toxic"
        if success is False:
            return "Toxic"
        return None

    if key == "bias":
        if success is True:
            return "Not Biased"
        if success is False:
            return "Biased"
        return None

    if key.startswith("sentiment"):
        return None  # handled in sentiment post-processing

    if success is True:
        return "Pass"
    if success is False:
        return "Fail"

    if key.startswith("sentiment") and isinstance(score, (int, float)):
        return "Neutral"
    return "Pass" if success is not False else "Fail"


def _derive_passed(label: str | None, success: Any) -> bool | None:
    if isinstance(success, bool):
        return success
    if not label:
        return None
    normalized = label.strip()
    if normalized in {
        "Relevant",
        "Not Hallucinated",
        "Not Toxic",
        "Not Biased",
        "Positive",
        "Neutral",
        "Pass",
    }:
        return True
    if normalized in {
        "Irrelevant",
        "Hallucinated",
        "Toxic",
        "Biased",
        "Negative",
        "Fail",
    }:
        return False
    return None


def _apply_sentiment_postprocessing(
    ctx: _MetricContext, label: str | None
) -> tuple[float | None, str | None]:
    if ctx.name not in {
        "sentiment",
        "sentiment [geval]",
        "sentiment [geval] [GEval]",
    }:
        return ctx.score, label
    if ctx.score is None:
        return ctx.score, label
    try:
        compound = max(-1.0, min(1.0, float(ctx.score)))
    except Exception:
        return ctx.score, label
    mapped = (compound + 1.0) / 2.0
    ctx.attributes.setdefault(
        "deepeval.sentiment.compound", round(compound, 6)
    )

    if label is None:
        if compound >= 0.25:
            label = "Positive"
        elif compound <= -0.25:
            label = "Negative"
        else:
            label = "Neutral"

    try:
        neg_strength = 1 - mapped
        pos_strength = mapped
        neu_strength = 1 - abs(compound)
        total = neg_strength + neu_strength + pos_strength
        if total > 0:
            neg_strength /= total
            neu_strength /= total
            pos_strength /= total
        ctx.attributes.update(
            {
                "deepeval.sentiment.neg": round(neg_strength, 6),
                "deepeval.sentiment.neu": round(neu_strength, 6),
                "deepeval.sentiment.pos": round(pos_strength, 6),
                "deepeval.sentiment.compound": round(compound, 6),
            }
        )
    except Exception:
        pass
    return mapped, label


_METRIC_REGISTRY: Mapping[str, str] = _DEEPEVAL_METRIC_REGISTRY


class DeepevalEvaluator(Evaluator):
    """Evaluator using Deepeval as an LLM-as-a-judge backend."""

    def __init__(
        self,
        metrics: Iterable[str] | None = None,
        *,
        invocation_type: str | None = None,
        options: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        super().__init__(
            metrics,
            invocation_type=invocation_type,
            options=options,
        )

    # ---- Defaults -----------------------------------------------------
    def default_metrics_by_type(self) -> Mapping[str, Sequence[str]]:
        return _DEFAULT_METRICS

    def default_metrics(self) -> Sequence[str]:  # pragma: no cover - fallback
        return _DEFAULT_METRICS["LLMInvocation"]

    # ---- Evaluation ---------------------------------------------------
    def evaluate(self, item: GenAI) -> list[EvaluationResult]:
        if isinstance(item, LLMInvocation):
            return list(self._evaluate_llm(item))
        if isinstance(item, AgentInvocation):
            return list(self._evaluate_agent(item))
        return []

    def _evaluate_llm(
        self, invocation: LLMInvocation
    ) -> Sequence[EvaluationResult]:
        # Tool-call only detection handled centrally by Manager.
        return self._evaluate_generic(invocation, "LLMInvocation")

    def _evaluate_agent(
        self, invocation: AgentInvocation
    ) -> Sequence[EvaluationResult]:
        # only evaluate for operation=invoke
        operation = getattr(invocation, "operation", None)
        if operation != "invoke_agent":
            try:
                genai_debug_log(
                    "evaluator.deepeval.skip.non_invoke_agent",
                    invocation,
                    operation=invocation.operation,
                )
            except Exception:  # pragma: no cover
                pass
            return []
        return self._evaluate_generic(invocation, "AgentInvocation")

    def _evaluate_generic(
        self, invocation: GenAI, invocation_type: str
    ) -> Sequence[EvaluationResult]:
        try:
            genai_debug_log(
                "evaluator.deepeval.start",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                invocation_type=invocation_type,
            )
        except Exception:  # pragma: no cover
            pass
        metric_specs = self._build_metric_specs()
        if not metric_specs:
            genai_debug_log(
                "evaluator.deepeval.skip.no_metrics",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                invocation_type=invocation_type,
            )
            return []
        test_case = self._build_test_case(invocation, invocation_type)
        if test_case is None:
            genai_debug_log(
                "evaluator.deepeval.error.missing_io",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                invocation_type=invocation_type,
            )
            return self._error_results(
                "Deepeval requires both input and output text to evaluate",
                ValueError,
            )
        # Ensure OpenAI API key is available for Deepeval metrics that rely on OpenAI.
        # Resolution order:
        # 1. Explicit in invocation.attributes['openai_api_key'] (if provided)
        # 2. Environment OPENAI_API_KEY
        # 3. Environment GENAI_OPENAI_API_KEY (custom fallback)
        # If unavailable we mark all metrics skipped with a clear explanation instead of raising.
        api_key: str | None = None
        try:
            raw_attrs = getattr(invocation, "attributes", None)
            attrs: dict[str, Any] = {}
            if isinstance(raw_attrs, MappingABC):
                for k, v in raw_attrs.items():
                    try:
                        attrs[str(k)] = v
                    except Exception:  # pragma: no cover
                        continue
            candidate_val = attrs.get("openai_api_key") or attrs.get("api_key")
            candidate: str | None = (
                str(candidate_val)
                if isinstance(candidate_val, (str, bytes))
                else None
            )
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv(
                "GENAI_OPENAI_API_KEY"
            )
            api_key = candidate or env_key
            if api_key:
                # Attempt to configure Deepeval/OpenAI client.
                try:  # pragma: no cover - external dependency
                    # Support legacy openai<1 and new openai>=1 semantics.
                    if not getattr(openai, "api_key", None):  # type: ignore[attr-defined]
                        try:
                            setattr(openai, "api_key", api_key)  # legacy style
                        except Exception:  # pragma: no cover
                            pass
                    # Ensure env var set for client() style usage.
                    if not os.getenv("OPENAI_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = api_key
                except Exception:
                    pass
        except Exception:  # pragma: no cover - defensive
            api_key = None
        try:
            genai_debug_log(
                "evaluator.deepeval.resolve_api_key",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                has_key=bool(api_key),
            )
        except Exception:
            pass
        # Do not fail early if API key missing; underlying Deepeval/OpenAI usage
        # will produce an error which we surface as evaluation error results.
        try:
            metrics, skipped_results = _instantiate_metrics(
                metric_specs, test_case, self._default_model()
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_results(str(exc), type(exc))
        if not metrics:
            genai_debug_log(
                "evaluator.deepeval.skip.no_valid_metrics",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                invocation_type=invocation_type,
            )
            return skipped_results or self._error_results(
                "No Deepeval metrics available", RuntimeError
            )
        try:
            evaluation = _run_deepeval(test_case, metrics, genai_debug_log)
            genai_debug_log(
                "evaluator.deepeval.complete",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                invocation_type=invocation_type,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - dependency/runtime failure
            genai_debug_log(
                "evaluator.deepeval.error.execution",
                invocation
                if isinstance(invocation, (LLMInvocation, AgentInvocation))
                else None,
                invocation_type=invocation_type,
            )
            return [
                *skipped_results,
                *self._error_results(str(exc), type(exc)),
            ]
        return [*skipped_results, *self._convert_results(evaluation)]

    # NOTE: unreachable code below; logging handled prior to return.

    # ---- Helpers ------------------------------------------------------
    def _build_metric_specs(self) -> Sequence[_MetricSpec]:
        specs: list[_MetricSpec] = []
        registry = _METRIC_REGISTRY

        for name in self.metrics:
            raw = (name or "").strip().lower()
            # Normalize any spaces / punctuation to underscores so that
            # variants like "answer relevancy" or "answer-relevance" resolve
            # to the canonical registry key "answer_relevancy".
            normalized = _re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
            key = normalized
            options = self.options.get(name, {})
            if key not in registry:
                specs.append(
                    _MetricSpec(
                        name=name,
                        options={
                            "__error__": f"Unknown Deepeval metric '{name}'",
                        },
                    )
                )
                continue
            parsed_options = {
                opt_key: _coerce_option(opt_value)
                for opt_key, opt_value in options.items()
            }
            specs.append(_MetricSpec(name=key, options=parsed_options))
        return specs

    # removed; see deepeval_metrics.instantiate_metrics

    # ---- Custom metric builders ------------------------------------
    # removed; see deepeval_metrics.build_hallucination_metric

    # removed; see deepeval_metrics.build_sentiment_metric

    def _build_test_case(
        self, invocation: GenAI, invocation_type: str
    ) -> Any | None:
        if isinstance(invocation, (LLMInvocation, AgentInvocation)):
            return _build_llm_test_case(invocation)
        return None

    # removed; see deepeval_runner.run_evaluation

    def _convert_results(self, evaluation: Any) -> Sequence[EvaluationResult]:
        results: list[EvaluationResult] = []
        try:
            test_results = getattr(evaluation, "test_results", [])
        except Exception:  # pragma: no cover - defensive
            return self._error_results(
                "Unexpected Deepeval response", RuntimeError
            )
        for test in test_results:
            metrics_data = getattr(test, "metrics_data", []) or []
            for metric in metrics_data:
                ctx = _build_metric_context(metric, test)
                label = _determine_label(ctx)
                score, label = _apply_sentiment_postprocessing(ctx, label)
                ctx.score = score
                passed = _derive_passed(label, ctx.success)

                result = EvaluationResult(
                    metric_name=ctx.name,
                    score=score,
                    label=label,
                    explanation=ctx.reason,
                    error=ctx.error,
                    attributes=ctx.attributes,
                )
                results.append(result)
                if passed is not None:
                    ctx.attributes["gen_ai.evaluation.passed"] = passed
        return results

    def _error_results(
        self, message: str, error_type: type[BaseException]
    ) -> Sequence[EvaluationResult]:
        _LOGGER.warning("Deepeval evaluation failed: %s", message)
        return [
            EvaluationResult(
                metric_name=metric,
                explanation=message,
                error=Error(message=message, type=error_type),
                attributes={"deepeval.error": message},
            )
            for metric in self.metrics
        ]

    @staticmethod
    def _coerce_option(value: Any) -> Any:
        # Best-effort recursive coercion; add explicit types to avoid Unknown complaints
        if isinstance(value, MappingABC):
            out: dict[Any, Any] = {}
            for k, v in value.items():  # type: ignore[assignment]
                out[k] = DeepevalEvaluator._coerce_option(v)
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

    # message serialization moved to normalizer

    @staticmethod
    # context extraction moved to normalizer

    @staticmethod
    # retrieval context extraction moved to normalizer

    @staticmethod
    # flatten helper moved to normalizer

    # per-metric param check handled in deepeval_metrics

    @staticmethod
    def _default_model() -> str | Any | None:
        """
        Get the default model for evaluations.

        Returns either:
        - A LiteLLMModel instance if DEEPEVAL_LLM_BASE_URL is configured
        - A model name string for standard OpenAI usage

        Environment Variables (for custom model):
            DEEPEVAL_LLM_BASE_URL: Custom LLM endpoint
            DEEPEVAL_LLM_MODEL: Model name
            DEEPEVAL_LLM_PROVIDER: Provider identifier
            DEEPEVAL_LLM_TOKEN_URL: OAuth2 token endpoint
            DEEPEVAL_LLM_CLIENT_ID: OAuth2 client ID
            DEEPEVAL_LLM_CLIENT_SECRET: OAuth2 client secret
            DEEPEVAL_LLM_CLIENT_APP_NAME: App key (for Cisco-style providers)
        """
        # Check for custom OAuth2/LiteLLM provider first
        try:
            custom_model = _create_eval_model()
            if custom_model is not None:
                return custom_model
        except Exception:
            pass  # Fall back to standard OpenAI

        # Fall back to model name for standard OpenAI
        model = (
            os.getenv("DEEPEVAL_EVALUATION_MODEL")
            or os.getenv("DEEPEVAL_MODEL")
            or os.getenv("OPENAI_MODEL")
        )
        if model:
            return model
        return "gpt-4o-mini"


def _factory(
    metrics: Iterable[str] | None = None,
    invocation_type: str | None = None,
    options: Mapping[str, Mapping[str, str]] | None = None,
) -> DeepevalEvaluator:
    return DeepevalEvaluator(
        metrics,
        invocation_type=invocation_type,
        options=options,
    )


_REGISTRATION = EvaluatorRegistration(
    factory=_factory,
    default_metrics_factory=lambda: _DEFAULT_METRICS,
)


def registration() -> EvaluatorRegistration:
    return _REGISTRATION


def register() -> None:
    register_evaluator(
        "deepeval",
        _REGISTRATION.factory,
        default_metrics=_REGISTRATION.default_metrics_factory,
    )


__all__ = [
    "DeepevalEvaluator",
    "registration",
    "register",
]
