"""Emitters responsible for emitting telemetry derived from evaluation results."""

from __future__ import annotations

import logging
import os
import re as _re
from typing import Any, Dict, Optional, Sequence

from opentelemetry._logs import Logger, get_logger
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)

from ..attributes import (
    GEN_AI_EVALUATION_ATTRIBUTES_PREFIX,
    GEN_AI_EVALUATION_EXPLANATION,
    GEN_AI_EVALUATION_NAME,
    GEN_AI_EVALUATION_SCORE_LABEL,
    GEN_AI_EVALUATION_SCORE_VALUE,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_ID,
    SERVER_ADDRESS,
    SERVER_PORT,
)
from ..interfaces import EmitterMeta
from ..span_context import (
    build_otel_context,
    extract_span_context,
    store_span_context,
)
from ..types import EvaluationResult, GenAI
from .utils import _evaluation_to_log_record


def _get_request_model(invocation: GenAI) -> str | None:
    return getattr(invocation, "request_model", None) or getattr(
        invocation, "model", None
    )


def _get_response_id(invocation: GenAI) -> str | None:  # best-effort
    return getattr(invocation, "response_id", None)


class _EvaluationEmitterBase(EmitterMeta):
    role = "evaluation"

    def on_start(self, obj: Any) -> None:  # pragma: no cover - default no-op
        return None

    def on_end(self, obj: Any) -> None:  # pragma: no cover - default no-op
        return None

    def on_error(
        self, error, obj: Any
    ) -> None:  # pragma: no cover - default no-op
        return None


def _canonicalize_metric_name(raw_name: str) -> Optional[str]:
    """Map raw evaluator metric names (possibly noisy) to canonical names.

    Handles legacy / provider-specific variants and formatting differences:
    - answer relevancy / answer_relevancy / answer relevance -> relevance
    - faithfulness -> hallucination (legacy synonym)
    - hallucination [geval] / hallucination_geval / hallucination-* -> hallucination
    - direct passthrough for: hallucination, sentiment, toxicity, bias
    Returns None if the metric is unsupported (ignored by emitters).
    """
    if not raw_name:
        return None
    lowered = raw_name.strip().lower()
    # Fast path exact matches first
    if lowered in {"bias", "toxicity", "sentiment", "hallucination"}:
        return lowered
    if lowered == "faithfulness":
        return "hallucination"
    # Normalize punctuation/whitespace to underscores for pattern matching
    normalized = _re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if normalized in {"answer_relevancy", "answer_relevance", "relevance"}:
        return "relevance"
    if normalized.startswith("hallucination"):
        return "hallucination"
    if normalized.startswith("sentiment"):
        # Allow variants like sentiment_geval, sentiment[geval], sentiment-geval
        return "sentiment"
    return None


# Debug logging configuration:
#   OTEL_GENAI_EVAL_DEBUG_SKIPS=1|true|yes  -> one-time logs when a measurement is skipped (already implemented)
#   OTEL_GENAI_EVAL_DEBUG_EACH=1|true|yes   -> verbose log line for every evaluation result processed (attempted measurement)
_EVAL_DEBUG_SKIPS = os.getenv("OTEL_GENAI_EVAL_DEBUG_SKIPS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_EVAL_DEBUG_EACH = os.getenv("OTEL_GENAI_EVAL_DEBUG_EACH", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class EvaluationMetricsEmitter(_EvaluationEmitterBase):
    """Records evaluation scores to metric-specific histograms.

    Instead of a single shared histogram (gen_ai.evaluation.score), we emit to
    gen_ai.evaluation.score.<metric_name>. This improves downstream aggregation
    clarity at the cost of additional instruments. A callable factory provided
    by the handler supplies (and caches) histogram instances.
    """

    role = "evaluation_metrics"
    name = "EvaluationMetrics"

    def __init__(
        self, histogram_factory
    ) -> None:  # callable(metric_name)->Histogram|None OR direct histogram
        # Backward-compatible: tests may pass a histogram instance directly.
        if hasattr(histogram_factory, "record") and not callable(  # type: ignore[arg-type]
            getattr(histogram_factory, "__call__", None)
        ):
            direct_hist = histogram_factory

            def _direct_factory(_name: str):  # ignore metric name, single hist
                return direct_hist

            self._hist_factory = _direct_factory
        else:
            self._hist_factory = histogram_factory

    def on_evaluation_results(  # type: ignore[override]
        self,
        results: Sequence[EvaluationResult],
        obj: Any | None = None,
    ) -> None:
        invocation = obj if isinstance(obj, GenAI) else None
        if invocation is None:
            if _EVAL_DEBUG_SKIPS:
                logging.getLogger(__name__).debug(
                    "EvaluationMetricsEmitter: skipping all results (no GenAI invocation provided)"
                )
            return
        # Per-emitter set of (reason, key) we have already logged to avoid noise.
        if not hasattr(self, "_logged_skip_keys"):
            self._logged_skip_keys = set()  # type: ignore[attr-defined]

        span_context = getattr(invocation, "span_context", None)
        if (
            span_context is None
            and getattr(invocation, "span", None) is not None
        ):
            span_context = extract_span_context(invocation.span)
            store_span_context(invocation, span_context)
        otel_context = build_otel_context(
            getattr(invocation, "span", None),
            span_context,
        )

        def _log_skip(
            reason: str,
            metric_raw: Any,
            extra: Optional[Dict[str, Any]] = None,
        ):
            if not _EVAL_DEBUG_SKIPS:
                return
            key = (reason, str(metric_raw))
            try:
                if key in self._logged_skip_keys:  # type: ignore[attr-defined]
                    return
                self._logged_skip_keys.add(key)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
            msg = f"EvaluationMetricsEmitter: skipped metric '{metric_raw}' reason={reason}"
            if extra:
                try:
                    msg += " " + " ".join(
                        f"{k}={v!r}" for k, v in extra.items() if v is not None
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            logging.getLogger(__name__).debug(msg)

        for res in results:
            canonical = _canonicalize_metric_name(
                getattr(res, "metric_name", "") or ""
            )
            raw_name = getattr(res, "metric_name", None)
            if _EVAL_DEBUG_EACH:
                logging.getLogger(__name__).debug(
                    "EvaluationMetricsEmitter: processing metric raw=%r canonical=%r score=%r type=%s label=%r",
                    raw_name,
                    canonical,
                    getattr(res, "score", None),
                    type(getattr(res, "score", None)).__name__,
                    getattr(res, "label", None),
                )
            if canonical is None:
                _log_skip("unsupported_metric_name", raw_name)
                continue
            if not isinstance(res.score, (int, float)):
                _log_skip(
                    "non_numeric_score",
                    raw_name,
                    {
                        "score_type": type(res.score).__name__,
                        "score_value": getattr(res, "score", None),
                    },
                )
                continue
            try:
                histogram = (
                    self._hist_factory(canonical)
                    if self._hist_factory
                    else None
                )  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                histogram = None
                _log_skip(
                    "histogram_factory_error", raw_name, {"error": repr(exc)}
                )
            if histogram is None:
                # Log once per metric name if histogram factory did not provide an instrument.
                try:
                    _once_key = f"_genai_eval_hist_missing_{canonical}"
                    if not getattr(self, _once_key, False):
                        logging.getLogger(__name__).debug(
                            "EvaluationMetricsEmitter: no histogram for canonical metric '%s' (factory returned None)",
                            canonical,
                        )
                        setattr(self, _once_key, True)
                except Exception:
                    pass
                _log_skip(
                    "no_histogram_instrument",
                    raw_name,
                    {"canonical": canonical},
                )
                continue
            elif _EVAL_DEBUG_EACH:
                logging.getLogger(__name__).debug(
                    "EvaluationMetricsEmitter: recording metric canonical=%r score=%r instrument=%s",
                    canonical,
                    getattr(res, "score", None),
                    type(histogram).__name__,
                )
            # Detect if we're in single metric mode by checking histogram name
            histogram_name = getattr(histogram, "name", None)
            is_single_metric_mode = histogram_name == "gen_ai.evaluation.score"

            attrs: Dict[str, Any] = {
                GEN_AI_EVALUATION_NAME: canonical,
            }
            # Only set gen_ai.operation.name for legacy multiple metric mode
            if not is_single_metric_mode:
                attrs[GEN_AI_OPERATION_NAME] = "evaluation"

            # If the source invocation carried agent identity, propagate
            agent_name = getattr(invocation, "agent_name", None)
            # Fallbacks: if instrumentation didn't populate agent_name field explicitly but
            # the invocation is an AgentInvocation, derive it from core fields to preserve identity.
            try:
                from opentelemetry.util.genai.types import (  # noqa: PLC0415
                    AgentInvocation as _AI,  # local import to avoid cycle
                )

                if agent_name is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_name = getattr(invocation, "name", None)
            except Exception:  # pragma: no cover - defensive
                pass

            # Set attributes based on mode
            if is_single_metric_mode:
                # Single metric mode: only include specified attributes
                if agent_name:
                    attrs["gen_ai.agent.name"] = agent_name
                req_model = _get_request_model(invocation)
                if req_model:
                    attrs[GEN_AI_REQUEST_MODEL] = req_model
                provider = getattr(invocation, "provider", None)
                if provider:
                    attrs[GEN_AI_PROVIDER_NAME] = provider
                # Add server.address and server.port when available
                server_address = getattr(invocation, "server_address", None)
                if server_address:
                    attrs[SERVER_ADDRESS] = server_address
                server_port = getattr(invocation, "server_port", None)
                if server_port is not None:
                    attrs[SERVER_PORT] = server_port
                if res.label is not None:
                    attrs[GEN_AI_EVALUATION_SCORE_LABEL] = res.label
                # Propagate evaluator-derived pass boolean if present
                passed = None
                try:
                    if isinstance(getattr(res, "attributes", None), dict):
                        passed = res.attributes.get("gen_ai.evaluation.passed")
                except Exception:  # pragma: no cover - defensive
                    passed = None
                if passed is None and res.label is not None:
                    label_text = str(res.label).strip().lower()
                    if label_text in {"pass", "passed"}:
                        passed = True
                    elif label_text in {"fail", "failed"}:
                        passed = False
                if isinstance(passed, bool):
                    attrs["gen_ai.evaluation.passed"] = passed
            else:
                # Legacy multiple metric mode: include all attributes
                if agent_name:
                    attrs["gen_ai.agent.name"] = agent_name
                workflow_id = getattr(invocation, "workflow_id", None)
                if workflow_id:
                    attrs["gen_ai.workflow.id"] = workflow_id
                req_model = _get_request_model(invocation)
                if req_model:
                    attrs[GEN_AI_REQUEST_MODEL] = req_model
                provider = getattr(invocation, "provider", None)
                if provider:
                    attrs[GEN_AI_PROVIDER_NAME] = provider
                # Add server.address and server.port when available
                server_address = getattr(invocation, "server_address", None)
                if server_address:
                    attrs[SERVER_ADDRESS] = server_address
                server_port = getattr(invocation, "server_port", None)
                if server_port is not None:
                    attrs[SERVER_PORT] = server_port
                if res.label is not None:
                    attrs[GEN_AI_EVALUATION_SCORE_LABEL] = res.label
                # Propagate evaluator-derived pass boolean if present
                passed = None
                try:
                    if isinstance(getattr(res, "attributes", None), dict):
                        passed = res.attributes.get("gen_ai.evaluation.passed")
                except Exception:  # pragma: no cover - defensive
                    passed = None
                if passed is None and res.label is not None:
                    label_text = str(res.label).strip().lower()
                    if label_text in {"pass", "passed"}:
                        passed = True
                    elif label_text in {"fail", "failed"}:
                        passed = False
                if isinstance(passed, bool):
                    attrs["gen_ai.evaluation.passed"] = passed
                attrs["gen_ai.evaluation.score.units"] = "score"
            if res.error is not None:
                if getattr(res.error, "message", None):
                    attrs[ErrorAttributes.ERROR_MESSAGE] = res.error.message
                if getattr(res.error, "type", None):
                    attrs[ErrorAttributes.ERROR_TYPE] = (
                        res.error.type.__qualname__
                    )
            try:
                if otel_context is not None:
                    histogram.record(  # type: ignore[attr-defined]
                        res.score,
                        attributes=attrs,
                        context=otel_context,
                    )
                else:
                    histogram.record(res.score, attributes=attrs)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                _log_skip(
                    "histogram_record_error", raw_name, {"error": repr(exc)}
                )
                if _EVAL_DEBUG_EACH:
                    logging.getLogger(__name__).debug(
                        "EvaluationMetricsEmitter: record failed canonical=%r score=%r error=%r",
                        canonical,
                        getattr(res, "score", None),
                        exc,
                    )
                pass


class EvaluationEventsEmitter(_EvaluationEmitterBase):
    """Emits one event per evaluation result."""

    role = "evaluation_events"
    name = "EvaluationEvents"

    def __init__(
        self,
        logger: Optional[Logger] = None,
        *,
        emit_legacy_event: bool = False,
    ) -> None:
        self._logger: Logger = logger or get_logger(__name__)
        self._emit_legacy_event = emit_legacy_event
        self._primary_event_name = "gen_ai.evaluation.result"
        self._legacy_event_name = "gen_ai.evaluation"
        self._py_logger = logging.getLogger(
            f"{__name__}.EvaluationEventsEmitter"
        )

    def on_evaluation_results(  # type: ignore[override]
        self,
        results: Sequence[EvaluationResult],
        obj: Any | None = None,
    ) -> None:
        invocation = obj if isinstance(obj, GenAI) else None
        if invocation is None or not results:
            return

        req_model = _get_request_model(invocation)
        provider = getattr(invocation, "provider", None)
        response_id = _get_response_id(invocation)

        for res in results:
            canonical = _canonicalize_metric_name(
                getattr(res, "metric_name", "") or ""
            )
            if canonical is None:
                continue
            base_attrs: Dict[str, Any] = {
                GEN_AI_OPERATION_NAME: "evaluation",
                GEN_AI_EVALUATION_NAME: canonical,
            }
            agent_name = getattr(invocation, "agent_name", None)
            agent_id = getattr(invocation, "agent_id", None)
            try:
                from opentelemetry.util.genai.types import (  # noqa: PLC0415
                    AgentInvocation as _AI,  # local import to avoid cycle
                )

                if agent_name is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_name = getattr(invocation, "name", None)
                if agent_id is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_id = str(getattr(invocation, "run_id", "")) or None
            except Exception:  # pragma: no cover - defensive
                pass
            workflow_id = getattr(invocation, "workflow_id", None)
            if agent_name:
                base_attrs["gen_ai.agent.name"] = agent_name
            if agent_id:
                base_attrs["gen_ai.agent.id"] = agent_id
            if workflow_id:
                base_attrs["gen_ai.workflow.id"] = workflow_id
            if req_model:
                base_attrs[GEN_AI_REQUEST_MODEL] = req_model
            if provider:
                base_attrs[GEN_AI_PROVIDER_NAME] = provider
            if response_id:
                base_attrs[GEN_AI_RESPONSE_ID] = response_id
            if isinstance(res.score, (int, float)):
                base_attrs[GEN_AI_EVALUATION_SCORE_VALUE] = res.score
            if res.label is not None:
                base_attrs[GEN_AI_EVALUATION_SCORE_LABEL] = res.label
            else:
                base_attrs[GEN_AI_EVALUATION_SCORE_LABEL] = "unknown"
            # Propagate pass boolean if available
            passed = None
            try:
                if isinstance(getattr(res, "attributes", None), dict):
                    passed = res.attributes.get("gen_ai.evaluation.passed")
            except Exception:  # pragma: no cover - defensive
                passed = None
            if isinstance(passed, bool):
                base_attrs["gen_ai.evaluation.passed"] = passed
            if isinstance(res.score, (int, float)):
                base_attrs["gen_ai.evaluation.score.units"] = "score"
            if res.error is not None:
                if getattr(res.error, "message", None):
                    base_attrs[ErrorAttributes.ERROR_MESSAGE] = (
                        res.error.message
                    )
                if getattr(res.error, "type", None):
                    base_attrs[ErrorAttributes.ERROR_TYPE] = (
                        res.error.type.__qualname__
                    )

            spec_attrs = dict(base_attrs)
            if res.explanation:
                spec_attrs[GEN_AI_EVALUATION_EXPLANATION] = res.explanation
            if res.attributes:
                for key, value in dict(res.attributes).items():
                    key_str = str(key)
                    spec_attrs[
                        f"{GEN_AI_EVALUATION_ATTRIBUTES_PREFIX}{key_str}"
                    ] = value
            if res.error is not None:
                if getattr(res.error, "message", None):
                    spec_attrs[ErrorAttributes.ERROR_MESSAGE] = (
                        res.error.message
                    )
                if getattr(res.error, "type", None):
                    spec_attrs[ErrorAttributes.ERROR_TYPE] = (
                        res.error.type.__qualname__
                    )

            primary_body: Dict[str, Any] = {}
            if isinstance(res.score, (int, float)):
                primary_body["score"] = res.score
            elif res.score is not None:
                primary_body["score"] = res.score
            if res.label is not None:
                primary_body["label"] = res.label
            if res.explanation:
                primary_body["explanation"] = res.explanation
            if res.attributes:
                primary_body["attributes"] = dict(res.attributes)
            if res.error is not None:
                primary_body["error"] = {
                    "type": res.error.type.__qualname__,
                    "message": getattr(res.error, "message", None),
                }

            try:
                record = _evaluation_to_log_record(
                    invocation,
                    self._primary_event_name,
                    spec_attrs,
                    body=primary_body or None,
                )
                self._logger.emit(record)
                if self._py_logger.isEnabledFor(logging.DEBUG):
                    self._py_logger.debug(
                        "Emitted evaluation log event metric=%s trace_id=%s span_id=%s",
                        canonical,
                        getattr(invocation, "trace_id", None),
                        getattr(invocation, "span_id", None),
                    )
            except Exception:  # pragma: no cover - defensive
                if self._py_logger.isEnabledFor(logging.DEBUG):
                    self._py_logger.debug(
                        "Failed to emit evaluation log event", exc_info=True
                    )

            if not self._emit_legacy_event:
                continue

            legacy_attrs = dict(base_attrs)
            legacy_body: Dict[str, Any] = {}
            if res.explanation:
                legacy_body["gen_ai.evaluation.explanation"] = res.explanation
            if res.attributes:
                legacy_body["gen_ai.evaluation.attributes"] = dict(
                    res.attributes
                )
            if res.error is not None and getattr(res.error, "message", None):
                legacy_attrs["error.message"] = res.error.message

            try:
                legacy_record = _evaluation_to_log_record(
                    invocation,
                    self._legacy_event_name,
                    legacy_attrs,
                    body=legacy_body or None,
                )
                self._logger.emit(legacy_record)
            except Exception:  # pragma: no cover - defensive
                if self._py_logger.isEnabledFor(logging.DEBUG):
                    self._py_logger.debug(
                        "Failed to emit legacy evaluation log event",
                        exc_info=True,
                    )


__all__ = [
    "EvaluationMetricsEmitter",
    "EvaluationEventsEmitter",
]
