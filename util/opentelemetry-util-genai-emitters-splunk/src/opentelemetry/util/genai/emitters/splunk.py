from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

# NOTE: We intentionally rely on the core ("original") evaluation metrics emitter
# for recording canonical evaluation metrics. The Splunk emitters now focus solely
# on providing a custom aggregated event schema for evaluation results and do NOT
# emit their own metrics to avoid duplication or confusion.
from opentelemetry.util.genai.emitters.spec import EmitterSpec
from opentelemetry.util.genai.emitters.utils import (
    _agent_to_log_record,
    _evaluation_to_log_record,
    _llm_invocation_to_log_record,
    _workflow_to_log_record,
)
from opentelemetry.util.genai.interfaces import EmitterMeta
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EvaluationResult,
    GenAI,
    LLMInvocation,
    Workflow,
)

try:  # optional debug logging
    from opentelemetry.util.genai.debug import genai_debug_log
except Exception:  # pragma: no cover

    def genai_debug_log(*_a: Any, **_k: Any) -> None:  # type: ignore
        return None


_LOGGER = logging.getLogger(__name__)

_EVENT_NAME_EVALUATIONS = "gen_ai.evaluation.results"
_RANGE_ATTRIBUTE_KEYS = (
    "score_range",
    "range",
    "score-range",
    "scoreRange",
    "range_values",
)
_MIN_ATTRIBUTE_KEYS = (
    "range_min",
    "score_min",
    "min",
    "lower_bound",
    "lower",
)
_MAX_ATTRIBUTE_KEYS = (
    "range_max",
    "score_max",
    "max",
    "upper_bound",
    "upper",
)

_INCLUDE_EVALUATION_MESSAGE_CONTENT = (
    os.environ.get("SPLUNK_EVALUATION_RESULTS_MESSAGE_CONTENT", "").lower()
    == "true"
)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_range_spec(value: Any) -> Optional[Tuple[float, float]]:
    # Elements may be heterogeneous/unknown; length check is safe.
    if isinstance(value, (list, tuple)) and len(value) >= 2:  # type: ignore[arg-type]
        start = _to_float(value[0])
        end = _to_float(value[1])
        if start is not None and end is not None:
            return start, end
    if isinstance(value, Mapping):
        start = None
        end = None
        for key in ("min", "lower", "start", "from", "low"):
            if key in value:
                start = _to_float(value[key])
                break
        for key in ("max", "upper", "end", "to", "high"):
            if key in value:
                end = _to_float(value[key])
                break
        if start is not None and end is not None:
            return start, end
    if isinstance(value, str):
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if len(matches) >= 2:
            start = _to_float(matches[0])
            end = _to_float(matches[1])
            if start is not None and end is not None:
                return start, end
    return None


def _extract_range(
    attributes: Mapping[str, Any],
) -> Optional[Tuple[float, float]]:
    for key in _RANGE_ATTRIBUTE_KEYS:
        if key in attributes:
            bounds = _parse_range_spec(attributes[key])
            if bounds is not None:
                return bounds
    start = None
    end = None
    for key in _MIN_ATTRIBUTE_KEYS:
        if key in attributes:
            start = _to_float(attributes[key])
            if start is not None:
                break
    for key in _MAX_ATTRIBUTE_KEYS:
        if key in attributes:
            end = _to_float(attributes[key])
            if end is not None:
                break
    if start is not None and end is not None:
        return start, end
    return None


# _sanitize_metric_suffix retained historically; removed after metrics pruning.


class SplunkConversationEventsEmitter(EmitterMeta):
    """Emit semantic-convention conversation / invocation events for LLM & Agent.

    Backward compatibility with the older custom 'gen_ai.splunk.conversation' event
    has been intentionally removed in this development branch.
    """

    role = "content_event"
    name = "splunk_conversation_event"

    def __init__(
        self, event_logger: Any, capture_content: bool = False
    ) -> None:
        self._event_logger = event_logger
        self._capture_content = capture_content

    def handles(self, obj: Any) -> bool:
        try:
            genai_debug_log(
                "splunk_evaluations.handles",
                obj if isinstance(obj, GenAI) else None,
                accepted=True,
                obj_type=type(obj).__name__ if obj is not None else None,
            )
        except Exception:  # pragma: no cover
            pass
        return True

    def on_start(self, obj: Any) -> None:
        return None

    def on_end(self, obj: Any) -> None:
        if self._event_logger is None:
            return
        # Emit semantic convention-aligned events for Workflow, LLM & Agent invocations.
        if isinstance(obj, Workflow):
            try:
                genai_debug_log(
                    "emitter.splunk.conversation.on_end.workflow",
                    obj,
                )
            except Exception:  # pragma: no cover
                pass
            try:
                rec = _workflow_to_log_record(obj, self._capture_content)
                if rec:
                    self._event_logger.emit(rec)
            except Exception:  # pragma: no cover - defensive
                pass
        elif isinstance(obj, LLMInvocation):
            try:
                genai_debug_log(
                    "emitter.splunk.conversation.on_end",
                    obj,
                    output_messages=len(
                        getattr(obj, "output_messages", []) or []
                    ),
                )
            except Exception:  # pragma: no cover
                pass
            try:
                rec = _llm_invocation_to_log_record(obj, self._capture_content)
                if rec:
                    self._event_logger.emit(rec)
            except Exception:  # pragma: no cover - defensive
                pass
        elif isinstance(obj, AgentInvocation):
            try:
                genai_debug_log(
                    "emitter.splunk.conversation.on_end.agent",
                    obj,
                )
            except Exception:  # pragma: no cover
                pass
            try:
                rec = _agent_to_log_record(obj, self._capture_content)
                if rec:
                    self._event_logger.emit(rec)
            except Exception:  # pragma: no cover - defensive
                pass

    def on_error(self, error: Any, obj: Any) -> None:
        return None

    def on_evaluation_results(
        self, results: Any, obj: Any | None = None
    ) -> None:
        return None


class SplunkEvaluationResultsEmitter(EmitterMeta):
    """Aggregate evaluation results for Splunk ingestion (events only).

    Metrics emission has been removed; canonical evaluation metrics are handled
    by the core evaluation metrics emitter. This class now buffers evaluation
    results per invocation and emits a single aggregated event at invocation end.
    """

    role = "evaluation_results"
    name = "splunk_evaluation_results"

    def __init__(
        self,
        event_logger: Any,
        capture_content: bool = False,
        *_deprecated_args: Any,
        **_deprecated_kwargs: Any,
    ) -> None:
        self._event_logger = event_logger
        self._capture_content = capture_content

    def handles(self, obj: Any) -> bool:
        return isinstance(obj, (Workflow, LLMInvocation, AgentInvocation))

    # Explicit no-op implementations to satisfy emitter protocol expectations
    def on_start(self, obj: Any) -> None:  # pragma: no cover - no-op
        return None

    def on_error(self, error: Any, obj: Any) -> None:  # pragma: no cover
        return None

    def on_evaluation_results(
        self,
        results: Sequence[EvaluationResult],
        obj: Any | None = None,
    ) -> None:
        invocation = (
            obj if isinstance(obj, (LLMInvocation, AgentInvocation)) else None
        )
        if invocation is None:
            try:
                genai_debug_log(
                    "emitter.splunk.evaluations.skip",
                    None,
                    reason="unsupported_invocation_type",
                )
            except Exception:  # pragma: no cover
                pass
            return
        if not results:
            try:
                genai_debug_log(
                    "emitter.splunk.evaluations.skip",
                    invocation,
                    reason="empty_results",
                )
            except Exception:  # pragma: no cover
                pass
            return
        # Manager now handles aggregation; it emits either one aggregated batch
        # or multiple smaller batches. Each call here represents what should be
        # a single Splunk event.
        enriched: List[
            Tuple[EvaluationResult, Optional[float], Optional[str]]
        ] = []
        for r in results:
            normalized, range_label = self._compute_normalized_score(r)
            enriched.append((r, normalized, range_label))
        self._emit_event(invocation, enriched)

    def on_end(self, obj: Any) -> None:
        return None

    # on_error handled above

    def _emit_event(
        self,
        invocation: GenAI,
        records: List[Tuple[EvaluationResult, Optional[float], Optional[str]]],
    ) -> None:
        if not records or self._event_logger is None:
            try:
                genai_debug_log(
                    "emitter.splunk.evaluations.skip",
                    invocation if isinstance(invocation, GenAI) else None,
                    reason="no_records_or_logger",
                    record_count=len(records),
                )
            except Exception:  # pragma: no cover
                pass
            return
        try:
            genai_debug_log(
                "emitter.splunk.evaluations.emit",
                invocation,
                records_count=len(records),
            )
        except Exception:  # pragma: no cover
            pass
        # Build messages & system instructions (opt-in via env var)
        input_messages = None
        output_messages = None
        system_instructions: Sequence[Any] = []
        if _INCLUDE_EVALUATION_MESSAGE_CONTENT:
            input_messages = (
                _coerce_messages(
                    invocation.input_messages, self._capture_content
                )
                if isinstance(invocation, LLMInvocation)
                else _coerce_iterable(
                    getattr(invocation, "input_context", None)
                )
            )
            output_messages = (
                _coerce_messages(
                    invocation.output_messages, self._capture_content
                )
                if isinstance(invocation, LLMInvocation)
                else _coerce_iterable(
                    getattr(invocation, "output_result", None)
                )
            )
            system_instruction = None
            if isinstance(invocation, LLMInvocation):
                system_instruction = invocation.attributes.get(
                    "system_instruction"
                ) or invocation.attributes.get("system_instructions")
                if not system_instruction and getattr(
                    invocation, "system", None
                ):
                    system_instruction = invocation.system
            elif isinstance(invocation, AgentInvocation):
                system_instruction = getattr(
                    invocation, "system_instructions", None
                )
            system_instructions = (
                _coerce_iterable(system_instruction)
                if system_instruction is not None
                else []
            )

        # Span / invocation attributes used as baseline
        attrs: Dict[str, Any] = {
            "event.name": _EVENT_NAME_EVALUATIONS,
            # Distinguish this aggregated evaluation logical operation
            "gen_ai.operation.name": "data_evaluation_results",
        }
        # Merge underlying span attributes first (APM attributes requirement)
        span_attr_map: Dict[str, Any] = {}
        if invocation.span and hasattr(invocation.span, "attributes"):
            try:  # pragma: no cover - defensive
                span_attr_map = dict(invocation.span.attributes)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                span_attr_map = {}
        for k, v in span_attr_map.items():
            attrs.setdefault(k, v)
        # Merge invocation-level attributes (excluding those we explicitly derive)
        for k, v in (invocation.attributes or {}).items():
            if k in ("system_instruction", "system_instructions"):
                continue
            attrs.setdefault(k, v)
        if not _INCLUDE_EVALUATION_MESSAGE_CONTENT:
            # Ensure message content never leaks from span/invocation attributes.
            for key in (
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.system.instructions",
                "gen_ai.system_instructions",
            ):
                attrs.pop(key, None)
        if getattr(invocation, "provider", None):
            attrs["gen_ai.system"] = invocation.provider
            attrs["gen_ai.provider.name"] = invocation.provider
        model_name = getattr(invocation, "request_model", None) or getattr(
            invocation, "model", None
        )
        if model_name:
            attrs["gen_ai.request.model"] = model_name
        resp_id = getattr(invocation, "response_id", None)
        response_id_value: Optional[str] = None
        if isinstance(resp_id, str) and resp_id:
            attrs["gen_ai.response.id"] = resp_id
            response_id_value = resp_id
        elif invocation.attributes:
            candidate_resp_id = invocation.attributes.get("gen_ai.response.id")
            if isinstance(candidate_resp_id, str) and candidate_resp_id:
                response_id_value = candidate_resp_id
        if getattr(invocation, "response_model_name", None):
            attrs["gen_ai.response.model"] = invocation.response_model_name
        # Usage tokens if available
        if getattr(invocation, "input_tokens", None) is not None:
            attrs["gen_ai.usage.input_tokens"] = invocation.input_tokens
        if getattr(invocation, "output_tokens", None) is not None:
            attrs["gen_ai.usage.output_tokens"] = invocation.output_tokens
        # Finish reasons (aggregate from output messages)
        finish_reasons: List[str] = []
        if isinstance(invocation, LLMInvocation):
            for msg in invocation.output_messages or []:
                fr = getattr(msg, "finish_reason", None) or getattr(
                    msg, "finish_reasons", None
                )
                if fr:
                    if isinstance(fr, (list, tuple)):
                        finish_reasons.extend([str(x) for x in fr])  # type: ignore[arg-type]
                    else:
                        finish_reasons.append(str(fr))
        if finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = finish_reasons
        if isinstance(invocation, AgentInvocation):
            if getattr(invocation, "name", None):
                attrs["gen_ai.agent.name"] = invocation.name
            agent_type = getattr(invocation, "agent_type", None)
            if agent_type:
                attrs["gen_ai.agent.type"] = agent_type
            description = getattr(invocation, "description", None)
            if description:
                attrs["gen_ai.agent.description"] = description
            attrs["gen_ai.agent.id"] = str(invocation.run_id)

        # Evaluation results array
        evaluations: list[Dict[str, Any]] = []
        for (
            result,
            _normalized,
            _range_label,
        ) in (
            records
        ):  # normalized retained only for potential future enrichment
            ev: Dict[str, Any] = {}
            metric_name = self._canonical_metric_name(result.metric_name)
            if metric_name:
                ev["gen_ai.evaluation.name"] = metric_name
            elif getattr(result, "metric_name", None):
                fallback_name = str(result.metric_name).strip().lower()
                if fallback_name:
                    ev["gen_ai.evaluation.name"] = fallback_name
            if isinstance(result.score, (int, float)):
                ev["gen_ai.evaluation.score.value"] = result.score
            if result.label is not None:
                ev["gen_ai.evaluation.score.label"] = result.label
            if result.explanation:
                ev["gen_ai.evaluation.explanation"] = result.explanation
            passed_value: Optional[bool] = None
            attr_response_id: Optional[str] = None
            if result.attributes:
                passed_attr = result.attributes.get("gen_ai.evaluation.passed")
                if isinstance(passed_attr, bool):
                    passed_value = passed_attr
                elif isinstance(passed_attr, str):
                    lowered = passed_attr.strip().lower()
                    if lowered in {"true", "false"}:
                        passed_value = lowered == "true"
                candidate_resp = result.attributes.get("gen_ai.response.id")
                if isinstance(candidate_resp, str) and candidate_resp:
                    attr_response_id = candidate_resp
            if passed_value is not None:
                ev["gen_ai.evaluation.passed"] = passed_value
            response_identifier = attr_response_id or response_id_value
            if response_identifier:
                ev["gen_ai.response.id"] = response_identifier
            if result.error is not None:
                ev["error.type"] = result.error.type.__qualname__
                if getattr(result.error, "message", None):
                    ev["error.message"] = result.error.message
            if ev:
                evaluations.append(ev)

        # Add conversation content arrays
        if _INCLUDE_EVALUATION_MESSAGE_CONTENT:
            if input_messages:
                attrs["gen_ai.input.messages"] = input_messages
            if output_messages:
                attrs["gen_ai.output.messages"] = output_messages
            if system_instructions:
                attrs["gen_ai.system_instructions"] = system_instructions

        # Trace/span correlation
        span_context = (
            invocation.span.get_span_context() if invocation.span else None
        )
        trace_id_hex = None
        span_id_hex = None
        if span_context and getattr(span_context, "is_valid", False):
            trace_id_hex = f"{span_context.trace_id:032x}"
            span_id_hex = f"{span_context.span_id:016x}"
            # Also attach as attributes for downstream search (Splunk style)
            attrs.setdefault("trace_id", trace_id_hex)
            attrs.setdefault("span_id", span_id_hex)

        # SDKLogRecord signature in current OTel version used elsewhere: body, attributes, event_name
        body = {"gen_ai.evaluations": evaluations}
        if _INCLUDE_EVALUATION_MESSAGE_CONTENT:
            body["gen_ai.system_instructions"] = system_instructions or None
            body["gen_ai.input.messages"] = input_messages or None
            body["gen_ai.output.messages"] = output_messages or None
        record = _evaluation_to_log_record(
            invocation,
            _EVENT_NAME_EVALUATIONS,
            attrs,
            body=body,
        )
        if record is None:
            try:
                genai_debug_log(
                    "emitter.splunk.evaluations.skip",
                    invocation if isinstance(invocation, GenAI) else None,
                    reason="record_none",
                )
            except Exception:  # pragma: no cover
                pass
            return
        try:
            self._event_logger.emit(record)
            try:
                genai_debug_log(
                    "emitter.splunk.evaluations.emitted",
                    invocation if isinstance(invocation, GenAI) else None,
                    record_count=len(records),
                )
            except Exception:  # pragma: no cover
                pass
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.debug(
                "SplunkEvaluationResultsEmitter failed to emit evaluation event",
                exc_info=True,
            )
            try:
                genai_debug_log(
                    "emitter.splunk.evaluations.error",
                    invocation if isinstance(invocation, GenAI) else None,
                    error=repr(exc),
                )
            except Exception:  # pragma: no cover
                pass

    # _record_metric removed (metrics no longer emitted)

    def _compute_normalized_score(
        self, result: EvaluationResult
    ) -> Tuple[Optional[float], Optional[str]]:
        score = result.score
        if not isinstance(score, (int, float)):
            return None, None
        score_f = float(score)
        if 0.0 <= score_f <= 1.0:
            return score_f, "[0,1]"
        attributes = result.attributes or {}
        bounds = _extract_range(attributes)
        if bounds is None:
            _LOGGER.debug(
                "Skipping metric for '%s': score %.3f outside [0,1] with no range",
                result.metric_name,
                score_f,
            )
            return None, None
        start, end = bounds
        # start/end are floats here; retain defensive shape check
        if end <= start:
            _LOGGER.debug(
                "Invalid range %s for metric '%s'", bounds, result.metric_name
            )
            return None, None
        if start != 0:
            _LOGGER.debug(
                "Range for metric '%s' starts at %s (expected 0)",
                result.metric_name,
                start,
            )
        normalized = (score_f - start) / (end - start)
        if normalized < 0 or normalized > 1:
            _LOGGER.debug(
                "Score %.3f for metric '%s' outside range %s; clamping",
                score_f,
                result.metric_name,
                bounds,
            )
        normalized = max(0.0, min(1.0, normalized))
        return normalized, f"[{start},{end}]"

    @staticmethod
    def _canonical_metric_name(metric_name: Optional[str]) -> str:
        if not metric_name:
            return ""
        text = str(metric_name).strip().lower()
        if not text:
            return ""
        text = re.sub(r"\s*\[.*\]\s*$", "", text)
        text = text.replace(" ", "_")
        mapping = {
            "answer_relevancy": "relevance",
            "answer_relevance": "relevance",
            "answer_relevancy_metric": "relevance",
            "answer_relevance_metric": "relevance",
        }
        return mapping.get(text, text)

    def _serialize_result(
        self,
        result: EvaluationResult,
        normalized: Optional[float],
        range_label: Optional[str],
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"name": result.metric_name}
        if result.score is not None:
            entry["score"] = result.score
        if normalized is not None:
            entry["normalized_score"] = normalized
        if range_label:
            entry["range"] = range_label
        if result.label is not None:
            entry["label"] = result.label
        if result.explanation:
            entry["explanation"] = result.explanation
        if result.attributes:
            entry["attributes"] = dict(result.attributes)
        if result.error is not None:
            entry["error"] = {
                "type": result.error.type.__qualname__,
                "message": result.error.message,
            }
        return entry


def splunk_emitters() -> list[EmitterSpec]:
    def _conversation_factory(ctx: Any) -> SplunkConversationEventsEmitter:
        capture_mode = getattr(ctx, "capture_event_content", False)
        return SplunkConversationEventsEmitter(
            event_logger=getattr(ctx, "event_logger", None),
            capture_content=cast(bool, capture_mode),
        )

    def _evaluation_factory(ctx: Any) -> SplunkEvaluationResultsEmitter:
        capture_mode = getattr(ctx, "capture_event_content", False)
        return SplunkEvaluationResultsEmitter(
            event_logger=getattr(ctx, "event_logger", None),
            capture_content=cast(bool, capture_mode),
        )

    return [
        EmitterSpec(
            name="SplunkConversationEvents",
            category="content_events",
            mode="replace-category",
            factory=_conversation_factory,
        ),
        EmitterSpec(
            name="SplunkEvaluationResults",
            category="evaluation",
            factory=_evaluation_factory,
        ),
    ]


def _coerce_messages(
    messages: Iterable[Any], capture_content: bool
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for msg in messages or []:
        data: Dict[str, Any]
        try:
            data = asdict(msg)
        except TypeError:
            if isinstance(msg, dict):
                data = cast(Dict[str, Any], dict(msg))  # type: ignore[arg-type]
            else:
                data = {"value": str(msg)}
        if not capture_content:
            parts = data.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "content" in part:
                    part["content"] = ""
        result.append(data)
    return result


def _coerce_iterable(values: Any) -> List[Any]:
    if isinstance(values, list):
        return cast(List[Any], values)
    if isinstance(values, tuple):
        return [*values]
    if values is None:
        return []
    return [values]


__all__ = [
    "SplunkConversationEventsEmitter",
    "SplunkEvaluationResultsEmitter",
    "splunk_emitters",
]
