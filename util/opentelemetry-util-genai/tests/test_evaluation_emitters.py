from __future__ import annotations

from typing import Any

from opentelemetry.util.genai.emitters.evaluation import (
    EvaluationEventsEmitter,
)
from opentelemetry.util.genai.types import (
    Error,
    EvaluationResult,
    LLMInvocation,
)


class _RecordingEventLogger:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def emit(self, event: Any) -> None:
        self.records.append(event)


def _build_invocation() -> LLMInvocation:
    invocation = LLMInvocation(request_model="gpt-test")
    invocation.provider = "openai"
    invocation.response_id = "resp-123"
    invocation.trace_id = 0x1234
    invocation.span_id = 0xABCD
    invocation.trace_flags = 1
    return invocation


def test_spec_event_emission_uses_semconv_attributes() -> None:
    logger = _RecordingEventLogger()
    emitter = EvaluationEventsEmitter(logger)
    invocation = _build_invocation()
    result = EvaluationResult(
        metric_name="bias",
        score=0.75,
        label="medium",
        explanation="Detected mild bias",
        attributes={"judge_model": "gpt-5-nano", 1: "int-key"},
        error=Error(message="timeout", type=TimeoutError),
    )

    emitter.on_evaluation_results([result], invocation)

    assert len(logger.records) == 1
    event = logger.records[0]
    assert event.event_name == "gen_ai.evaluation.result"
    assert event.trace_id == invocation.trace_id
    assert event.span_id == invocation.span_id
    attrs = event.attributes
    assert attrs["gen_ai.evaluation.name"] == "bias"
    assert attrs["gen_ai.evaluation.score.value"] == 0.75
    assert attrs["gen_ai.evaluation.explanation"] == "Detected mild bias"
    assert attrs["gen_ai.evaluation.attributes.judge_model"] == "gpt-5-nano"
    assert attrs["gen_ai.evaluation.attributes.1"] == "int-key"
    assert attrs["error.message"] == "timeout"
    assert attrs["error.type"] == "TimeoutError"
    assert event.body == {
        "score": 0.75,
        "label": "medium",
        "explanation": "Detected mild bias",
        "attributes": {"judge_model": "gpt-5-nano", 1: "int-key"},
        "error": {"type": "TimeoutError", "message": "timeout"},
    }


def test_legacy_event_emission_when_flag_enabled() -> None:
    logger = _RecordingEventLogger()
    emitter = EvaluationEventsEmitter(logger, emit_legacy_event=True)
    invocation = _build_invocation()
    result = EvaluationResult(
        metric_name="toxicity",
        explanation="All clear",
        attributes={"detail": "sample"},
        error=Error(message="failure", type=RuntimeError),
    )

    emitter.on_evaluation_results([result], invocation)

    assert len(logger.records) == 2
    new_event, legacy_event = logger.records
    assert new_event.event_name == "gen_ai.evaluation.result"
    assert legacy_event.event_name == "gen_ai.evaluation"
    assert new_event.trace_id == invocation.trace_id
    assert new_event.span_id == invocation.span_id
    assert legacy_event.trace_id == invocation.trace_id
    assert legacy_event.span_id == invocation.span_id
    assert legacy_event.body == {
        "gen_ai.evaluation.explanation": "All clear",
        "gen_ai.evaluation.attributes": {"detail": "sample"},
    }
    assert legacy_event.attributes["error.message"] == "failure"
