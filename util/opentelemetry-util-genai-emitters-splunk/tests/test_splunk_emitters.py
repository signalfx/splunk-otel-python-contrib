from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.util.genai.emitters.spec import EmitterFactoryContext
from opentelemetry.util.genai.emitters.splunk import (
    SplunkConversationEventsEmitter,
    SplunkEvaluationResultsEmitter,
    splunk_emitters,
)
from opentelemetry.util.genai.types import (
    EvaluationResult,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


class _CapturingLogger:
    def __init__(self) -> None:
        self.records = []

    def emit(self, record) -> None:
        self.records.append(record)


class _FakeHistogram:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records = []

    def record(self, value, attributes=None) -> None:
        self.records.append((value, attributes or {}))


class _FakeMeter:
    def __init__(self) -> None:
        self.histograms: dict[str, _FakeHistogram] = {}

    def create_histogram(self, name, unit=None, description=None):
        histogram = _FakeHistogram(name)
        self.histograms[name] = histogram
        return histogram


def _build_invocation() -> LLMInvocation:
    invocation = LLMInvocation(request_model="gpt-test")
    invocation.provider = "openai"
    invocation.input_messages = [
        InputMessage(role="user", parts=[Text(content="Hello")])
    ]
    invocation.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[Text(content="Hi")],
            finish_reason="stop",
        )
    ]
    invocation.attributes["system_instruction"] = ["be nice"]
    return invocation


def test_splunk_emitters_specs() -> None:
    specs = splunk_emitters()
    categories = {spec.category for spec in specs}
    assert categories == {"content_events", "evaluation"}

    conversation_spec = next(
        spec for spec in specs if spec.category == "content_events"
    )
    evaluation_spec = next(
        spec for spec in specs if spec.category == "evaluation"
    )

    conversation_context = EmitterFactoryContext(
        tracer=None,
        meter=metrics.get_meter(__name__),
        event_logger=_CapturingLogger(),
        content_logger=None,
        evaluation_histogram=None,
        capture_span_content=False,
        capture_event_content=True,
    )
    conversation_emitter = conversation_spec.factory(conversation_context)
    assert isinstance(conversation_emitter, SplunkConversationEventsEmitter)

    evaluation_context = EmitterFactoryContext(
        tracer=None,
        meter=_FakeMeter(),
        event_logger=_CapturingLogger(),
        content_logger=None,
        evaluation_histogram=None,
        capture_span_content=False,
        capture_event_content=True,
    )
    evaluation_emitter = evaluation_spec.factory(evaluation_context)
    assert isinstance(evaluation_emitter, SplunkEvaluationResultsEmitter)


def test_conversation_event_emission() -> None:
    logger = _CapturingLogger()
    specs = splunk_emitters()
    conversation_spec = next(
        spec for spec in specs if spec.category == "content_events"
    )
    context = EmitterFactoryContext(
        tracer=None,
        meter=metrics.get_meter(__name__),
        event_logger=logger,
        content_logger=None,
        evaluation_histogram=None,
        capture_span_content=False,
        capture_event_content=True,
    )
    emitter = conversation_spec.factory(context)
    invocation = _build_invocation()

    emitter.on_end(invocation)

    assert logger.records
    record = logger.records[0]
    # Updated to match current implementation - uses semantic convention event name
    assert (
        record.attributes["event.name"]
        == "gen_ai.client.inference.operation.details"
    )
    assert record.body["gen_ai.input.messages"][0]["role"] == "user"
    assert record.body["gen_ai.output.messages"][0]["role"] == "assistant"


def test_evaluation_results_aggregation_and_metrics() -> None:
    import importlib
    import os

    # Enable message content inclusion for this test
    os.environ["SPLUNK_EVALUATION_RESULTS_MESSAGE_CONTENT"] = "true"
    try:
        # Reload module to pick up environment variable
        from opentelemetry.util.genai.emitters import splunk as splunk_module

        importlib.reload(splunk_module)

        logger = _CapturingLogger()
        meter = _FakeMeter()
        specs = splunk_module.splunk_emitters()
        evaluation_spec = next(
            spec for spec in specs if spec.category == "evaluation"
        )
        context = EmitterFactoryContext(
            tracer=None,
            meter=meter,
            event_logger=logger,
            content_logger=None,
            evaluation_histogram=None,
            capture_span_content=False,
            capture_event_content=True,
        )
        emitter = evaluation_spec.factory(context)
        invocation = _build_invocation()

        results = [
            EvaluationResult(
                metric_name="accuracy",
                score=3.0,
                label="medium",
                explanation="Normalized via range",
                attributes={"range": [0, 4], "judge_model": "llama3"},
            ),
            EvaluationResult(
                metric_name="toxicity/v1",
                score=0.2,
                label="low",
            ),
            EvaluationResult(
                metric_name="readability",
                score=5.0,
                label="high",
            ),
        ]

        emitter.on_evaluation_results(results, invocation)

        # Metrics emission has been removed from Splunk emitters
        # (canonical metrics are handled by core evaluation metrics emitter)
        # So we no longer check for histograms

        assert len(logger.records) == 1
        record = logger.records[0]
        # Updated event name to match current implementation
        assert record.attributes["event.name"] == "gen_ai.evaluation.results"
        # Updated body structure to match current implementation
        evaluations = record.body["gen_ai.evaluations"]
        assert len(evaluations) == 3

        accuracy_entry = next(
            e
            for e in evaluations
            if e.get("gen_ai.evaluation.name") == "accuracy"
        )
        assert accuracy_entry["gen_ai.evaluation.score.value"] == 3.0
        assert accuracy_entry["gen_ai.evaluation.score.label"] == "medium"

        toxicity_entry = next(
            e
            for e in evaluations
            if e.get("gen_ai.evaluation.name") == "toxicity/v1"
        )
        assert toxicity_entry["gen_ai.evaluation.score.value"] == 0.2
        assert toxicity_entry["gen_ai.evaluation.score.label"] == "low"

        readability_entry = next(
            e
            for e in evaluations
            if e.get("gen_ai.evaluation.name") == "readability"
        )
        assert readability_entry["gen_ai.evaluation.score.value"] == 5.0

        # Updated body structure for message content (when env var is set)
        input_messages = record.body["gen_ai.input.messages"]
        assert input_messages[0]["parts"][0]["content"] == "Hello"
        system_instructions = record.body["gen_ai.system_instructions"]
        assert system_instructions == ["be nice"]

        assert record.attributes["event.name"] == "gen_ai.evaluation.results"
        assert record.attributes["gen_ai.request.model"] == "gpt-test"
        assert record.attributes["gen_ai.provider.name"] == "openai"
    finally:
        # Clean up environment variable and reload module
        os.environ.pop("SPLUNK_EVALUATION_RESULTS_MESSAGE_CONTENT", None)
        from opentelemetry.util.genai.emitters import splunk as splunk_module

        importlib.reload(splunk_module)
