import json

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
)
from opentelemetry.util.genai.emitters.composite import CompositeEmitter
from opentelemetry.util.genai.emitters.content_events import (
    ContentEventsEmitter,
)
from opentelemetry.util.genai.emitters.span import SpanEmitter
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    EvaluationResult,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    Workflow,
)


class DummyLogger:
    def __init__(self):
        self.emitted = []

    def emit(self, record):
        self.emitted.append(record)


def _build_composite(logger: DummyLogger, capture_content: bool):
    provider = TracerProvider()
    span = SpanEmitter(
        tracer=provider.get_tracer(__name__), capture_content=False
    )  # span kept lean for event mode
    content = ContentEventsEmitter(
        logger=logger, capture_content=capture_content
    )
    return CompositeEmitter(
        span_emitters=[span],
        metrics_emitters=[],
        content_event_emitters=[content],
        evaluation_emitters=[],
    )


def test_events_without_content_capture(sample_invocation):
    logger = DummyLogger()
    gen = _build_composite(logger, capture_content=False)
    # Start and finish to emit events
    gen.on_start(sample_invocation)
    gen.on_end(sample_invocation)

    # No events should be emitted when capture_content=False
    assert len(logger.emitted) == 0


def test_events_with_content_capture(sample_invocation, monkeypatch):
    logger = DummyLogger()
    gen = _build_composite(logger, capture_content=True)
    gen.on_start(sample_invocation)
    gen.on_end(sample_invocation)

    # Single event should include both input and output payloads
    assert len(logger.emitted) == 1

    event = logger.emitted[0]
    body = event.body or {}
    inputs = body.get("gen_ai.input.messages") or []
    outputs = body.get("gen_ai.output.messages") or []

    assert inputs and inputs[0]["parts"][0]["content"] == "hello user"
    assert outputs and outputs[0]["parts"][0]["content"] == "hello back"
    assert sample_invocation.trace_id is not None
    assert sample_invocation.span_id is not None
    assert event.trace_id == sample_invocation.trace_id
    assert event.span_id == sample_invocation.span_id


class _RecordingEvaluationEmitter:
    role = "evaluation"

    def __init__(self) -> None:
        self.call_log: list[tuple[str, object]] = []

    def on_evaluation_results(self, results, obj=None):
        self.call_log.append(("results", list(results)))

    def on_end(self, obj):
        self.call_log.append(("end", obj))

    def on_error(self, error, obj):
        self.call_log.append(("error", error))


def test_evaluation_emitters_receive_lifecycle_callbacks():
    emitter = _RecordingEvaluationEmitter()
    composite = CompositeEmitter(
        span_emitters=[],
        metrics_emitters=[],
        content_event_emitters=[],
        evaluation_emitters=[emitter],
    )
    invocation = LLMInvocation(request_model="eval-model")
    result = EvaluationResult(metric_name="bias", score=0.1)

    composite.on_evaluation_results([result], invocation)
    composite.on_end(invocation)
    composite.on_error(RuntimeError("boom"), invocation)

    assert ("results", [result]) in emitter.call_log
    assert any(entry[0] == "end" for entry in emitter.call_log)
    assert any(entry[0] == "error" for entry in emitter.call_log)


@pytest.fixture
def sample_invocation():
    input_msg = InputMessage(role="user", parts=[Text(content="hello user")])
    output_msg = OutputMessage(
        role="assistant",
        parts=[Text(content="hello back")],
        finish_reason="stop",
    )
    inv = LLMInvocation(request_model="test-model")
    inv.input_messages = [input_msg]
    inv.output_messages = [output_msg]
    return inv


"""
Removed tests that depended on environment variable gating. Emission now controlled solely by capture_content flag.
"""


def test_span_emitter_filters_non_gen_ai_attributes():
    provider = TracerProvider()
    emitter = SpanEmitter(
        tracer=provider.get_tracer(__name__), capture_content=False
    )
    invocation = LLMInvocation(request_model="example-model")
    invocation.provider = "example-provider"
    invocation.framework = "langchain"
    invocation.agent_id = "agent-123"
    invocation.attributes.update(
        {
            "request_top_p": 0.42,
            "custom": "value",
            "gen_ai.request.id": "req-789",
            "ls_temperature": 0.55,
        }
    )

    emitter.on_start(invocation)
    invocation.response_model_name = "example-model-v2"
    invocation.response_id = "resp-456"
    invocation.input_tokens = 10
    invocation.output_tokens = 5
    invocation.attributes["gen_ai.response.finish_reasons"] = ["stop"]

    emitter.on_end(invocation)

    span = invocation.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    assert attrs.get("gen_ai.agent.id") == "agent-123"
    assert attrs.get("gen_ai.request.id") == "req-789"
    assert "request_top_p" not in attrs
    assert "custom" not in attrs
    assert "ls_temperature" not in attrs
    assert "traceloop.association.properties.ls_temperature" not in attrs
    assert all(not key.startswith("traceloop.") for key in attrs.keys())
    assert any(key.startswith("gen_ai.") for key in attrs)


def test_span_emitter_sets_server_attributes():
    provider = TracerProvider()
    emitter = SpanEmitter(
        tracer=provider.get_tracer(__name__), capture_content=False
    )
    invocation = LLMInvocation(
        request_model="example-model",
        server_address="api.service.local",
        server_port=3000,
    )

    emitter.on_start(invocation)
    emitter.on_end(invocation)

    span = invocation.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )
    assert attrs.get("server.address") == "api.service.local"
    assert attrs.get("server.port") == 3000


def test_embedding_error_closes_span():
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=False)
    embedding = EmbeddingInvocation(
        request_model="emb-model",
        input_texts=["hello"],
    )

    emitter.on_start(embedding)
    span = embedding.span
    assert span is not None

    emitter.on_error(Error(message="boom", type=RuntimeError), embedding)

    assert span.end_time is not None
    assert not span.is_recording()


def test_span_emitter_workflow_captures_content():
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=True)

    workflow = Workflow(
        name="trip_planner",
        workflow_type="sequential",
        input_messages=[
            InputMessage(
                role="user", parts=[Text(content="Plan a trip to Rome")]
            )
        ],
        output_messages=[
            OutputMessage(
                role="assistant",
                parts=[Text(content="Here is your itinerary")],
            )
        ],
    )

    emitter.on_start(workflow)
    emitter.on_end(workflow)

    span = workflow.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    operation_value = attrs.get(GenAI.GEN_AI_OPERATION_NAME)
    assert operation_value == "invoke_workflow"

    input_messages_raw = attrs.get(GEN_AI_INPUT_MESSAGES)
    assert input_messages_raw is not None
    input_messages = json.loads(input_messages_raw)
    assert input_messages[0]["parts"][0]["content"] == "Plan a trip to Rome"

    output_messages_raw = attrs.get(GEN_AI_OUTPUT_MESSAGES)
    assert output_messages_raw is not None
    output_messages = json.loads(output_messages_raw)
    assert (
        output_messages[0]["parts"][0]["content"] == "Here is your itinerary"
    )


def test_invoke_agent_span_emitter_for_sampled_attribute():
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer)

    agent_invocation = AgentInvocation(
        name="weather_agent",
    )

    emitter.on_start(agent_invocation)
    emitter.on_end(agent_invocation)

    span = agent_invocation.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    sampled_value = attrs.get("gen_ai.evaluation.sampled")
    assert sampled_value is True


def test_agent_invocation_with_structured_messages():
    """Test AgentInvocation with structured input_messages/output_messages."""
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=True)

    input_msg = InputMessage(
        role="user", parts=[Text(content="What's the weather?")]
    )
    output_msg = OutputMessage(
        role="assistant",
        parts=[Text(content="It's sunny today!")],
        finish_reason="stop",
    )
    agent = AgentInvocation(name="weather_agent")
    agent.input_messages = [input_msg]
    agent.output_messages = [output_msg]

    emitter.on_start(agent)
    emitter.on_end(agent)

    span = agent.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    # Verify structured messages are rendered correctly
    input_messages_raw = attrs.get(GEN_AI_INPUT_MESSAGES)
    assert input_messages_raw is not None
    input_messages = json.loads(input_messages_raw)
    assert input_messages[0]["parts"][0]["content"] == "What's the weather?"

    output_messages_raw = attrs.get(GEN_AI_OUTPUT_MESSAGES)
    assert output_messages_raw is not None
    output_messages = json.loads(output_messages_raw)
    assert output_messages[0]["parts"][0]["content"] == "It's sunny today!"


def test_agent_invocation_prefers_structured_over_legacy():
    """Test that structured messages are preferred over legacy string fields."""
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=True)

    input_msg = InputMessage(
        role="user", parts=[Text(content="Structured input")]
    )
    output_msg = OutputMessage(
        role="assistant",
        parts=[Text(content="Structured output")],
        finish_reason="stop",
    )
    agent = AgentInvocation(name="test_agent")
    # Set both structured and legacy fields
    agent.input_messages = [input_msg]
    agent.output_messages = [output_msg]
    agent.input_context = "Legacy input (should be ignored)"
    agent.output_result = "Legacy output (should be ignored)"

    emitter.on_start(agent)
    emitter.on_end(agent)

    span = agent.span
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    # Verify structured messages take precedence
    input_messages = json.loads(attrs.get(GEN_AI_INPUT_MESSAGES))
    assert input_messages[0]["parts"][0]["content"] == "Structured input"

    output_messages = json.loads(attrs.get(GEN_AI_OUTPUT_MESSAGES))
    assert output_messages[0]["parts"][0]["content"] == "Structured output"


def test_workflow_with_structured_messages():
    """Test Workflow with structured input_messages/output_messages."""
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=True)

    input_msg = InputMessage(
        role="user", parts=[Text(content="Plan a trip to Paris")]
    )
    output_msg = OutputMessage(
        role="assistant",
        parts=[Text(content="Here is your Paris itinerary")],
        finish_reason="stop",
    )
    workflow = Workflow(name="travel_planner")
    workflow.input_messages = [input_msg]
    workflow.output_messages = [output_msg]

    emitter.on_start(workflow)
    emitter.on_end(workflow)

    span = workflow.span
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    input_messages = json.loads(attrs.get(GEN_AI_INPUT_MESSAGES))
    assert input_messages[0]["parts"][0]["content"] == "Plan a trip to Paris"

    output_messages = json.loads(attrs.get(GEN_AI_OUTPUT_MESSAGES))
    assert (
        output_messages[0]["parts"][0]["content"]
        == "Here is your Paris itinerary"
    )


def test_llm_span_emitter_for_sampled_attribute():
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer)

    llm_invocation = LLMInvocation(request_model="test-model")

    emitter.on_start(llm_invocation)
    emitter.on_end(llm_invocation)

    span = llm_invocation.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )

    sampled_value = attrs.get("gen_ai.evaluation.sampled")
    assert sampled_value is True
