import json
from unittest.mock import MagicMock

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
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    EvaluationResult,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Step,
    Text,
    ToolCall,
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


# ---- TelemetryHandler._maybe_mark_conversation_root tests ----------------


def test_handler_marks_workflow_root_when_no_parent_span():
    """Handler auto-sets conversation_root=True on Workflow with no parent_span."""
    workflow = Workflow(name="root_wf")
    TelemetryHandler._maybe_mark_conversation_root(workflow)
    assert workflow.conversation_root is True


def test_handler_marks_agent_root_when_no_parent_span():
    """Handler auto-sets conversation_root=True on AgentInvocation with no parent_span."""
    agent = AgentInvocation(name="root_agent")
    TelemetryHandler._maybe_mark_conversation_root(agent)
    assert agent.conversation_root is True


def test_handler_skips_root_when_parent_span_exists():
    """Handler does NOT set conversation_root when parent_span is present."""
    workflow = Workflow(name="child_wf")
    workflow.parent_span = MagicMock()  # simulate a parent span
    TelemetryHandler._maybe_mark_conversation_root(workflow)
    assert workflow.conversation_root is None


def test_handler_respects_explicit_conversation_root_false():
    """Handler does NOT override conversation_root when explicitly set to False."""
    workflow = Workflow(name="wf", conversation_root=False)
    TelemetryHandler._maybe_mark_conversation_root(workflow)
    assert workflow.conversation_root is False


def test_handler_respects_explicit_conversation_root_true_with_parent():
    """Handler preserves conversation_root=True even if parent_span exists."""
    agent = AgentInvocation(name="forced_root", conversation_root=True)
    agent.parent_span = MagicMock()
    TelemetryHandler._maybe_mark_conversation_root(agent)
    assert agent.conversation_root is True


def test_handler_idempotent_when_already_true():
    """Calling _maybe_mark_conversation_root twice does not change result."""
    workflow = Workflow(name="wf")
    TelemetryHandler._maybe_mark_conversation_root(workflow)
    assert workflow.conversation_root is True
    TelemetryHandler._maybe_mark_conversation_root(workflow)
    assert workflow.conversation_root is True


def test_handler_skips_llm_invocation():
    """LLMInvocation is never marked as conversation root."""
    llm = LLMInvocation(request_model="test-model")
    TelemetryHandler._maybe_mark_conversation_root(llm)
    assert llm.conversation_root is None


def test_handler_skips_step():
    """Step entity is never marked as conversation root."""
    step = Step(name="my_step")
    TelemetryHandler._maybe_mark_conversation_root(step)
    assert step.conversation_root is None


def test_handler_skips_tool_call():
    """ToolCall entity is never marked as conversation root."""
    tool = ToolCall(name="my_tool")
    TelemetryHandler._maybe_mark_conversation_root(tool)
    assert tool.conversation_root is None


def test_handler_skips_embedding_invocation():
    """EmbeddingInvocation entity is never marked as conversation root."""
    emb = EmbeddingInvocation(request_model="embed-model")
    TelemetryHandler._maybe_mark_conversation_root(emb)
    assert emb.conversation_root is None


def test_handler_skips_agent_creation():
    """AgentCreation is never marked as conversation root (only invocations)."""
    creation = AgentCreation(name="new_agent")
    TelemetryHandler._maybe_mark_conversation_root(creation)
    assert creation.conversation_root is None


def test_workflow_conversation_root_attribute_on_span():
    """Workflow with conversation_root=True emits gen_ai.conversation_root on span."""
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=False)

    workflow = Workflow(name="root_workflow", conversation_root=True)
    emitter.on_start(workflow)
    emitter.on_end(workflow)

    span = workflow.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )
    assert attrs.get("gen_ai.conversation_root") is True


def test_workflow_no_conversation_root_attribute_when_none():
    """Workflow without conversation_root does not emit the attribute on span."""
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=False)

    workflow = Workflow(name="child_workflow")
    emitter.on_start(workflow)
    emitter.on_end(workflow)

    span = workflow.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )
    assert "gen_ai.conversation_root" not in attrs


def test_llm_invocation_no_conversation_root():
    """LLMInvocation does not emit conversation_root (only Workflow/Agent roots do)."""
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=False)

    llm = LLMInvocation(request_model="test-model")
    emitter.on_start(llm)
    emitter.on_end(llm)

    span = llm.span
    assert span is not None
    attrs = getattr(span, "attributes", None) or getattr(
        span, "_attributes", {}
    )
    assert "gen_ai.conversation_root" not in attrs


# ---- Non-GenAI parent span scenarios ------------------------------------
#
# These tests verify conversation_root detection when GenAI spans live under
# a non-GenAI span (e.g. HTTP server span).  The heuristic checks entity-level
# parent_span (GenAI parent), not the OTel trace parent.


def _make_handler_with_exporter():
    """Create a TelemetryHandler + span exporter for integration tests."""
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    class _Collector(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    exporter = _Collector()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    TelemetryHandler._reset_for_testing()
    handler = TelemetryHandler(tracer_provider=tp)
    return handler, exporter, tp


def test_genai_root_under_http_span_gets_conversation_root():
    """A GenAI agent under a non-GenAI HTTP span should still get
    conversation_root=True because entity.parent_span is None."""
    handler, exporter, tp = _make_handler_with_exporter()
    tracer = tp.get_tracer(__name__)

    # Simulate an HTTP server span as the trace root
    with tracer.start_as_current_span("GET /chat", kind=None):
        # GenAI agent created while the HTTP span is current context.
        # No GenAI parent_span set — the handler should mark it as root.
        agent = AgentInvocation(name="chat_agent")
        agent.input_messages = [
            InputMessage(role="user", parts=[Text("hello")])
        ]
        handler.start_agent(agent)
        handler.stop_agent(agent)

    tp.force_flush()

    # Find spans by name
    span_map = {s.name: s for s in exporter.spans}
    assert "invoke_agent chat_agent" in span_map
    assert "GET /chat" in span_map

    agent_span = span_map["invoke_agent chat_agent"]
    http_span_exported = span_map["GET /chat"]

    # Agent span is a child of the HTTP span (OTel context propagation)
    assert agent_span.parent is not None
    assert agent_span.parent.span_id == http_span_exported.context.span_id

    # Agent span still marked as conversation root
    attrs = dict(agent_span.attributes or {})
    assert attrs.get("gen_ai.conversation_root") is True


def test_genai_workflow_under_http_span_gets_conversation_root():
    """Same as above but with a Workflow entity."""
    handler, exporter, tp = _make_handler_with_exporter()
    tracer = tp.get_tracer(__name__)

    with tracer.start_as_current_span("POST /run"):
        wf = Workflow(name="my_pipeline")
        wf.input_messages = [InputMessage(role="user", parts=[Text("run it")])]
        handler.start_workflow(wf)
        handler.stop_workflow(wf)

    tp.force_flush()

    span_map = {s.name: s for s in exporter.spans}
    wf_span = span_map["workflow my_pipeline"]
    http_exported = span_map["POST /run"]

    # Workflow is child of HTTP span
    assert wf_span.parent is not None
    assert wf_span.parent.span_id == http_exported.context.span_id

    # Workflow still marked as conversation root
    attrs = dict(wf_span.attributes or {})
    assert attrs.get("gen_ai.conversation_root") is True


def test_two_parallel_agents_under_http_span_both_get_conversation_root():
    """Two independent GenAI agents as siblings under an HTTP span.
    Both have parent_span=None, so both get conversation_root=True.

    This is the CURRENT behavior — not necessarily desired.  See
    findings documentation for discussion.
    """
    handler, exporter, tp = _make_handler_with_exporter()
    tracer = tp.get_tracer(__name__)

    with tracer.start_as_current_span("POST /multi-agent"):
        agent_a = AgentInvocation(name="agent_a")
        agent_a.input_messages = [
            InputMessage(role="user", parts=[Text("task A")])
        ]
        handler.start_agent(agent_a)
        handler.stop_agent(agent_a)

        agent_b = AgentInvocation(name="agent_b")
        agent_b.input_messages = [
            InputMessage(role="user", parts=[Text("task B")])
        ]
        handler.start_agent(agent_b)
        handler.stop_agent(agent_b)

    tp.force_flush()

    agent_spans = [s for s in exporter.spans if "invoke_agent" in s.name]
    assert len(agent_spans) == 2

    # BOTH agents get conversation_root=True (current behavior)
    for s in agent_spans:
        attrs = dict(s.attributes or {})
        assert attrs.get("gen_ai.conversation_root") is True, (
            f"Expected conversation_root=True on {s.name}"
        )


def test_http_genai_nongenai_genai_mixed_hierarchy():
    """HTTP → GenAI agent → non-GenAI span → nested GenAI agent.

    The first GenAI agent has no GenAI parent_span → conversation_root=True.
    The second GenAI agent, if given the first as parent_span, should NOT
    be marked as root.
    """
    handler, exporter, tp = _make_handler_with_exporter()
    tracer = tp.get_tracer(__name__)

    with tracer.start_as_current_span("GET /pipeline"):
        # First GenAI agent (root)
        outer_agent = AgentInvocation(name="orchestrator")
        outer_agent.input_messages = [
            InputMessage(role="user", parts=[Text("go")])
        ]
        handler.start_agent(outer_agent)

        # Non-GenAI middleware span (e.g. a database call or HTTP request)
        with tracer.start_as_current_span("db.query"):
            # Second GenAI agent — instrumentation sets parent_span
            inner_agent = AgentInvocation(name="inner_agent")
            inner_agent.parent_span = outer_agent.span
            inner_agent.input_messages = [
                InputMessage(role="user", parts=[Text("sub-task")])
            ]
            handler.start_agent(inner_agent)
            handler.stop_agent(inner_agent)

        handler.stop_agent(outer_agent)

    tp.force_flush()

    span_map = {s.name: s for s in exporter.spans}
    outer_span = span_map["invoke_agent orchestrator"]
    inner_span = span_map["invoke_agent inner_agent"]

    # Outer agent IS conversation root
    assert (
        dict(outer_span.attributes or {}).get("gen_ai.conversation_root")
        is True
    )

    # Inner agent is NOT conversation root (has parent_span)
    assert "gen_ai.conversation_root" not in dict(inner_span.attributes or {})


def test_mixed_hierarchy_without_parent_span_both_roots():
    """HTTP → GenAI → non-GenAI → GenAI, but the inner GenAI does NOT
    have parent_span explicitly set.

    With _inherit_parent_span, the inner agent auto-inherits the outer
    agent's span via _current_genai_span ContextVar, so only the outer
    agent is marked conversation_root=True.
    """
    handler, exporter, tp = _make_handler_with_exporter()
    tracer = tp.get_tracer(__name__)

    with tracer.start_as_current_span("GET /pipeline"):
        outer_agent = AgentInvocation(name="outer")
        outer_agent.input_messages = [
            InputMessage(role="user", parts=[Text("go")])
        ]
        handler.start_agent(outer_agent)

        with tracer.start_as_current_span("middleware.call"):
            inner_agent = AgentInvocation(name="inner")
            inner_agent.input_messages = [
                InputMessage(role="user", parts=[Text("sub")])
            ]
            handler.start_agent(inner_agent)
            handler.stop_agent(inner_agent)

        handler.stop_agent(outer_agent)

    tp.force_flush()

    agent_spans = [s for s in exporter.spans if "invoke_agent" in s.name]
    assert len(agent_spans) == 2

    outer_attrs = dict(
        next(s for s in agent_spans if "outer" in s.name).attributes or {}
    )
    inner_attrs = dict(
        next(s for s in agent_spans if "inner" in s.name).attributes or {}
    )
    assert outer_attrs.get("gen_ai.conversation_root") is True
    assert inner_attrs.get("gen_ai.conversation_root") is None


# ---- create_and_start_root factory method tests ----------------------------


def test_create_and_start_root_creates_agent_by_default():
    """Default (no env var) should create AgentInvocation."""
    import os

    # Ensure env var is not set
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW", None)

    handler, exporter, tp = _make_handler_with_exporter()

    entity = handler.create_and_start_root(
        "my_agent",
        framework="test",
        system="pytest",
        input_messages=[InputMessage(role="user", parts=[Text("hello")])],
    )

    assert isinstance(entity, AgentInvocation)
    assert entity.name == "my_agent"
    assert entity.agent_name == "my_agent"
    assert entity.framework == "test"
    assert entity.system == "pytest"
    assert entity.input_messages is not None
    assert len(entity.input_messages) == 1

    handler.finish(entity)
    tp.force_flush()

    # Should have created an agent span
    spans = exporter.spans
    assert len(spans) == 1
    assert "invoke_agent my_agent" in spans[0].name


def test_create_and_start_root_creates_workflow_with_env_var(monkeypatch):
    """When env var is set, should create Workflow."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW", "true"
    )

    handler, exporter, tp = _make_handler_with_exporter()

    entity = handler.create_and_start_root(
        "my_workflow",
        workflow_type="test.workflow",
        framework="test",
        system="pytest",
    )

    assert isinstance(entity, Workflow)
    assert entity.name == "my_workflow"
    assert entity.workflow_type == "test.workflow"

    handler.finish(entity)
    tp.force_flush()

    spans = exporter.spans
    assert len(spans) == 1
    assert "workflow my_workflow" in spans[0].name


def test_create_and_start_root_force_workflow_overrides_default():
    """force_workflow=True should create Workflow even without env var."""
    import os

    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW", None)

    handler, exporter, tp = _make_handler_with_exporter()

    entity = handler.create_and_start_root(
        "my_agent",
        force_workflow=True,
        framework="test",
    )

    assert isinstance(entity, Workflow)
    assert entity.name == "my_agent"

    handler.finish(entity)
    tp.force_flush()

    spans = exporter.spans
    assert "workflow my_agent" in spans[0].name


def test_create_and_start_root_workflow_name_becomes_name():
    """workflow_name='Custom Name' should use that as workflow name."""
    import os

    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW", None)

    handler, exporter, tp = _make_handler_with_exporter()

    entity = handler.create_and_start_root(
        "default_name",
        workflow_name="Custom Workflow Name",
    )

    assert isinstance(entity, Workflow)
    assert entity.name == "Custom Workflow Name"

    handler.finish(entity)
    tp.force_flush()

    spans = exporter.spans
    assert "workflow Custom Workflow Name" in spans[0].name


def test_finish_dispatches_correctly():
    """finish() should call stop_workflow or stop_agent based on type."""
    import os

    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW", None)

    handler, exporter, tp = _make_handler_with_exporter()

    # Test with AgentInvocation
    agent = handler.create_and_start_root("test_agent")
    result = handler.finish(agent)
    assert result is agent
    assert isinstance(result, AgentInvocation)

    # Test with Workflow
    workflow = handler.create_and_start_root("test_wf", force_workflow=True)
    result = handler.finish(workflow)
    assert result is workflow
    assert isinstance(result, Workflow)

    tp.force_flush()

    spans = exporter.spans
    assert len(spans) == 2


def test_fail_dispatches_correctly():
    """fail() should fail the entity and end its span."""
    import os

    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW", None)

    handler, exporter, tp = _make_handler_with_exporter()

    # Create and fail an agent
    agent = handler.create_and_start_root("failing_agent")
    error = Error(message="test error", type=ValueError)
    result = handler.fail(agent, error)
    assert result is agent

    tp.force_flush()

    spans = exporter.spans
    assert len(spans) == 1
    # The span should exist and be ended (error path)
