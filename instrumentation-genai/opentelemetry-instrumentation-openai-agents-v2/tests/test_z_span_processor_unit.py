# pylint: disable=wrong-import-position,wrong-import-order,import-error,no-name-in-module,unexpected-keyword-arg,no-value-for-parameter,redefined-outer-name,too-many-locals,too-many-statements,too-many-branches

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from agents.tracing import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    ResponseSpanData,
)

import opentelemetry.semconv._incubating.attributes.gen_ai_attributes as _gen_ai_attributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as _server_attributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import get_telemetry_handler


def _ensure_semconv_enums() -> None:
    if not hasattr(_gen_ai_attributes, "GenAiProviderNameValues"):

        class _GenAiProviderNameValues(Enum):
            OPENAI = "openai"
            GCP_GEN_AI = "gcp.gen_ai"
            GCP_VERTEX_AI = "gcp.vertex_ai"
            GCP_GEMINI = "gcp.gemini"
            ANTHROPIC = "anthropic"
            COHERE = "cohere"
            AZURE_AI_INFERENCE = "azure.ai.inference"
            AZURE_AI_OPENAI = "azure.ai.openai"
            IBM_WATSONX_AI = "ibm.watsonx.ai"
            AWS_BEDROCK = "aws.bedrock"
            PERPLEXITY = "perplexity"
            X_AI = "x.ai"
            DEEPSEEK = "deepseek"
            GROQ = "groq"
            MISTRAL_AI = "mistral.ai"

        class _GenAiOperationNameValues(Enum):
            CHAT = "chat"
            GENERATE_CONTENT = "generate_content"
            TEXT_COMPLETION = "text_completion"
            EMBEDDINGS = "embeddings"
            CREATE_AGENT = "create_agent"
            INVOKE_AGENT = "invoke_agent"
            EXECUTE_TOOL = "execute_tool"

        class _GenAiOutputTypeValues(Enum):
            TEXT = "text"
            JSON = "json"
            IMAGE = "image"
            SPEECH = "speech"

        _gen_ai_attributes.GenAiProviderNameValues = _GenAiProviderNameValues
        _gen_ai_attributes.GenAiOperationNameValues = _GenAiOperationNameValues
        _gen_ai_attributes.GenAiOutputTypeValues = _GenAiOutputTypeValues

    if not hasattr(_server_attributes, "SERVER_ADDRESS"):
        _server_attributes.SERVER_ADDRESS = "server.address"
    if not hasattr(_server_attributes, "SERVER_PORT"):
        _server_attributes.SERVER_PORT = "server.port"


_ensure_semconv_enums()

ServerAttributes = _server_attributes

sp = importlib.import_module(
    "opentelemetry.instrumentation.openai_agents.span_processor"
)

try:
    from opentelemetry.sdk.trace.export import (  # type: ignore[attr-defined]
        InMemorySpanExporter,
        SimpleSpanProcessor,
    )
except ImportError:  # pragma: no cover
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )


def _collect(iterator) -> dict[str, Any]:
    return dict(iterator)


@pytest.fixture
def processor_setup():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = get_telemetry_handler(tracer_provider=provider)
    processor = sp.GenAISemanticProcessor(
        handler=handler, system_name="openai"
    )
    yield processor, exporter
    processor.shutdown()
    exporter.clear()


def test_time_helpers():
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert sp._as_utc_nano(dt) == 1704067200 * 1_000_000_000

    class Fallback:
        def __str__(self) -> str:
            return "fallback"

    # Accept any JSON formatting as long as it round-trips correctly.
    assert json.loads(sp.safe_json_dumps({"foo": "bar"})) == {"foo": "bar"}
    assert sp.safe_json_dumps(Fallback()) == "fallback"


def test_infer_server_attributes_variants(monkeypatch):
    assert sp._infer_server_attributes(None) == {}
    assert sp._infer_server_attributes(123) == {}

    attrs = sp._infer_server_attributes("https://api.example.com:8080/v1")
    assert attrs[ServerAttributes.SERVER_ADDRESS] == "api.example.com"
    assert attrs[ServerAttributes.SERVER_PORT] == 8080

    def boom(_: str):
        raise ValueError("unparsable url")

    monkeypatch.setattr(sp, "urlparse", boom)
    assert sp._infer_server_attributes("bad") == {}


def test_operation_and_span_naming(processor_setup):
    processor, _ = processor_setup

    generation = GenerationSpanData(input=[{"role": "user"}], model="gpt-4o")
    assert (
        processor._get_operation_name(generation) == sp.GenAIOperationName.CHAT
    )

    completion = GenerationSpanData(input=[])
    assert (
        processor._get_operation_name(completion)
        == sp.GenAIOperationName.TEXT_COMPLETION
    )

    embeddings = GenerationSpanData(input=None)
    setattr(embeddings, "embedding_dimension", 128)
    assert (
        processor._get_operation_name(embeddings)
        == sp.GenAIOperationName.EMBEDDINGS
    )

    agent_create = AgentSpanData(operation=" CREATE ")
    assert (
        processor._get_operation_name(agent_create)
        == sp.GenAIOperationName.CREATE_AGENT
    )

    agent_invoke = AgentSpanData(operation="invoke_agent")
    assert (
        processor._get_operation_name(agent_invoke)
        == sp.GenAIOperationName.INVOKE_AGENT
    )

    agent_default = AgentSpanData(operation=None)
    assert (
        processor._get_operation_name(agent_default)
        == sp.GenAIOperationName.INVOKE_AGENT
    )

    function_data = FunctionSpanData()
    assert (
        processor._get_operation_name(function_data)
        == sp.GenAIOperationName.EXECUTE_TOOL
    )

    response_data = ResponseSpanData()
    assert (
        processor._get_operation_name(response_data)
        == sp.GenAIOperationName.CHAT
    )

    class UnknownSpanData:
        pass

    unknown = UnknownSpanData()
    assert processor._get_operation_name(unknown) == "unknown"

    assert (
        sp.get_span_name(sp.GenAIOperationName.CHAT, model="gpt-4o")
        == "chat gpt-4o"
    )
    assert (
        sp.get_span_name(
            sp.GenAIOperationName.EXECUTE_TOOL, tool_name="weather"
        )
        == "execute_tool weather"
    )
    assert (
        sp.get_span_name(sp.GenAIOperationName.INVOKE_AGENT, agent_name=None)
        == "invoke_agent"
    )
    assert (
        sp.get_span_name(sp.GenAIOperationName.CREATE_AGENT, agent_name=None)
        == "create_agent"
    )


def test_attribute_builders(processor_setup):
    processor, _ = processor_setup

    payload = sp.ContentPayload(
        input_messages=[
            {
                "role": "user",
                "parts": [{"type": "text", "content": "hi"}],
            }
        ],
        output_messages=[
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "hello"}],
            }
        ],
        system_instructions=[{"type": "text", "content": "be helpful"}],
    )
    model_config = {
        "base_url": "https://api.openai.com:443/v1",
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 3,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.4,
        "seed": 1234,
        "n": 2,
        "max_tokens": 128,
        "stop": ["foo", None, "bar"],
    }
    generation_span = GenerationSpanData(
        input=[{"role": "user"}],
        output=[{"finish_reason": "stop"}],
        model="gpt-4o",
        model_config=model_config,
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
        },
    )
    gen_attrs = _collect(
        processor._get_attributes_from_generation_span_data(
            generation_span, payload
        )
    )
    assert gen_attrs[sp.GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert gen_attrs[sp.GEN_AI_REQUEST_MAX_TOKENS] == 128
    assert gen_attrs[sp.GEN_AI_REQUEST_STOP_SEQUENCES] == [
        "foo",
        None,
        "bar",
    ]
    assert gen_attrs[ServerAttributes.SERVER_ADDRESS] == "api.openai.com"
    assert gen_attrs[ServerAttributes.SERVER_PORT] == 443
    assert gen_attrs[sp.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert gen_attrs[sp.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert gen_attrs[sp.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    assert json.loads(gen_attrs[sp.GEN_AI_INPUT_MESSAGES])[0]["role"] == "user"
    assert (
        json.loads(gen_attrs[sp.GEN_AI_OUTPUT_MESSAGES])[0]["role"]
        == "assistant"
    )
    assert (
        json.loads(gen_attrs[sp.GEN_AI_SYSTEM_INSTRUCTIONS])[0]["content"]
        == "be helpful"
    )
    assert gen_attrs[sp.GEN_AI_OUTPUT_TYPE] == sp.GenAIOutputType.TEXT

    class _Usage:
        def __init__(self) -> None:
            self.input_tokens = None
            self.prompt_tokens = 7
            self.output_tokens = None
            self.completion_tokens = 2
            self.total_tokens = 9

    class _Response:
        def __init__(self) -> None:
            self.id = "resp-1"
            self.model = "gpt-4o"
            self.usage = _Usage()
            self.output = [{"finish_reason": "stop"}]

    response_span = ResponseSpanData(response=_Response())
    response_attrs = _collect(
        processor._get_attributes_from_response_span_data(
            response_span, sp.ContentPayload()
        )
    )
    assert response_attrs[sp.GEN_AI_RESPONSE_ID] == "resp-1"
    assert response_attrs[sp.GEN_AI_RESPONSE_MODEL] == "gpt-4o"
    assert response_attrs[sp.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    assert response_attrs[sp.GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert response_attrs[sp.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    assert response_attrs[sp.GEN_AI_OUTPUT_TYPE] == sp.GenAIOutputType.TEXT

    agent_span = AgentSpanData(
        name="helper",
        output_type="json",
        description="desc",
        agent_id="agent-123",
        model="model-x",
        operation="invoke_agent",
    )
    agent_attrs = _collect(
        processor._get_attributes_from_agent_span_data(agent_span, None)
    )
    assert agent_attrs[sp.GEN_AI_AGENT_NAME] == "helper"
    assert agent_attrs[sp.GEN_AI_AGENT_ID] == "agent-123"
    assert agent_attrs[sp.GEN_AI_REQUEST_MODEL] == "model-x"
    assert agent_attrs[sp.GEN_AI_OUTPUT_TYPE] == sp.GenAIOutputType.TEXT

    # Fallback to aggregated model when span data lacks it
    agent_span_no_model = AgentSpanData(
        name="helper-2",
        output_type="json",
        description="desc",
        agent_id="agent-456",
        operation="invoke_agent",
    )
    agent_content = {
        "input_messages": [],
        "output_messages": [],
        "system_instructions": [],
        "request_model": "gpt-fallback",
    }
    agent_attrs_fallback = _collect(
        processor._get_attributes_from_agent_span_data(
            agent_span_no_model, agent_content
        )
    )
    assert agent_attrs_fallback[sp.GEN_AI_REQUEST_MODEL] == "gpt-fallback"

    function_span = FunctionSpanData(name="lookup_weather")
    function_span.tool_type = "extension"
    function_span.call_id = "call-42"
    function_span.description = "desc"
    function_payload = sp.ContentPayload(
        tool_arguments={"city": "seattle"},
        tool_result={"temperature": 70},
    )
    function_attrs = _collect(
        processor._get_attributes_from_function_span_data(
            function_span, function_payload
        )
    )
    assert function_attrs[sp.GEN_AI_TOOL_NAME] == "lookup_weather"
    assert function_attrs[sp.GEN_AI_TOOL_TYPE] == "extension"
    assert function_attrs[sp.GEN_AI_TOOL_CALL_ID] == "call-42"
    assert function_attrs[sp.GEN_AI_TOOL_DESCRIPTION] == "desc"
    assert function_attrs[sp.GEN_AI_TOOL_CALL_ARGUMENTS] == {"city": "seattle"}
    assert function_attrs[sp.GEN_AI_TOOL_CALL_RESULT] == {"temperature": 70}
    assert function_attrs[sp.GEN_AI_OUTPUT_TYPE] == sp.GenAIOutputType.JSON


def test_extract_genai_attributes_unknown_type(processor_setup):
    processor, _ = processor_setup

    class UnknownSpanData:
        pass

    class StubSpan:
        def __init__(self) -> None:
            self.span_data = UnknownSpanData()

    attrs = _collect(
        processor._extract_genai_attributes(
            StubSpan(), sp.ContentPayload(), None
        )
    )
    assert attrs[sp.GEN_AI_PROVIDER_NAME] == "openai"
    assert attrs[sp.GEN_AI_SYSTEM_KEY] == "openai"
    assert sp.GEN_AI_OPERATION_NAME not in attrs


def test_span_status_helper():
    status = sp._get_span_status(
        SimpleNamespace(error={"message": "boom", "data": "bad"})
    )
    assert status.status_code is StatusCode.ERROR
    assert status.description == "boom: bad"

    ok_status = sp._get_span_status(SimpleNamespace(error=None))
    assert ok_status.status_code is StatusCode.OK


@dataclass
class FakeTrace:
    name: str
    trace_id: str
    started_at: str | None = None
    ended_at: str | None = None


@dataclass
class FakeSpan:
    trace_id: str
    span_id: str
    span_data: Any
    parent_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    error: dict[str, Any] | None = None


@pytest.mark.skip(
    reason="Integration test - handler/emitter span export not working in unit test setup"
)
def test_span_lifecycle_and_shutdown(processor_setup):
    processor, exporter = processor_setup

    trace = FakeTrace(
        name="workflow",
        trace_id="trace-1",
        started_at="not-a-timestamp",
        ended_at="2024-01-01T00:00:05Z",
    )
    processor.on_trace_start(trace)

    parent_span = FakeSpan(
        trace_id="trace-1",
        span_id="span-1",
        span_data=AgentSpanData(
            operation="invoke", name="agent", model="gpt-4o"
        ),
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:02Z",
    )
    processor.on_span_start(parent_span)

    missing_span = FakeSpan(
        trace_id="trace-1",
        span_id="missing",
        span_data=FunctionSpanData(name="lookup"),
        started_at="2024-01-01T00:00:01Z",
        ended_at="2024-01-01T00:00:02Z",
    )
    processor.on_span_end(missing_span)

    child_span = FakeSpan(
        trace_id="trace-1",
        span_id="span-2",
        parent_id="span-1",
        span_data=FunctionSpanData(name="lookup"),
        started_at="2024-01-01T00:00:02Z",
        ended_at="2024-01-01T00:00:03Z",
        error={"message": "boom", "data": "bad"},
    )
    processor.on_span_start(child_span)
    processor.on_span_end(child_span)

    processor.on_span_end(parent_span)
    processor.on_trace_end(trace)

    linger_trace = FakeTrace(
        name="linger",
        trace_id="trace-2",
        started_at="2024-01-01T00:00:06Z",
    )
    processor.on_trace_start(linger_trace)
    linger_span = FakeSpan(
        trace_id="trace-2",
        span_id="span-3",
        span_data=AgentSpanData(operation=None),
        started_at="2024-01-01T00:00:06Z",
    )
    processor.on_span_start(linger_span)

    assert processor.force_flush() is None
    processor.shutdown()

    finished = exporter.get_finished_spans()
    statuses = {span.name: span.status for span in finished}

    assert (
        statuses["execute_tool lookup"].status_code is StatusCode.ERROR
        and statuses["execute_tool lookup"].description == "boom: bad"
    )
    assert statuses["invoke_agent agent"].status_code is StatusCode.OK
    assert (
        statuses["invoke_agent"].status_code is StatusCode.ERROR
        and statuses["invoke_agent"].description == "Application shutdown"
    )


@pytest.mark.skip(
    reason="Integration test - handler/emitter span export not working in unit test setup"
)
def test_chat_span_renamed_with_model(processor_setup):
    processor, exporter = processor_setup

    trace = FakeTrace(name="workflow", trace_id="trace-rename")
    processor.on_trace_start(trace)

    agent = FakeSpan(
        trace_id=trace.trace_id,
        span_id="agent-span",
        span_data=AgentSpanData(
            operation="invoke_agent",
            name="Agent",
        ),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:02Z",
    )
    processor.on_span_start(agent)

    generation_data = GenerationSpanData(
        input=[{"role": "user", "content": "question"}],
        output=[{"finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )
    generation_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="child-span",
        parent_id=agent.span_id,
        span_data=generation_data,
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:01Z",
    )
    processor.on_span_start(generation_span)

    # Model becomes available before span end (e.g., once response arrives)
    generation_data.model = "gpt-4o"

    processor.on_span_end(generation_span)
    processor.on_span_end(agent)
    processor.on_trace_end(trace)

    span_names = {span.name for span in exporter.get_finished_spans()}
    assert "chat gpt-4o" in span_names


def test_workflow_and_agent_spans_created(processor_setup):
    """Test that workflow wraps invoke_agent spans directly (no step wrapper)."""
    processor, exporter = processor_setup

    trace = FakeTrace(name="workflow", trace_id="trace-agents")
    processor.on_trace_start(trace)

    # Workflow entity should be created when the trace starts.
    # Name comes from trace.name if available, otherwise defaults to "OpenAIAgents"
    assert processor._workflow is not None
    assert getattr(processor._workflow, "name", None) == "workflow"

    agent_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="agent-span",
        span_data=AgentSpanData(
            operation="invoke_agent",
            name="Helper",
            model="gpt-4o",
        ),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:01Z",
    )

    processor.on_span_start(agent_span)

    # Agent spans should be direct children of workflow (no step wrapper)
    # The invoke_agent span is parented to the workflow span directly

    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)

    # Workflow is stopped when trace ends (using with trace() pattern)
    assert processor._workflow is None


def test_workflow_lifecycle_with_trace(processor_setup):
    """Test workflow lifecycle with trace() context manager pattern.

    When using `with trace()` from OpenAI Agents SDK, each trace creates
    and stops its own workflow. For multi-agent scenarios, all agents
    should be wrapped in a single `with trace()` block.
    """
    processor, _ = processor_setup

    # First trace creates and stops its workflow
    trace1 = FakeTrace(name="trace1", trace_id="trace-1")
    processor.on_trace_start(trace1)
    workflow_1 = processor._workflow
    assert workflow_1 is not None

    agent1_span = FakeSpan(
        trace_id=trace1.trace_id,
        span_id="agent-1",
        span_data=AgentSpanData(operation="invoke_agent", name="Agent1"),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:01Z",
    )
    processor.on_span_start(agent1_span)
    processor.on_span_end(agent1_span)
    processor.on_trace_end(trace1)

    # Workflow is stopped when trace ends
    assert processor._workflow is None

    # Second trace creates a new workflow
    trace2 = FakeTrace(name="trace2", trace_id="trace-2")
    processor.on_trace_start(trace2)

    # New workflow created
    workflow_2 = processor._workflow
    assert workflow_2 is not None

    agent2_span = FakeSpan(
        trace_id=trace2.trace_id,
        span_id="agent-2",
        span_data=AgentSpanData(operation="invoke_agent", name="Agent2"),
        started_at="2025-01-01T00:00:02Z",
        ended_at="2025-01-01T00:00:03Z",
    )
    processor.on_span_start(agent2_span)
    processor.on_span_end(agent2_span)
    processor.on_trace_end(trace2)

    # Workflow is stopped when trace ends
    assert processor._workflow is None


@pytest.mark.skip(
    reason="Integration test - handler/emitter span export not working in unit test setup"
)
def test_llm_and_tool_entities_lifecycle(processor_setup):
    """Test LLM and tool entity lifecycle - parented to workflow directly."""
    processor, exporter = processor_setup

    trace = FakeTrace(name="workflow", trace_id="trace-llm-tool")
    processor.on_trace_start(trace)

    agent_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="agent-span",
        span_data=AgentSpanData(operation="invoke_agent", name="Agent"),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:02Z",
    )
    processor.on_span_start(agent_span)

    # Generation (LLM) child span
    generation_data = GenerationSpanData(
        input=[{"role": "user", "content": "question"}],
        output=[{"finish_reason": "stop"}],
        model="gpt-4o",
    )
    generation_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="llm-span",
        parent_id=agent_span.span_id,
        span_data=generation_data,
        started_at="2025-01-01T00:00:01Z",
        ended_at="2025-01-01T00:00:02Z",
    )

    processor.on_span_start(generation_span)

    # LLMInvocation should be created immediately in on_span_start (unified tracking)
    llm_state = processor._invocations.get(str(generation_span.span_id))
    # LLM should be parented to agent (correct parent-child relationship)
    agent_state = processor._invocations.get(str(agent_span.span_id))
    if llm_state is not None and agent_state is not None:
        assert (
            getattr(llm_state.invocation, "parent_run_id", None)
            == agent_state.invocation.run_id
        )

    processor.on_span_end(generation_span)
    # After on_span_end, invocation should be cleaned up
    assert str(generation_span.span_id) not in processor._invocations

    # Function (tool) child span
    function_data = FunctionSpanData(name="lookup")
    function_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="tool-span",
        parent_id=agent_span.span_id,
        span_data=function_data,
        started_at="2025-01-01T00:00:02Z",
        ended_at="2025-01-01T00:00:03Z",
    )

    processor.on_span_start(function_span)

    # Tool should be tracked in unified _invocations dict
    tool_state = processor._invocations.get(str(function_span.span_id))
    assert tool_state is not None

    # Tool should be parented to agent (found via parent span lookup)
    if processor._workflow is not None:
        # Parent run_id should be set (either to agent or workflow)
        assert (
            getattr(tool_state.invocation, "parent_run_id", None) is not None
        )

    processor.on_span_end(function_span)
    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)

    # Internal maps should be cleaned up
    assert str(function_span.span_id) not in processor._invocations

    # Sanity check that spans were exported as usual.
    exported_names = {span.name for span in exporter.get_finished_spans()}
    assert "invoke_agent Agent" in exported_names
    assert (
        "chat gpt-4o" in exported_names
        or "text_completion gpt-4o" in exported_names
    )


def test_on_span_error_fails_invocation(processor_setup):
    """Test that on_span_error properly fails the invocation and cleans up state."""
    processor, _ = processor_setup

    trace = FakeTrace(name="error-workflow", trace_id="trace-error-1")
    processor.on_trace_start(trace)

    # Create an agent span
    agent_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="agent-error-span",
        span_data=AgentSpanData(operation="invoke_agent", name="ErrorAgent"),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:01Z",
    )
    processor.on_span_start(agent_span)

    # Verify agent is tracked
    assert str(agent_span.span_id) in processor._invocations

    # Simulate an error on the span
    test_error = RuntimeError("Test error occurred")
    processor.on_span_error(agent_span, test_error)

    # After on_span_error, invocation should be cleaned up
    assert str(agent_span.span_id) not in processor._invocations

    # Clean up
    processor.on_trace_end(trace)


def test_on_trace_error_fails_workflow(processor_setup):
    """Test that on_trace_error properly fails the workflow."""
    processor, _ = processor_setup

    trace = FakeTrace(name="error-workflow", trace_id="trace-error-2")
    processor.on_trace_start(trace)

    # Verify workflow is created
    assert processor._workflow is not None
    # Verify trace is tracked in invocations
    assert str(trace.trace_id) in processor._invocations

    # Simulate an error on the trace
    test_error = ValueError("Workflow error occurred")
    processor.on_trace_error(trace, test_error)

    # After on_trace_error, invocation state should be removed
    # (workflow instance vars are not cleaned in error state - cleaned on shutdown)
    assert str(trace.trace_id) not in processor._invocations


def test_on_span_error_with_llm_invocation(processor_setup):
    """Test that on_span_error properly fails LLM invocations."""
    processor, _ = processor_setup

    trace = FakeTrace(name="llm-error-workflow", trace_id="trace-llm-error")
    processor.on_trace_start(trace)

    # Create an agent span first (LLM spans need a parent agent)
    agent_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="parent-agent-span",
        span_data=AgentSpanData(operation="invoke_agent", name="ParentAgent"),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:05Z",
    )
    processor.on_span_start(agent_span)

    # Create an LLM span as child of agent
    llm_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="llm-error-span",
        parent_id=agent_span.span_id,
        span_data=GenerationSpanData(
            input=[{"role": "user", "content": "test"}],
            model="gpt-4o",
        ),
        started_at="2025-01-01T00:00:01Z",
        ended_at="2025-01-01T00:00:02Z",
    )
    processor.on_span_start(llm_span)

    # Verify LLM is tracked
    assert str(llm_span.span_id) in processor._invocations

    # Simulate an error on the LLM span
    test_error = ConnectionError("API connection failed")
    processor.on_span_error(llm_span, test_error)

    # After on_span_error, LLM invocation should be cleaned up
    assert str(llm_span.span_id) not in processor._invocations

    # Agent should still be tracked
    assert str(agent_span.span_id) in processor._invocations

    # Clean up
    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)


def test_on_span_error_with_tool_invocation(processor_setup):
    """Test that on_span_error properly fails tool invocations."""
    processor, _ = processor_setup

    trace = FakeTrace(name="tool-error-workflow", trace_id="trace-tool-error")
    processor.on_trace_start(trace)

    # Create an agent span first (tool spans need a parent agent)
    agent_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="tool-parent-agent",
        span_data=AgentSpanData(operation="invoke_agent", name="ToolAgent"),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:05Z",
    )
    processor.on_span_start(agent_span)

    # Create a tool span as child of agent
    tool_span = FakeSpan(
        trace_id=trace.trace_id,
        span_id="tool-error-span",
        parent_id=agent_span.span_id,
        span_data=FunctionSpanData(name="failing_tool"),
        started_at="2025-01-01T00:00:01Z",
        ended_at="2025-01-01T00:00:02Z",
    )
    processor.on_span_start(tool_span)

    # Verify tool is tracked
    assert str(tool_span.span_id) in processor._invocations

    # Simulate an error on the tool span
    test_error = TimeoutError("Tool execution timed out")
    processor.on_span_error(tool_span, test_error)

    # After on_span_error, tool invocation should be cleaned up
    assert str(tool_span.span_id) not in processor._invocations

    # Agent should still be tracked
    assert str(agent_span.span_id) in processor._invocations

    # Clean up
    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)


def test_on_span_error_nonexistent_span(processor_setup):
    """Test that on_span_error handles nonexistent spans gracefully."""
    processor, _ = processor_setup

    # Create a span that was never started (not in invocations)
    unknown_span = FakeSpan(
        trace_id="unknown-trace",
        span_id="unknown-span",
        span_data=AgentSpanData(operation="invoke_agent", name="Unknown"),
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:01Z",
    )

    # This should not raise an exception
    test_error = RuntimeError("Error on unknown span")
    processor.on_span_error(unknown_span, test_error)

    # Verify no state was created
    assert str(unknown_span.span_id) not in processor._invocations
