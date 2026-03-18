# Copyright The OpenTelemetry Authors
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import pytest

_PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if _PACKAGE_SRC.exists():
    sys.path.insert(0, str(_PACKAGE_SRC))

from opentelemetry.instrumentation.langchain.callback_handler import (  # noqa: E402
    LangchainCallbackHandler,
)
from opentelemetry.util.genai.types import Step, ToolCall, Workflow  # noqa: E402

try:  # pragma: no cover - optional dependency in CI
    from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402
except (
    ModuleNotFoundError
):  # pragma: no cover - allow running subset without langchain_core
    HumanMessage = None  # type: ignore[assignment]

LANGCHAIN_CORE_AVAILABLE = HumanMessage is not None


class _StubTelemetryHandler:
    def __init__(self) -> None:
        self.started_agents = []
        self.stopped_agents = []
        self.failed_agents = []
        self.started_llms = []
        self.stopped_llms = []
        self.started_tools = []
        self.stopped_tools = []
        self.failed_tools = []
        self.started_steps = []
        self.stopped_steps = []
        self.failed_steps = []
        self.started_workflows = []
        self.stopped_workflows = []

    def start_agent(self, agent):
        # Simulate store_span_context() assigning span_id after span creation
        if agent.span_id is None:
            agent.span_id = id(agent) & 0xFFFFFFFFFFFFFFFF
        self.started_agents.append(agent)
        return agent

    def stop_agent(self, agent):
        self.stopped_agents.append(agent)
        return agent

    def fail_agent(self, agent, error):
        self.failed_agents.append((agent, error))
        return agent

    def start_llm(self, invocation):
        self.started_llms.append(invocation)
        return invocation

    def stop_llm(self, invocation):
        self.stopped_llms.append(invocation)
        return invocation

    def evaluate_llm(self, invocation):  # pragma: no cover - simple stub
        return []

    def start_tool_call(self, call):
        self.started_tools.append(call)
        return call

    def stop_tool_call(self, call):
        self.stopped_tools.append(call)
        return call

    def fail_tool_call(self, call, error):
        self.failed_tools.append((call, error))
        return call

    def start_step(self, step):
        self.started_steps.append(step)
        return step

    def stop_step(self, step):
        self.stopped_steps.append(step)
        return step

    def fail_step(self, step, error):
        self.failed_steps.append((step, error))
        return step

    def start_workflow(self, workflow):
        self.started_workflows.append(workflow)
        return workflow

    def stop_workflow(self, workflow):
        self.stopped_workflows.append(workflow)
        return workflow

    def fail_workflow(self, workflow, error):
        return workflow


@pytest.fixture(name="handler_with_stub")
def _handler_with_stub_fixture() -> Tuple[
    LangchainCallbackHandler, _StubTelemetryHandler
]:
    stub = _StubTelemetryHandler()
    handler = LangchainCallbackHandler(telemetry_handler=stub)
    return handler, stub


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_agent_invocation_links_util_handler(handler_with_stub):
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={
            "name": "AgentExecutor",
            "id": ["langchain", "agents", "AgentExecutor"],
        },
        inputs={"messages": [HumanMessage(content="plan my trip")]},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"ls_agent_type": "react", "ls_model_name": "gpt-5-nano"},
    )

    assert stub.started_agents, "Agent start was not forwarded to util handler"
    agent = stub.started_agents[-1]
    assert agent.operation == "invoke_agent"
    assert agent.input_messages and len(agent.input_messages) > 0
    assert "plan my trip" in agent.input_messages[0].parts[0].content

    llm_run_id = uuid4()
    handler.on_chat_model_start(
        serialized={"name": "ChatOpenAI"},
        messages=[[HumanMessage(content="hello")]],
        run_id=llm_run_id,
        parent_run_id=agent_run_id,
        invocation_params={"model_name": "gpt-5-nano"},
        metadata={"ls_provider": "openai"},
    )

    assert stub.started_llms, "LLM invocation was not recorded"
    llm_invocation = stub.started_llms[-1]
    assert llm_invocation.agent_name == agent.name
    assert llm_invocation.agent_id is not None

    handler.on_chain_end(
        outputs={"messages": [AIMessage(content="done")]}, run_id=agent_run_id
    )

    assert stub.stopped_agents, "Agent stop was not forwarded to util handler"
    stopped_agent = stub.stopped_agents[-1]
    assert stopped_agent.output_messages and len(stopped_agent.output_messages) > 0
    assert "done" in stopped_agent.output_messages[0].parts[0].content


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_agent_failure_forwards_to_util(handler_with_stub):
    handler, stub = handler_with_stub

    failing_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={},
        run_id=failing_run_id,
        tags=["agent"],
    )

    error = RuntimeError("boom")
    handler.on_chain_error(error, run_id=failing_run_id)

    assert stub.failed_agents, "Agent failure was not propagated"
    failed_agent, recorded_error = stub.failed_agents[-1]
    assert recorded_error.message == str(error)
    assert recorded_error.type is RuntimeError


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_chain_metadata_maps_to_tool_call(handler_with_stub):
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "find weather"},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "weather_agent"},
    )

    tool_run_id = uuid4()
    tool_metadata = {
        "gen_ai.tool.name": "get_weather",
        "gen_ai.tool.arguments": {"city": "Berlin"},
        "gen_ai.step.type": "tool_invocation",
    }
    handler.on_chain_start(
        serialized={"name": "RunnableTool"},
        inputs={"city": "Berlin"},
        run_id=tool_run_id,
        parent_run_id=agent_run_id,
        metadata=tool_metadata,
    )

    assert stub.started_tools, "Tool metadata did not trigger ToolCall entity"
    tool = stub.started_tools[-1]
    assert isinstance(tool, ToolCall)
    assert tool.name == "get_weather"
    assert tool.arguments == {"city": "Berlin"}
    assert tool.agent_id is not None
    assert tool.attributes.get("gen_ai.tool.arguments") is None

    handler.on_chain_end(
        outputs={"temperature": 20}, run_id=tool_run_id, parent_run_id=agent_run_id
    )

    assert stub.stopped_tools and stub.stopped_tools[-1] is tool
    assert tool.attributes.get("tool.response") == '{"temperature": 20}'


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_tool_callbacks_use_tool_call(handler_with_stub):
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "weather_agent"},
    )

    tool_run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "weather_tool", "id": "tool-1"},
        input_str="ignored",
        run_id=tool_run_id,
        parent_run_id=agent_run_id,
        metadata={"model_name": "fake"},
        inputs={"city": "Madrid"},
    )

    assert stub.started_tools, "Tool callback did not create ToolCall"
    tool = stub.started_tools[-1]
    assert isinstance(tool, ToolCall)
    assert tool.name == "weather_tool"
    assert tool.id == "tool-1"
    assert tool.arguments == {"city": "Madrid"}
    assert tool.attributes.get("tool.arguments") == '{"city": "Madrid"}'

    handler.on_tool_end(
        output={"result": "sunny"}, run_id=tool_run_id, parent_run_id=agent_run_id
    )

    assert stub.stopped_tools and stub.stopped_tools[-1] is tool
    assert tool.attributes.get("tool.response") == '{"result": "sunny"}'


def test_chain_without_tool_creates_step(handler_with_stub):
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "plan my trip"},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "planner_agent"},
    )

    step_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "PlannerChain"},
        inputs={"question": "Where should I go?"},
        run_id=step_run_id,
        parent_run_id=agent_run_id,
        metadata={},
    )

    assert stub.started_steps, "Chain start without tool metadata should create a Step"
    step = stub.started_steps[-1]
    assert isinstance(step, Step)
    assert step.step_type == "chain"
    assert step.agent_id is not None
    assert step.agent_name == "planner_agent"


def test_step_outputs_recorded_on_chain_end(handler_with_stub):
    handler, stub = handler_with_stub

    parent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={},
        run_id=parent_run_id,
        tags=["agent"],
        metadata={"agent_name": "planner_agent"},
    )

    step_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "PlannerChain"},
        inputs={"question": "What should I pack?"},
        run_id=step_run_id,
        parent_run_id=parent_run_id,
        metadata={},
    )

    handler.on_chain_end(
        outputs={"answer": "Sunscreen"}, run_id=step_run_id, parent_run_id=parent_run_id
    )

    assert stub.stopped_steps, "Step end should be forwarded to util handler"


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_llm_attributes_independent_of_emitters(monkeypatch):
    def _build_handler() -> Tuple[LangchainCallbackHandler, _StubTelemetryHandler]:
        stub_handler = _StubTelemetryHandler()
        handler = LangchainCallbackHandler(telemetry_handler=stub_handler)
        return handler, stub_handler

    def _invoke_with_env(env_value: Optional[str]):
        if env_value is None:
            monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", raising=False)
        else:
            monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", env_value)

        handler, stub_handler = _build_handler()
        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI", "id": ["langchain", "ChatOpenAI"]},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            invocation_params={
                "model_name": "gpt-5-nano",
                "top_p": 0.5,
                "seed": 42,
                "model_kwargs": {"user": "abc"},
            },
            metadata={
                "ls_provider": "openai",
                "ls_max_tokens": 256,
                "custom_meta": "value",
            },
            tags=["agent"],
        )
        return stub_handler.started_llms[-1]

    invocation_default = _invoke_with_env(None)
    invocation_traceloop = _invoke_with_env("traceloop_compat")

    assert invocation_default.attributes == invocation_traceloop.attributes, (
        "Emitter env toggle should not change recorded attributes"
    )

    attrs = invocation_default.attributes
    assert invocation_default.request_model == "gpt-5-nano"
    assert invocation_default.provider == "openai"
    assert attrs["request_top_p"] == 0.5
    assert attrs["request_seed"] == 42
    assert attrs["request_max_tokens"] == 256
    assert attrs["custom_meta"] == "value"
    assert attrs["tags"] == ["agent"]
    assert attrs["callback.name"] == "ChatOpenAI"
    assert attrs["callback.id"] == ["langchain", "ChatOpenAI"]
    assert "traceloop.callback_name" not in attrs
    # ls_* fields are excluded from attributes - extracted to dedicated fields instead
    assert "ls_provider" not in attrs
    assert "ls_max_tokens" not in attrs
    assert "ls_model_name" not in attrs
    assert "langchain_legacy" not in attrs
    assert "model_kwargs" in attrs


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_workflow_output_fallback_when_no_ai_messages(handler_with_stub):
    """Root workflow span should capture state fields as output
    when no AIMessages are present (common LangGraph pattern)."""
    handler, stub = handler_with_stub

    wf_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "LangGraph"},
        inputs={"messages": [HumanMessage(content="Investigate alert")]},
        run_id=wf_run_id,
        # No agent tags -> creates Workflow, not AgentInvocation
    )

    assert stub.started_workflows, "Workflow was not started"
    wf = stub.started_workflows[-1]
    assert isinstance(wf, Workflow)
    assert wf.input_messages and len(wf.input_messages) == 1
    assert "Investigate alert" in wf.input_messages[0].parts[0].content

    # End with state that has no AI messages (only structured fields)
    handler.on_chain_end(
        outputs={
            "messages": [HumanMessage(content="Investigate alert")],
            "triage_result": {"service_id": "svc-123"},
            "confidence_score": 0.85,
        },
        run_id=wf_run_id,
    )

    assert stub.stopped_workflows, "Workflow was not stopped"
    stopped_wf = stub.stopped_workflows[-1]
    assert stopped_wf.output_messages and len(stopped_wf.output_messages) == 1
    output_text = stopped_wf.output_messages[0].parts[0].content
    assert "triage_result" in output_text
    assert "svc-123" in output_text
    assert "confidence_score" in output_text


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_workflow_output_captures_last_ai_message(handler_with_stub):
    """Root workflow span should capture only the last AIMessage,
    not all intermediate ones."""
    handler, stub = handler_with_stub

    wf_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "LangGraph"},
        inputs={"messages": [HumanMessage(content="help me")]},
        run_id=wf_run_id,
    )

    handler.on_chain_end(
        outputs={
            "messages": [
                HumanMessage(content="help me"),
                AIMessage(content="Let me call a tool"),
                AIMessage(content="Here is the final answer"),
            ],
        },
        run_id=wf_run_id,
    )

    assert stub.stopped_workflows
    stopped_wf = stub.stopped_workflows[-1]
    assert len(stopped_wf.output_messages) == 1
    assert "final answer" in stopped_wf.output_messages[0].parts[0].content


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_agent_output_captures_last_ai_message(handler_with_stub):
    """Agent span should capture only the last AIMessage output."""
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"messages": [HumanMessage(content="do something")]},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "test_agent"},
    )

    handler.on_chain_end(
        outputs={
            "messages": [
                HumanMessage(content="do something"),
                AIMessage(content="I will call a tool first"),
                AIMessage(content="Tool result processed"),
                AIMessage(content="Here is your answer"),
            ],
        },
        run_id=agent_run_id,
    )

    assert stub.stopped_agents
    stopped = stub.stopped_agents[-1]
    assert len(stopped.output_messages) == 1
    assert "Here is your answer" in stopped.output_messages[0].parts[0].content


# ---------------------------------------------------------------------------
# Interrupt / error classification tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_graph_interrupt_classifies_as_interrupt(handler_with_stub):
    """A GraphInterrupt exception should produce INTERRUPT classification, not REAL_ERROR."""
    handler, stub = handler_with_stub

    # Create a mock GraphInterrupt exception (avoid importing langgraph)
    class GraphInterrupt(Exception):
        pass

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"messages": [HumanMessage(content="plan my trip")]},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "planner", "thread_id": "t-123"},
    )

    handler.on_chain_error(GraphInterrupt("paused"), run_id=agent_run_id)

    assert stub.failed_agents, "GraphInterrupt should trigger fail_agent"
    _, error = stub.failed_agents[-1]
    from opentelemetry.util.genai.types import ErrorClassification

    assert error.classification == ErrorClassification.INTERRUPT
    assert error.type is GraphInterrupt


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_cancelled_error_classifies_as_cancellation(handler_with_stub):
    """CancelledError should produce CANCELLATION classification."""
    import asyncio

    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "planner"},
    )

    handler.on_chain_error(asyncio.CancelledError(), run_id=agent_run_id)

    assert stub.failed_agents
    _, error = stub.failed_agents[-1]
    from opentelemetry.util.genai.types import ErrorClassification

    assert error.classification == ErrorClassification.CANCELLATION


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_runtime_error_classifies_as_real_error(handler_with_stub):
    """RuntimeError should produce REAL_ERROR classification (regression)."""
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "planner"},
    )

    handler.on_chain_error(RuntimeError("boom"), run_id=agent_run_id)

    assert stub.failed_agents
    _, error = stub.failed_agents[-1]
    from opentelemetry.util.genai.types import ErrorClassification

    assert error.classification == ErrorClassification.REAL_ERROR


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
def test_thread_id_sets_conversation_id(handler_with_stub):
    """thread_id in metadata should flow to conversation_id on the entity."""
    handler, stub = handler_with_stub

    agent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"messages": [HumanMessage(content="hello")]},
        run_id=agent_run_id,
        tags=["agent"],
        metadata={"agent_name": "planner", "thread_id": "t-456"},
    )

    assert stub.started_agents
    agent = stub.started_agents[-1]
    assert agent.conversation_id == "t-456"

    handler.on_chain_end(
        outputs={"messages": [AIMessage(content="done")]},
        run_id=agent_run_id,
    )
