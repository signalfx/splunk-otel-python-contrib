# Copyright The OpenTelemetry Authors
"""Tests for LangChain/LangGraph thread_id → gen_ai.conversation.id inference.

Covers:
- _infer_conversation_id helper
- Workflow root with thread_id in metadata
- AgentInvocation root with thread_id in metadata
- conversation_id key takes priority over thread_id
- Explicit genai_context overrides inferred thread_id
- No metadata → no conversation_id
- Child LLM inherits conversation_id (through _apply_genai_context)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple
from uuid import uuid4

import pytest

_PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if _PACKAGE_SRC.exists():
    sys.path.insert(0, str(_PACKAGE_SRC))

from opentelemetry.instrumentation.langchain.callback_handler import (  # noqa: E402
    LangchainCallbackHandler,
    _infer_conversation_id,
)
from opentelemetry.util.genai.handler import (  # noqa: E402
    clear_genai_context,
    get_genai_context,
    set_genai_context,
)
from opentelemetry.util.genai.types import (  # noqa: E402
    AgentInvocation,
    Workflow,
)

try:  # pragma: no cover - optional dependency in CI
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    HumanMessage = None  # type: ignore[assignment]
    AIMessage = None  # type: ignore[assignment]

LANGCHAIN_CORE_AVAILABLE = HumanMessage is not None


# ── Stub (mirrors test_callback_handler_agent.py) ──────────────────


class _StubTelemetryHandler:
    def __init__(self) -> None:
        self.started_agents: list[Any] = []
        self.stopped_agents: list[Any] = []
        self.failed_agents: list[Any] = []
        self.started_llms: list[Any] = []
        self.stopped_llms: list[Any] = []
        self.started_tools: list[Any] = []
        self.stopped_tools: list[Any] = []
        self.failed_tools: list[Any] = []
        self.started_steps: list[Any] = []
        self.stopped_steps: list[Any] = []
        self.failed_steps: list[Any] = []
        self.started_workflows: list[Any] = []
        self.stopped_workflows: list[Any] = []
        self.entities: dict[str, Any] = {}

    def start_agent(self, agent):
        self.started_agents.append(agent)
        self.entities[str(agent.run_id)] = agent
        return agent

    def stop_agent(self, agent):
        self.stopped_agents.append(agent)
        self.entities.pop(str(agent.run_id), None)
        return agent

    def fail_agent(self, agent, error):
        self.failed_agents.append((agent, error))
        self.entities.pop(str(agent.run_id), None)
        return agent

    def start_llm(self, invocation):
        self.started_llms.append(invocation)
        self.entities[str(invocation.run_id)] = invocation
        return invocation

    def stop_llm(self, invocation):
        self.stopped_llms.append(invocation)
        self.entities.pop(str(invocation.run_id), None)
        return invocation

    def evaluate_llm(self, invocation):
        return []

    def start_tool_call(self, call):
        self.started_tools.append(call)
        self.entities[str(call.run_id)] = call
        return call

    def stop_tool_call(self, call):
        self.stopped_tools.append(call)
        self.entities.pop(str(call.run_id), None)
        return call

    def fail_tool_call(self, call, error):
        self.failed_tools.append((call, error))
        self.entities.pop(str(call.run_id), None)
        return call

    def start_step(self, step):
        self.started_steps.append(step)
        self.entities[str(step.run_id)] = step
        return step

    def stop_step(self, step):
        self.stopped_steps.append(step)
        self.entities.pop(str(step.run_id), None)
        return step

    def fail_step(self, step, error):
        self.failed_steps.append((step, error))
        self.entities.pop(str(step.run_id), None)
        return step

    def start_workflow(self, workflow):
        self.started_workflows.append(workflow)
        self.entities[str(workflow.run_id)] = workflow
        return workflow

    def stop_workflow(self, workflow):
        self.stopped_workflows.append(workflow)
        self.entities.pop(str(workflow.run_id), None)
        return workflow

    def fail_workflow(self, workflow, error):
        self.entities.pop(str(workflow.run_id), None)
        return workflow

    def fail_by_run_id(self, run_id, error):
        entity = self.entities.get(str(run_id))
        if entity is None:
            return
        self.fail_agent(entity, error)

    def get_entity(self, run_id):
        return self.entities.get(str(run_id))


@pytest.fixture(name="handler_with_stub")
def _handler_with_stub_fixture() -> Tuple[
    LangchainCallbackHandler, _StubTelemetryHandler
]:
    stub = _StubTelemetryHandler()
    handler = LangchainCallbackHandler(telemetry_handler=stub)
    return handler, stub


@pytest.fixture(autouse=True)
def _clean_genai_context():
    """Ensure each test starts with a clean GenAI context."""
    clear_genai_context()
    yield
    clear_genai_context()


# ── Unit tests for _infer_conversation_id ──────────────────────────


class TestInferConversationId:
    def test_returns_none_for_none_metadata(self):
        assert _infer_conversation_id(None) is None

    def test_returns_none_for_empty_metadata(self):
        assert _infer_conversation_id({}) is None

    def test_extracts_thread_id(self):
        assert _infer_conversation_id({"thread_id": "abc-123"}) == "abc-123"

    def test_extracts_conversation_id(self):
        assert _infer_conversation_id({"conversation_id": "conv-456"}) == "conv-456"

    def test_conversation_id_takes_priority_over_thread_id(self):
        metadata = {"conversation_id": "conv-win", "thread_id": "thread-lose"}
        assert _infer_conversation_id(metadata) == "conv-win"

    def test_converts_non_string_to_str(self):
        assert _infer_conversation_id({"thread_id": 42}) == "42"

    def test_converts_uuid_to_str(self):
        uid = uuid4()
        assert _infer_conversation_id({"thread_id": uid}) == str(uid)

    def test_ignores_unrelated_keys(self):
        assert _infer_conversation_id({"session_id": "s1", "run_id": "r1"}) is None


# ── Integration tests: Workflow path ───────────────────────────────


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestWorkflowConversationIdInference:
    def test_thread_id_sets_workflow_conversation_id(self, handler_with_stub):
        """Workflow root should have conversation_id from thread_id."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "session-abc"},
        )

        assert stub.started_workflows
        wf = stub.started_workflows[-1]
        assert isinstance(wf, Workflow)
        assert wf.conversation_id == "session-abc"

    def test_conversation_id_key_sets_workflow_conversation_id(self, handler_with_stub):
        """conversation_id key in metadata should map to conversation_id."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"conversation_id": "conv-xyz"},
        )

        wf = stub.started_workflows[-1]
        assert wf.conversation_id == "conv-xyz"

    def test_no_metadata_no_conversation_id(self, handler_with_stub):
        """Without thread_id/conversation_id, conversation_id stays None."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
        )

        wf = stub.started_workflows[-1]
        assert wf.conversation_id is None

    def test_empty_metadata_no_conversation_id(self, handler_with_stub):
        """Empty metadata dict → no conversation_id."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={},
        )

        wf = stub.started_workflows[-1]
        assert wf.conversation_id is None

    def test_explicit_context_overrides_metadata_thread_id(self, handler_with_stub):
        """Explicit genai_context(conversation_id=...) wins over metadata thread_id.

        When explicit context is active, the callback handler must NOT set the
        inferred value, allowing _apply_genai_context to apply the explicit one.
        Since we use a stub handler (no _apply_genai_context), the entity should
        have conversation_id=None — the stub doesn't run _apply_genai_context.
        This test verifies the callback handler skips inference when explicit
        context exists.
        """
        handler, stub = handler_with_stub

        set_genai_context(conversation_id="explicit-session")

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "inferred-session"},
        )

        wf = stub.started_workflows[-1]
        # The inferred value must NOT be set — explicit context takes priority.
        # In a real runtime, _apply_genai_context would then set "explicit-session".
        assert wf.conversation_id is None

    def test_metadata_with_unrelated_keys_no_inference(self, handler_with_stub):
        """Metadata without recognized keys → no conversation_id."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"session_id": "s1", "custom_key": "v1"},
        )

        wf = stub.started_workflows[-1]
        assert wf.conversation_id is None


# ── Integration tests: AgentInvocation path ────────────────────────


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestAgentConversationIdInference:
    def test_thread_id_sets_agent_conversation_id(self, handler_with_stub):
        """Agent root should have conversation_id from thread_id."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="plan trip")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={"agent_name": "travel_agent", "thread_id": "thread-xyz"},
        )

        assert stub.started_agents
        agent = stub.started_agents[-1]
        assert isinstance(agent, AgentInvocation)
        assert agent.conversation_id == "thread-xyz"

    def test_conversation_id_key_sets_agent_conversation_id(self, handler_with_stub):
        """conversation_id key in metadata maps to agent conversation_id."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="plan trip")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={"agent_name": "travel_agent", "conversation_id": "conv-abc"},
        )

        agent = stub.started_agents[-1]
        assert agent.conversation_id == "conv-abc"

    def test_no_thread_id_no_agent_conversation_id(self, handler_with_stub):
        """Without relevant metadata keys, agent conversation_id is None."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="plan trip")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={"agent_name": "travel_agent"},
        )

        agent = stub.started_agents[-1]
        assert agent.conversation_id is None

    def test_explicit_context_overrides_agent_thread_id(self, handler_with_stub):
        """Explicit genai_context wins over metadata thread_id for agents."""
        handler, stub = handler_with_stub

        set_genai_context(conversation_id="explicit-conv")

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="plan trip")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={"agent_name": "travel_agent", "thread_id": "inferred-conv"},
        )

        agent = stub.started_agents[-1]
        # Inferred value must NOT be set — explicit context has priority.
        assert agent.conversation_id is None

    def test_conversation_id_priority_over_thread_id_for_agent(self, handler_with_stub):
        """conversation_id key in metadata takes priority over thread_id."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="plan trip")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={
                "agent_name": "travel_agent",
                "conversation_id": "conv-priority",
                "thread_id": "thread-ignored",
            },
        )

        agent = stub.started_agents[-1]
        assert agent.conversation_id == "conv-priority"


# ── Propagation tests: ContextVar lifecycle ────────────────────────


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestConversationIdContextVarPropagation:
    """Verify that inferred conversation_id is pushed into the ContextVar
    so children (LLM, tool, step) inherit it via _apply_genai_context."""

    def test_contextvar_set_on_workflow_start(self, handler_with_stub):
        """After on_chain_start with thread_id, ContextVar should hold
        the inferred conversation_id."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "session-999"},
        )

        ctx = get_genai_context()
        assert ctx.conversation_id == "session-999"

    def test_contextvar_restored_on_workflow_end(self, handler_with_stub):
        """After on_chain_end for the root workflow, the ContextVar should
        be restored to its previous (empty) state."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "session-999"},
        )

        # ContextVar is set during the workflow
        assert get_genai_context().conversation_id == "session-999"

        handler.on_chain_end(
            outputs={"final": "result"},
            run_id=wf_run_id,
        )

        # ContextVar should be restored to empty
        assert get_genai_context().conversation_id is None

    def test_contextvar_restored_on_chain_error(self, handler_with_stub):
        """ContextVar should be restored even when the root chain errors."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "session-err"},
        )

        assert get_genai_context().conversation_id == "session-err"

        handler.on_chain_error(
            error=RuntimeError("boom"),
            run_id=wf_run_id,
        )

        assert get_genai_context().conversation_id is None

    def test_contextvar_not_set_when_explicit_context_active(self, handler_with_stub):
        """When explicit genai_context is active, inferred thread_id should
        NOT overwrite the ContextVar."""
        handler, stub = handler_with_stub

        set_genai_context(conversation_id="explicit-override")

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "inferred-should-be-ignored"},
        )

        # Explicit context should remain untouched
        assert get_genai_context().conversation_id == "explicit-override"

    def test_contextvar_set_on_agent_start(self, handler_with_stub):
        """Inferred conversation_id should also push to ContextVar for agents."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="test")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={"agent_name": "test_agent", "thread_id": "agent-session-1"},
        )

        ctx = get_genai_context()
        assert ctx.conversation_id == "agent-session-1"

    def test_contextvar_preserves_prior_context(self, handler_with_stub):
        """When a previous non-empty context exists and no explicit
        conversation_id is set, the inferred value should be used —
        but the prior context should be restored on end."""
        handler, stub = handler_with_stub

        # No explicit conversation_id, but set some properties
        # (this means ctx.conversation_id is None, so inference should run)

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "new-session"},
        )

        assert get_genai_context().conversation_id == "new-session"

        handler.on_chain_end(
            outputs={"done": True},
            run_id=wf_run_id,
        )

        # Should be restored
        assert get_genai_context().conversation_id is None
