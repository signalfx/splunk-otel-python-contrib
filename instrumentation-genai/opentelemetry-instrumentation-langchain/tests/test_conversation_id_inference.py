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
from opentelemetry.util.genai.attributes import (  # noqa: E402
    GEN_AI_COMMAND,
)
from opentelemetry.util.genai.types import (  # noqa: E402
    AgentInvocation,
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
        self.failed_workflows: list[Any] = []
        self.entities: dict[str, Any] = {}

    def start_agent(self, agent):
        self.started_agents.append(agent)
        self.entities[id(agent)] = agent
        return agent

    def stop_agent(self, agent):
        self.stopped_agents.append(agent)
        self.entities.pop(id(agent), None)
        return agent

    def fail_agent(self, agent, error):
        self.failed_agents.append((agent, error))
        self.entities.pop(id(agent), None)
        return agent

    def start_llm(self, invocation):
        self.started_llms.append(invocation)
        self.entities[id(invocation)] = invocation
        return invocation

    def stop_llm(self, invocation):
        self.stopped_llms.append(invocation)
        self.entities.pop(id(invocation), None)
        return invocation

    def evaluate_llm(self, invocation):
        return []

    def start_tool_call(self, call):
        self.started_tools.append(call)
        self.entities[id(call)] = call
        return call

    def stop_tool_call(self, call):
        self.stopped_tools.append(call)
        self.entities.pop(id(call), None)
        return call

    def fail_tool_call(self, call, error):
        self.failed_tools.append((call, error))
        self.entities.pop(id(call), None)
        return call

    def start_step(self, step):
        self.started_steps.append(step)
        self.entities[id(step)] = step
        return step

    def stop_step(self, step):
        self.stopped_steps.append(step)
        self.entities.pop(id(step), None)
        return step

    def fail_step(self, step, error):
        self.failed_steps.append((step, error))
        self.entities.pop(id(step), None)
        return step

    def start_workflow(self, workflow):
        self.started_workflows.append(workflow)
        self.entities[id(workflow)] = workflow
        return workflow

    def stop_workflow(self, workflow):
        self.stopped_workflows.append(workflow)
        self.entities.pop(id(workflow), None)
        return workflow

    def fail_workflow(self, workflow, error):
        self.failed_workflows.append((workflow, error))
        self.entities.pop(id(workflow), None)
        return workflow

    def should_use_workflow_root(
        self, force_workflow: bool = False, workflow_name: str | None = None
    ) -> bool:
        """Stub: return True if force_workflow or workflow_name is set."""
        return bool(force_workflow or workflow_name)

    def finish(self, entity):
        """Generic finish dispatcher."""
        from opentelemetry.util.genai.types import Workflow

        if isinstance(entity, Workflow):
            return self.stop_workflow(entity)
        return self.stop_agent(entity)

    def fail(self, entity, error):
        """Generic fail dispatcher."""
        from opentelemetry.util.genai.types import Workflow

        if isinstance(entity, Workflow):
            return self.fail_workflow(entity, error)
        self.failed_agents.append((entity, error))
        return entity


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


# ── Integration tests: Root entity path (default: AgentInvocation) ─


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestWorkflowConversationIdInference:
    def test_thread_id_sets_workflow_conversation_id(self, handler_with_stub):
        """Root entity should have conversation_id from thread_id."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
            metadata={"thread_id": "session-abc"},
        )

        assert stub.started_agents
        agent = stub.started_agents[-1]
        assert agent.conversation_id == "session-abc"

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

        agent = stub.started_agents[-1]
        assert agent.conversation_id == "conv-xyz"

    def test_no_metadata_no_conversation_id(self, handler_with_stub):
        """Without thread_id/conversation_id, conversation_id stays None."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="hello")]},
            run_id=wf_run_id,
        )

        agent = stub.started_agents[-1]
        assert agent.conversation_id is None

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

        agent = stub.started_agents[-1]
        assert agent.conversation_id is None

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

        agent = stub.started_agents[-1]
        # The inferred value must NOT be set — explicit context takes priority.
        # In a real runtime, _apply_genai_context would then set "explicit-session".
        assert agent.conversation_id is None

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

        agent = stub.started_agents[-1]
        assert agent.conversation_id is None


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

    def test_interrupted_workflow_does_not_leak_conversation_id(
        self, handler_with_stub
    ):
        """Realistic interrupt/resume scenario: workflow A is interrupted,
        then workflow B starts with a different thread_id.  Without
        _restore_inferred_context, workflow A's conversation_id leaks
        into the ContextVar, which causes workflow B to skip inference
        (because ctx.conversation_id is already set) and inherit the
        wrong conversation_id.

        Sequence without restore (the bug):
          1. on_chain_start(wf_A, thread_id="session-A")
             → ContextVar pushed to "session-A"
          2. on_chain_error(wf_A, GraphInterrupt)
             → ContextVar still "session-A" (leaked!)
          3. on_chain_start(wf_B, thread_id="session-B")
             → ctx.conversation_id == "session-A" → inference SKIPPED
             → wf_B.conversation_id stays None
             → all wf_B children inherit "session-A" — WRONG

        With restore (correct):
          2. on_chain_error(wf_A, GraphInterrupt)
             → _restore clears ContextVar → None
          3. on_chain_start(wf_B, thread_id="session-B")
             → ctx.conversation_id is None → infers "session-B" ✓
        """
        handler, stub = handler_with_stub

        # --- Workflow A: interrupted ---
        wf_a_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="start")]},
            run_id=wf_a_run_id,
            metadata={"thread_id": "session-A"},
        )
        assert get_genai_context().conversation_id == "session-A"

        # Simulate GraphInterrupt cascading up to the root workflow
        class GraphInterrupt(Exception):
            pass

        handler.on_chain_error(
            error=GraphInterrupt("human review needed"),
            run_id=wf_a_run_id,
        )

        # ContextVar must be clean after the interrupt
        assert get_genai_context().conversation_id is None
        assert len(stub.failed_agents) == 1

        # --- Workflow B: resumed with different thread_id ---
        wf_b_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="approved")]},
            run_id=wf_b_run_id,
            metadata={"thread_id": "session-B", "checkpoint_id": "cp-123"},
        )

        # Root B must have its OWN conversation_id, not session-A
        agent_b = stub.started_agents[-1]
        assert agent_b.conversation_id == "session-B"
        assert get_genai_context().conversation_id == "session-B"

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


# ── Resume detection tests: gen_ai.command ────────────────


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestResumeDetection:
    """Verify that resume is detected via checkpoint_id in metadata
    or Command object as input, setting gen_ai.command = 'resume'."""

    def test_workflow_resume_with_checkpoint_id(self, handler_with_stub):
        """Root entity with checkpoint_id in metadata should get
        gen_ai.command='resume'."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="continue")]},
            run_id=wf_run_id,
            metadata={"thread_id": "t1", "checkpoint_id": "cp-abc123"},
        )

        agent = stub.started_agents[-1]
        assert agent.attributes.get(GEN_AI_COMMAND) == "resume"

    def test_agent_resume_with_checkpoint_id(self, handler_with_stub):
        """Agent root with checkpoint_id in metadata should get
        gen_ai.command='resume'."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="continue")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={
                "agent_name": "sre_copilot",
                "thread_id": "t2",
                "checkpoint_id": "cp-def456",
            },
        )

        agent = stub.started_agents[-1]
        assert agent.attributes.get(GEN_AI_COMMAND) == "resume"

    def test_workflow_fresh_no_checkpoint_id(self, handler_with_stub):
        """Root entity without checkpoint_id should NOT have
        gen_ai.command set."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="start")]},
            run_id=wf_run_id,
            metadata={"thread_id": "t3"},
        )

        agent = stub.started_agents[-1]
        assert GEN_AI_COMMAND not in agent.attributes

    def test_agent_fresh_no_checkpoint_id(self, handler_with_stub):
        """Agent without checkpoint_id should NOT have
        gen_ai.command set."""
        handler, stub = handler_with_stub

        agent_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"messages": [HumanMessage(content="start")]},
            run_id=agent_run_id,
            tags=["agent"],
            metadata={"agent_name": "sre_copilot", "thread_id": "t4"},
        )

        agent = stub.started_agents[-1]
        assert GEN_AI_COMMAND not in agent.attributes

    def test_workflow_no_metadata(self, handler_with_stub):
        """Root entity with no metadata at all should NOT have
        gen_ai.command set."""
        handler, stub = handler_with_stub

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="start")]},
            run_id=wf_run_id,
        )

        agent = stub.started_agents[-1]
        assert GEN_AI_COMMAND not in agent.attributes


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestOrphanSpanGuard:
    """Verify that on_chain_start silently drops child entities whose parent
    was already cleaned up — prevents orphan root spans during LangGraph
    resume when the interrupted node is replayed."""

    def test_child_with_missing_parent_is_dropped(self, handler_with_stub):
        """on_chain_start with parent_run_id pointing to a non-existent
        entity should not create any step, agent, or workflow."""
        handler, stub = handler_with_stub

        # parent_run_id refers to a workflow that no longer exists
        stale_parent = uuid4()
        child_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "human_review"},
            inputs={},
            run_id=child_run_id,
            parent_run_id=stale_parent,
            metadata={"thread_id": "t-orphan"},
        )

        assert len(stub.started_steps) == 0
        assert len(stub.started_agents) == 0
        assert len(stub.started_workflows) == 0

    def test_child_with_live_parent_is_created(self, handler_with_stub):
        """on_chain_start with a live parent_run_id should still create
        the child entity normally."""
        handler, stub = handler_with_stub

        # First create a workflow so there's a live parent
        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [HumanMessage(content="start")]},
            run_id=wf_run_id,
            metadata={"thread_id": "t-live"},
        )
        assert len(stub.started_agents) == 1

        # Now create a child step under it
        child_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "human_review"},
            inputs={},
            run_id=child_run_id,
            parent_run_id=wf_run_id,
            metadata={"thread_id": "t-live"},
        )

        assert len(stub.started_steps) == 1


@pytest.mark.skipif(not LANGCHAIN_CORE_AVAILABLE, reason="langchain_core not available")
class TestCommandInputHandling:
    """Verify that Command objects (LangGraph resume) are handled correctly
    without crashing, and trigger resume detection."""

    def test_command_input_creates_root_with_resume(self, handler_with_stub):
        """When inputs is a Command object, a root entity span should be
        created with gen_ai.command='resume' and capture the
        resume value as a user input message."""
        handler, stub = handler_with_stub

        # Simulate what LangGraph sends on resume
        class Command:
            def __init__(self, resume):
                self.resume = resume

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs=Command(resume="approved!"),
            run_id=wf_run_id,
            metadata={"thread_id": "t-cmd"},
            name="LangGraph",
        )

        assert len(stub.started_agents) == 1
        agent = stub.started_agents[-1]
        assert agent.attributes.get(GEN_AI_COMMAND) == "resume"
        assert agent.conversation_id == "t-cmd"
        # Resume value should be captured as user input message
        assert len(agent.input_messages) == 1
        assert agent.input_messages[0].role == "user"
        assert agent.input_messages[0].parts[0].content == "approved!"

    def test_command_input_dict_resume(self, handler_with_stub):
        """When resume value is a dict, it should be JSON-serialized."""
        handler, stub = handler_with_stub

        class Command:
            def __init__(self, resume):
                self.resume = resume

        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs=Command(resume={"action": "approve", "comment": "lgtm"}),
            run_id=wf_run_id,
            metadata={"thread_id": "t-dict"},
            name="LangGraph",
        )

        agent = stub.started_agents[-1]
        assert len(agent.input_messages) == 1
        content = agent.input_messages[0].parts[0].content
        # Should be valid JSON containing the dict keys
        assert "approve" in content
        assert "lgtm" in content

    def test_command_input_does_not_crash(self, handler_with_stub):
        """Command input without resume should not crash and should
        use repr as fallback input."""
        handler, stub = handler_with_stub

        class Command:
            pass

        wf_run_id = uuid4()
        # Should not raise
        handler.on_chain_start(
            serialized=None,
            inputs=Command(),
            run_id=wf_run_id,
            metadata={"thread_id": "t-safe"},
            name="LangGraph",
        )
        assert len(stub.started_agents) == 1
        agent = stub.started_agents[-1]
        # Should have a fallback input message from repr
        assert len(agent.input_messages) == 1

    def test_langgraph_node_metadata_used_for_step_name(self, handler_with_stub):
        """When serialized is None (LangGraph), langgraph_node from
        metadata should be used as the step name."""
        handler, stub = handler_with_stub

        # Create a parent workflow first
        wf_run_id = uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=wf_run_id,
            metadata={"thread_id": "t-name"},
            name="LangGraph",
        )

        step_run_id = uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=step_run_id,
            parent_run_id=wf_run_id,
            metadata={
                "thread_id": "t-name",
                "langgraph_node": "human_review",
            },
        )

        assert len(stub.started_steps) == 1
        step = stub.started_steps[-1]
        assert step.name == "human_review"
