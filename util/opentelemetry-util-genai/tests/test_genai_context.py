"""
Tests for GenAI context management in opentelemetry-util-genai.

This module tests:
- GenAIContext dataclass with properties dict
- Context propagation via set_genai_context/get_genai_context
- Context manager (genai_context) with restore
- _apply_genai_context priority and property merge
- Association properties on spans as gen_ai.association.properties.<key>
- OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION disable flag
"""

import os

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION,
)
from opentelemetry.util.genai.handler import (
    GenAIContext,
    _apply_genai_context,
    clear_genai_context,
    genai_context,
    get_genai_context,
    get_telemetry_handler,
    set_genai_context,
)
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    LLMInvocation,
    RetrievalInvocation,
    Step,
    ToolCall,
    Workflow,
)


class TestGenAIContextDataclass:
    """Tests for GenAIContext dataclass."""

    def test_creation_with_all_fields(self):
        ctx = GenAIContext(
            conversation_id="conv-123",
            properties={"user.id": "alice", "customer.id": "acme"},
        )
        assert ctx.conversation_id == "conv-123"
        assert ctx.properties == {"user.id": "alice", "customer.id": "acme"}

    def test_creation_partial(self):
        ctx = GenAIContext(conversation_id="conv-123")
        assert ctx.conversation_id == "conv-123"
        assert ctx.properties == {}

    def test_creation_empty(self):
        ctx = GenAIContext()
        assert ctx.conversation_id is None
        assert ctx.properties == {}

    def test_is_empty_true(self):
        ctx = GenAIContext()
        assert ctx.is_empty() is True

    def test_is_empty_false_conversation(self):
        ctx = GenAIContext(conversation_id="c")
        assert ctx.is_empty() is False

    def test_is_empty_false_properties(self):
        ctx = GenAIContext(properties={"k": "v"})
        assert ctx.is_empty() is False

    def test_properties_default_factory(self):
        """Each instance gets its own dict."""
        ctx1 = GenAIContext()
        ctx2 = GenAIContext()
        ctx1.properties["k"] = "v"
        assert "k" not in ctx2.properties


class TestGenAIContextFunctions:
    """Tests for set/get/clear GenAI context."""

    def setup_method(self):
        clear_genai_context()

    def teardown_method(self):
        clear_genai_context()

    def test_set_and_get(self):
        set_genai_context(
            conversation_id="conv-abc",
            properties={"user.id": "alice"},
        )
        ctx = get_genai_context()
        assert ctx.conversation_id == "conv-abc"
        assert ctx.properties == {"user.id": "alice"}

    def test_set_conversation_only(self):
        set_genai_context(conversation_id="conv-xyz")
        ctx = get_genai_context()
        assert ctx.conversation_id == "conv-xyz"
        assert ctx.properties == {}

    def test_set_properties_only(self):
        set_genai_context(properties={"tenant": "acme"})
        ctx = get_genai_context()
        assert ctx.conversation_id is None
        assert ctx.properties == {"tenant": "acme"}

    def test_clear(self):
        set_genai_context(conversation_id="conv-123")
        clear_genai_context()
        ctx = get_genai_context()
        assert ctx.conversation_id is None
        assert ctx.properties == {}

    def test_get_when_not_set(self):
        ctx = get_genai_context()
        assert ctx.conversation_id is None
        assert ctx.properties == {}


class TestGenAIContextManager:
    """Tests for genai_context context manager."""

    def setup_method(self):
        clear_genai_context()

    def teardown_method(self):
        clear_genai_context()

    def test_sets_context(self):
        with genai_context(
            conversation_id="ctx-conv",
            properties={"user.id": "bob"},
        ) as ctx:
            assert ctx.conversation_id == "ctx-conv"
            assert ctx.properties == {"user.id": "bob"}
            current = get_genai_context()
            assert current.conversation_id == "ctx-conv"

    def test_clears_on_exit(self):
        with genai_context(conversation_id="temp"):
            pass
        ctx = get_genai_context()
        assert ctx.conversation_id is None

    def test_restores_previous(self):
        set_genai_context(conversation_id="outer")
        with genai_context(conversation_id="inner"):
            assert get_genai_context().conversation_id == "inner"
        assert get_genai_context().conversation_id == "outer"

    def test_nested(self):
        with genai_context(
            conversation_id="outer",
            properties={"level": "1"},
        ):
            assert get_genai_context().conversation_id == "outer"
            with genai_context(
                conversation_id="inner",
                properties={"level": "2"},
            ):
                assert get_genai_context().conversation_id == "inner"
                assert get_genai_context().properties == {"level": "2"}
            assert get_genai_context().conversation_id == "outer"
            assert get_genai_context().properties == {"level": "1"}


class TestApplyGenAIContext:
    """Tests for _apply_genai_context internal function."""

    def setup_method(self):
        clear_genai_context()
        os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION, None)

    def teardown_method(self):
        clear_genai_context()
        os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION, None)

    def test_apply_from_contextvars(self):
        set_genai_context(
            conversation_id="ctx-conv",
            properties={"user.id": "alice", "customer.id": "acme"},
        )
        inv = LLMInvocation(request_model="model-1")
        _apply_genai_context(inv)
        assert inv.conversation_id == "ctx-conv"
        assert inv.association_properties == {
            "user.id": "alice",
            "customer.id": "acme",
        }

    def test_explicit_conversation_has_priority(self):
        set_genai_context(conversation_id="ctx-conv")
        inv = LLMInvocation(
            request_model="model-1",
            conversation_id="explicit-conv",
        )
        _apply_genai_context(inv)
        assert inv.conversation_id == "explicit-conv"

    def test_property_merge_context_then_invocation(self):
        """Invocation-level properties override same-key context properties."""
        set_genai_context(
            properties={"user.id": "ctx-user", "tenant": "ctx-tenant"},
        )
        inv = LLMInvocation(request_model="model-1")
        inv.association_properties = {"user.id": "inv-user", "extra": "val"}
        _apply_genai_context(inv)
        assert inv.association_properties["user.id"] == "inv-user"
        assert inv.association_properties["tenant"] == "ctx-tenant"
        assert inv.association_properties["extra"] == "val"

    def test_no_context_no_change(self):
        inv = LLMInvocation(request_model="model-1")
        _apply_genai_context(inv)
        assert inv.conversation_id is None
        assert inv.association_properties == {}

    def test_propagation_disabled(self):
        """When CONTEXT_PROPAGATION=false, context is not applied."""
        os.environ[OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION] = "false"
        set_genai_context(
            conversation_id="ctx-conv",
            properties={"user.id": "alice"},
        )
        inv = LLMInvocation(request_model="model-1")
        _apply_genai_context(inv)
        assert inv.conversation_id is None
        assert inv.association_properties == {}

    def test_propagation_enabled_explicitly(self):
        """When CONTEXT_PROPAGATION=true, context is applied normally."""
        os.environ[OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION] = "true"
        set_genai_context(conversation_id="ctx-conv")
        inv = LLMInvocation(request_model="model-1")
        _apply_genai_context(inv)
        assert inv.conversation_id == "ctx-conv"

    def test_propagation_default_is_enabled(self):
        """By default (no env var), context propagation is enabled."""
        set_genai_context(conversation_id="ctx-conv")
        inv = LLMInvocation(request_model="model-1")
        _apply_genai_context(inv)
        assert inv.conversation_id == "ctx-conv"

    def test_propagation_disabled_explicit_value_preserved(self):
        """Even when propagation disabled, explicit invocation values stay."""
        os.environ[OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION] = "false"
        set_genai_context(conversation_id="ctx-conv")
        inv = LLMInvocation(
            request_model="model-1",
            conversation_id="explicit-conv",
        )
        inv.association_properties = {"key": "val"}
        _apply_genai_context(inv)
        # Explicit values on invocation are untouched
        assert inv.conversation_id == "explicit-conv"
        assert inv.association_properties == {"key": "val"}


class TestGenAIContextInHandler:
    """Tests for GenAI context integration with TelemetryHandler."""

    def setup_method(self):
        clear_genai_context()

    def teardown_method(self):
        clear_genai_context()

    def test_start_llm(self):
        handler = get_telemetry_handler()
        set_genai_context(
            conversation_id="llm-conv",
            properties={"user.id": "alice"},
        )
        inv = LLMInvocation(request_model="gpt-4")
        handler.start_llm(inv)
        assert inv.conversation_id == "llm-conv"
        assert inv.association_properties == {"user.id": "alice"}
        handler.stop_llm(inv)

    def test_start_embedding(self):
        handler = get_telemetry_handler()
        set_genai_context(conversation_id="emb-conv")
        inv = EmbeddingInvocation(
            request_model="text-embedding-3-small",
            input_texts=["hello"],
        )
        handler.start_embedding(inv)
        assert inv.conversation_id == "emb-conv"
        handler.stop_embedding(inv)

    def test_start_agent(self):
        handler = get_telemetry_handler()
        set_genai_context(
            conversation_id="agent-conv",
            properties={"customer.id": "tenant-1"},
        )
        inv = AgentInvocation(name="test-agent")
        handler.start_agent(inv)
        assert inv.conversation_id == "agent-conv"
        assert inv.association_properties == {"customer.id": "tenant-1"}
        handler.stop_agent(inv)

    def test_start_workflow(self):
        handler = get_telemetry_handler()
        set_genai_context(conversation_id="wf-conv")
        wf = Workflow(name="test-workflow")
        handler.start_workflow(wf)
        assert wf.conversation_id == "wf-conv"
        handler.stop_workflow(wf)

    def test_start_tool_call(self):
        handler = get_telemetry_handler()
        set_genai_context(conversation_id="tool-conv")
        tool = ToolCall(name="search")
        handler.start_tool_call(tool)
        assert tool.conversation_id == "tool-conv"
        handler.stop_tool_call(tool)

    def test_start_retrieval(self):
        handler = get_telemetry_handler()
        set_genai_context(conversation_id="retrieval-conv")
        inv = RetrievalInvocation(query="test query")
        handler.start_retrieval(inv)
        assert inv.conversation_id == "retrieval-conv"
        handler.stop_retrieval(inv)

    def test_start_step(self):
        handler = get_telemetry_handler()
        set_genai_context(conversation_id="step-conv")
        step = Step(name="test-step")
        handler.start_step(step)
        assert step.conversation_id == "step-conv"
        handler.stop_step(step)

    def test_with_context_manager(self):
        handler = get_telemetry_handler()
        with genai_context(
            conversation_id="conv-123",
            properties={"user.id": "alice"},
        ):
            inv = LLMInvocation(request_model="gpt-4")
            handler.start_llm(inv)
            assert inv.conversation_id == "conv-123"
            assert inv.association_properties == {"user.id": "alice"}
            handler.stop_llm(inv)


class TestSemanticConventionAttributes:
    """Tests for context attributes in semantic conventions."""

    def test_conversation_id(self):
        inv = LLMInvocation(
            request_model="gpt-4",
            conversation_id="conv-123",
        )
        attrs = inv.semantic_convention_attributes()
        assert attrs["gen_ai.conversation.id"] == "conv-123"

    def test_association_properties_prefix(self):
        inv = LLMInvocation(request_model="gpt-4")
        inv.association_properties = {
            "user.id": "alice",
            "customer.id": "acme",
        }
        attrs = inv.semantic_convention_attributes()
        assert attrs["gen_ai.association.properties.user.id"] == "alice"
        assert attrs["gen_ai.association.properties.customer.id"] == "acme"

    def test_none_values_excluded(self):
        inv = LLMInvocation(request_model="gpt-4")
        attrs = inv.semantic_convention_attributes()
        assert "gen_ai.conversation.id" not in attrs
        prefix_keys = [
            k for k in attrs if k.startswith("gen_ai.association.properties.")
        ]
        assert prefix_keys == []

    def test_all_invocation_types_have_fields(self):
        types_to_test = [
            LLMInvocation(request_model="model"),
            EmbeddingInvocation(request_model="model", input_texts=["t"]),
            RetrievalInvocation(query="q"),
            ToolCall(name="tool"),
            AgentInvocation(name="agent"),
            Workflow(name="workflow"),
            Step(name="step"),
        ]
        for inv in types_to_test:
            assert hasattr(inv, "conversation_id"), f"{type(inv).__name__}"
            assert hasattr(inv, "association_properties"), (
                f"{type(inv).__name__}"
            )

            inv.conversation_id = "test-conv"
            inv.association_properties = {"user.id": "test-user"}

            attrs = inv.semantic_convention_attributes()
            assert attrs.get("gen_ai.conversation.id") == "test-conv"
            assert (
                attrs.get("gen_ai.association.properties.user.id")
                == "test-user"
            )
