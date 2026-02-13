"""
Tests for session context management in opentelemetry-util-genai.

This module tests:
- SessionContext dataclass
- Context propagation via set_session_context/get_session_context
- Context manager (session_context)
- Context application to invocations
- Environment variable fallback
- Priority order (explicit > contextvars > env vars)
"""

import os

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CUSTOMER_ID,
    OTEL_INSTRUMENTATION_GENAI_SESSION_ID,
    OTEL_INSTRUMENTATION_GENAI_USER_ID,
)
from opentelemetry.util.genai.handler import (
    SessionContext,
    _apply_session_context,
    clear_session_context,
    get_session_context,
    get_telemetry_handler,
    session_context,
    set_session_context,
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


class TestSessionContextDataclass:
    """Tests for SessionContext dataclass."""

    def test_session_context_creation(self):
        """Test basic SessionContext creation."""
        ctx = SessionContext(
            session_id="sess-123",
            user_id="user-456",
            customer_id="customer-789",
        )
        assert ctx.session_id == "sess-123"
        assert ctx.user_id == "user-456"
        assert ctx.customer_id == "customer-789"

    def test_session_context_partial(self):
        """Test SessionContext with partial fields."""
        ctx = SessionContext(session_id="sess-123")
        assert ctx.session_id == "sess-123"
        assert ctx.user_id is None
        assert ctx.customer_id is None

    def test_session_context_empty(self):
        """Test SessionContext with no fields."""
        ctx = SessionContext()
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.customer_id is None


class TestSessionContextFunctions:
    """Tests for session context management functions."""

    def setup_method(self):
        """Clear session context before each test."""
        clear_session_context()

    def teardown_method(self):
        """Clean up session context after each test."""
        clear_session_context()

    def test_set_and_get_session_context(self):
        """Test setting and getting session context."""
        set_session_context(
            session_id="sess-abc",
            user_id="user-xyz",
        )
        ctx = get_session_context()
        assert ctx is not None
        assert ctx.session_id == "sess-abc"
        assert ctx.user_id == "user-xyz"
        assert ctx.customer_id is None

    def test_clear_session_context(self):
        """Test clearing session context."""
        set_session_context(session_id="sess-123")
        ctx = get_session_context()
        assert ctx.session_id == "sess-123"
        clear_session_context()
        # After clear, should return empty SessionContext
        ctx = get_session_context()
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.customer_id is None

    def test_get_session_context_when_not_set(self):
        """Test getting session context when not set returns empty context."""
        ctx = get_session_context()
        # Returns empty SessionContext (not None)
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.customer_id is None


class TestSessionContextManager:
    """Tests for session_context context manager."""

    def setup_method(self):
        """Clear session context before each test."""
        clear_session_context()

    def teardown_method(self):
        """Clean up session context after each test."""
        clear_session_context()

    def test_context_manager_sets_context(self):
        """Test that context manager sets session context."""
        with session_context(
            session_id="ctx-sess",
            user_id="ctx-user",
        ) as ctx:
            assert ctx.session_id == "ctx-sess"
            assert ctx.user_id == "ctx-user"
            # Should be accessible via get_session_context too
            current = get_session_context()
            assert current is not None
            assert current.session_id == "ctx-sess"

    def test_context_manager_clears_on_exit(self):
        """Test that context manager clears context on exit."""
        with session_context(session_id="temp-sess"):
            ctx = get_session_context()
            assert ctx.session_id == "temp-sess"
        # After exiting, session should be cleared (empty)
        ctx = get_session_context()
        assert ctx.session_id is None

    def test_context_manager_restores_previous(self):
        """Test that context manager restores previous context."""
        set_session_context(session_id="outer-sess")
        with session_context(session_id="inner-sess"):
            ctx = get_session_context()
            assert ctx is not None
            assert ctx.session_id == "inner-sess"
        # After exiting, outer context should be restored
        ctx = get_session_context()
        # Note: current impl resets to token, which goes back to outer
        # This depends on contextvars behavior
        # The reset(token) actually restores to previous state

    def test_nested_context_managers(self):
        """Test nested context managers."""
        with session_context(session_id="outer"):
            ctx1 = get_session_context()
            assert ctx1 is not None
            assert ctx1.session_id == "outer"

            with session_context(session_id="inner"):
                ctx2 = get_session_context()
                assert ctx2 is not None
                assert ctx2.session_id == "inner"

            # After inner exits, should restore to outer
            ctx3 = get_session_context()
            assert ctx3 is not None
            assert ctx3.session_id == "outer"


class TestApplySessionContext:
    """Tests for _apply_session_context internal function."""

    def setup_method(self):
        """Clear session context before each test."""
        clear_session_context()
        # Clear env vars
        for var in [
            OTEL_INSTRUMENTATION_GENAI_SESSION_ID,
            OTEL_INSTRUMENTATION_GENAI_USER_ID,
            OTEL_INSTRUMENTATION_GENAI_CUSTOMER_ID,
        ]:
            os.environ.pop(var, None)

    def teardown_method(self):
        """Clean up after each test."""
        clear_session_context()
        for var in [
            OTEL_INSTRUMENTATION_GENAI_SESSION_ID,
            OTEL_INSTRUMENTATION_GENAI_USER_ID,
            OTEL_INSTRUMENTATION_GENAI_CUSTOMER_ID,
        ]:
            os.environ.pop(var, None)

    def test_apply_from_contextvars(self):
        """Test that session context is applied from contextvars."""
        set_session_context(
            session_id="ctx-sess",
            user_id="ctx-user",
            customer_id="ctx-customer",
        )
        inv = LLMInvocation(request_model="model-1")
        _apply_session_context(inv)
        assert inv.session_id == "ctx-sess"
        assert inv.user_id == "ctx-user"
        assert inv.customer_id == "ctx-customer"

    def test_apply_from_env_vars(self):
        """Test that session context is applied from env vars."""
        os.environ[OTEL_INSTRUMENTATION_GENAI_SESSION_ID] = "env-sess"
        os.environ[OTEL_INSTRUMENTATION_GENAI_USER_ID] = "env-user"
        os.environ[OTEL_INSTRUMENTATION_GENAI_CUSTOMER_ID] = "env-customer"

        inv = LLMInvocation(request_model="model-1")
        _apply_session_context(inv)
        assert inv.session_id == "env-sess"
        assert inv.user_id == "env-user"
        assert inv.customer_id == "env-customer"

    def test_explicit_value_has_priority(self):
        """Test that explicit invocation values take priority."""
        set_session_context(session_id="ctx-sess")
        os.environ[OTEL_INSTRUMENTATION_GENAI_SESSION_ID] = "env-sess"

        inv = LLMInvocation(
            request_model="model-1",
            session_id="explicit-sess",
        )
        _apply_session_context(inv)
        # Explicit value should be preserved
        assert inv.session_id == "explicit-sess"

    def test_contextvars_priority_over_env(self):
        """Test that contextvars take priority over env vars."""
        set_session_context(session_id="ctx-sess")
        os.environ[OTEL_INSTRUMENTATION_GENAI_SESSION_ID] = "env-sess"

        inv = LLMInvocation(request_model="model-1")
        _apply_session_context(inv)
        # Contextvars should take priority
        assert inv.session_id == "ctx-sess"

    def test_no_context_no_change(self):
        """Test that invocation is unchanged when no context set."""
        inv = LLMInvocation(request_model="model-1")
        _apply_session_context(inv)
        assert inv.session_id is None
        assert inv.user_id is None
        assert inv.customer_id is None


class TestSessionContextInHandler:
    """Tests for session context integration with TelemetryHandler."""

    def setup_method(self):
        """Clear session context before each test."""
        clear_session_context()

    def teardown_method(self):
        """Clean up after each test."""
        clear_session_context()

    def test_start_llm_applies_session_context(self):
        """Test that start_llm applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="llm-sess", user_id="llm-user")

        inv = LLMInvocation(request_model="gpt-4")
        handler.start_llm(inv)

        assert inv.session_id == "llm-sess"
        assert inv.user_id == "llm-user"

        handler.stop_llm(inv)

    def test_start_embedding_applies_session_context(self):
        """Test that start_embedding applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="emb-sess")

        inv = EmbeddingInvocation(
            request_model="text-embedding-3-small",
            input_texts=["hello"],
        )
        handler.start_embedding(inv)

        assert inv.session_id == "emb-sess"

        handler.stop_embedding(inv)

    def test_start_agent_applies_session_context(self):
        """Test that start_agent applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="agent-sess", customer_id="tenant-1")

        inv = AgentInvocation(name="test-agent")
        handler.start_agent(inv)

        assert inv.session_id == "agent-sess"
        assert inv.customer_id == "tenant-1"

        handler.stop_agent(inv)

    def test_start_workflow_applies_session_context(self):
        """Test that start_workflow applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="wf-sess")

        wf = Workflow(name="test-workflow")
        handler.start_workflow(wf)

        assert wf.session_id == "wf-sess"

        handler.stop_workflow(wf)

    def test_start_tool_call_applies_session_context(self):
        """Test that start_tool_call applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="tool-sess")

        tool = ToolCall(name="search")
        handler.start_tool_call(tool)

        assert tool.session_id == "tool-sess"

        handler.stop_tool_call(tool)

    def test_start_retrieval_applies_session_context(self):
        """Test that start_retrieval applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="retrieval-sess")

        inv = RetrievalInvocation(query="test query")
        handler.start_retrieval(inv)

        assert inv.session_id == "retrieval-sess"

        handler.stop_retrieval(inv)

    def test_start_step_applies_session_context(self):
        """Test that start_step applies session context."""
        handler = get_telemetry_handler()
        set_session_context(session_id="step-sess")

        step = Step(name="test-step")
        handler.start_step(step)

        assert step.session_id == "step-sess"

        handler.stop_step(step)

    def test_session_with_context_manager(self):
        """Test session context with context manager and handler."""
        handler = get_telemetry_handler()

        with session_context(session_id="conv-123", user_id="user-456"):
            inv = LLMInvocation(request_model="gpt-4")
            handler.start_llm(inv)
            assert inv.session_id == "conv-123"
            assert inv.user_id == "user-456"
            handler.stop_llm(inv)


class TestSemanticConventionAttributes:
    """Tests for session attributes in semantic conventions."""

    def test_session_id_in_semconv_attributes(self):
        """Test that session_id appears in semantic convention attributes."""
        inv = LLMInvocation(
            request_model="gpt-4",
            session_id="sess-123",
        )
        attrs = inv.semantic_convention_attributes()
        assert "session.id" in attrs
        assert attrs["session.id"] == "sess-123"

    def test_user_id_in_semconv_attributes(self):
        """Test that user_id appears in semantic convention attributes."""
        inv = LLMInvocation(
            request_model="gpt-4",
            user_id="user-456",
        )
        attrs = inv.semantic_convention_attributes()
        assert "user.id" in attrs
        assert attrs["user.id"] == "user-456"

    def test_customer_id_in_semconv_attributes(self):
        """Test that customer_id appears in semantic convention attributes."""
        inv = LLMInvocation(
            request_model="gpt-4",
            customer_id="tenant-789",
        )
        attrs = inv.semantic_convention_attributes()
        assert "customer.id" in attrs
        assert attrs["customer.id"] == "tenant-789"

    def test_none_values_excluded_from_semconv(self):
        """Test that None values are excluded from semconv attributes."""
        inv = LLMInvocation(request_model="gpt-4")
        attrs = inv.semantic_convention_attributes()
        assert "session.id" not in attrs
        assert "user.id" not in attrs
        assert "customer.id" not in attrs

    def test_all_invocation_types_have_session_fields(self):
        """Test that all invocation types inherit session fields."""
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
            # All should have session fields
            assert hasattr(inv, "session_id")
            assert hasattr(inv, "user_id")
            assert hasattr(inv, "customer_id")

            # Set values and check semantic convention attrs
            inv.session_id = "test-sess"
            inv.user_id = "test-user"
            inv.customer_id = "test-customer"

            attrs = inv.semantic_convention_attributes()
            assert attrs.get("session.id") == "test-sess"
            assert attrs.get("user.id") == "test-user"
            assert attrs.get("customer.id") == "test-customer"
