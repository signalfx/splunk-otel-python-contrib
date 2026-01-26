"""Tests for parent span filtering logic in _should_sample_for_evaluation."""

import os
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


class TestParentSpanFiltering(unittest.TestCase):
    """Test suite for parent span filtering in evaluation sampling."""

    def setUp(self) -> None:
        """Reset handler singleton before each test."""
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")
        # Use a real tracer provider for span tests
        self.tracer_provider = TracerProvider()

    def _build_llm_invocation(
        self,
        operation: str = "chat",
        parent_run_id=None,
        with_parent_span: bool = False,
    ) -> LLMInvocation:
        """Build a test LLM invocation."""
        invocation = LLMInvocation(
            request_model="test-model",
            operation=operation,
            parent_run_id=parent_run_id,
        )
        invocation.input_messages.append(
            InputMessage(role="user", parts=[Text(content="test")])
        )
        invocation.output_messages.append(
            OutputMessage(
                role="assistant",
                parts=[Text(content="response")],
                finish_reason="stop",
            )
        )

        if with_parent_span:
            # Create a mock span with parent
            mock_span = Mock()
            mock_parent_context = SpanContext(
                trace_id=12345,
                span_id=67890,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
            )
            mock_span.parent = mock_parent_context
            invocation.span = mock_span

        return invocation

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_llm_chat_with_parent_run_id_is_sampled(self) -> None:
        """LLM chat invocation with parent_run_id should be sampled for evaluation."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        parent_id = uuid4()
        invocation = self._build_llm_invocation(
            operation="chat", parent_run_id=parent_id
        )

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "LLM chat invocation with parent_run_id should be sampled",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_llm_chat_with_parent_span_is_sampled(self) -> None:
        """LLM chat invocation with parent span should be sampled for evaluation."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = self._build_llm_invocation(
            operation="chat", with_parent_span=True
        )

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "LLM chat invocation with parent span should be sampled",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_llm_chat_without_parent_is_not_sampled(self) -> None:
        """Root LLM chat invocation (no parent) should NOT be sampled when filter enabled."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = self._build_llm_invocation(operation="chat")

        result = handler._should_sample_for_evaluation(invocation)

        self.assertFalse(
            result,
            "Root LLM chat invocation should NOT be sampled when filter enabled",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "false"},
        clear=True,
    )
    def test_llm_chat_without_parent_is_sampled_when_filter_disabled(
        self,
    ) -> None:
        """Root LLM chat invocation should be sampled when filter is disabled."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = self._build_llm_invocation(operation="chat")

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "Root LLM chat invocation should be sampled when filter disabled",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_llm_embedding_without_parent_is_sampled(self) -> None:
        """Embedding operations should be sampled regardless of parent (filter only applies to chat)."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = EmbeddingInvocation(
            request_model="test-embedding-model",
        )

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "Embedding invocations should be sampled regardless of parent",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_agent_invocation_without_parent_is_sampled(self) -> None:
        """Agent invocations should be sampled regardless of parent (filter only applies to LLM chat)."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = AgentInvocation(name="test-agent")

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "Agent invocations should be sampled regardless of parent",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_llm_non_chat_operation_without_parent_is_sampled(self) -> None:
        """Non-chat LLM operations should be sampled regardless of parent."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = self._build_llm_invocation(operation="completion")

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "Non-chat LLM operations should be sampled regardless of parent",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_parent_run_id_takes_precedence_over_span(self) -> None:
        """parent_run_id check should work even without parent span."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        parent_id = uuid4()
        invocation = self._build_llm_invocation(
            operation="chat",
            parent_run_id=parent_id,
            with_parent_span=False,  # No span parent
        )

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "parent_run_id should be sufficient even without parent span",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_both_parent_checks_present(self) -> None:
        """When both parent_run_id and parent span are present, should be sampled."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        parent_id = uuid4()
        invocation = self._build_llm_invocation(
            operation="chat",
            parent_run_id=parent_id,
            with_parent_span=True,
        )

        result = handler._should_sample_for_evaluation(invocation)

        self.assertTrue(
            result,
            "Invocation with both parent checks should be sampled",
        )


class TestEvaluationSamplingIntegration(unittest.TestCase):
    """Integration tests for evaluation sampling with actual handler lifecycle."""

    def setUp(self) -> None:
        """Reset handler singleton before each test."""
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")
        self.tracer_provider = TracerProvider()

    def _build_invocation(self, parent_run_id=None) -> LLMInvocation:
        """Build a test LLM invocation."""
        invocation = LLMInvocation(
            request_model="test-model",
            operation="chat",
            parent_run_id=parent_run_id,
        )
        invocation.input_messages.append(
            InputMessage(role="user", parts=[Text(content="test")])
        )
        invocation.output_messages.append(
            OutputMessage(
                role="assistant",
                parts=[Text(content="response")],
                finish_reason="stop",
            )
        )
        return invocation

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_invocation_with_parent_marked_for_evaluation(self) -> None:
        """Invocation with parent should have sample_for_evaluation=True."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        parent_id = uuid4()
        invocation = self._build_invocation(parent_run_id=parent_id)

        handler.start_llm(invocation)
        handler.stop_llm(invocation)

        self.assertTrue(
            invocation.sample_for_evaluation,
            "Invocation with parent should be marked for evaluation",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "true"},
        clear=True,
    )
    def test_root_invocation_not_marked_for_evaluation(self) -> None:
        """Root invocation should have sample_for_evaluation=False when filter enabled."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = self._build_invocation()  # No parent

        handler.start_llm(invocation)
        handler.stop_llm(invocation)

        self.assertFalse(
            invocation.sample_for_evaluation,
            "Root invocation should NOT be marked for evaluation",
        )

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVAL_REQUIRE_PARENT_SPAN: "false"},
        clear=True,
    )
    def test_root_invocation_marked_when_filter_disabled(self) -> None:
        """Root invocation should be marked for evaluation when filter is disabled."""
        handler = get_telemetry_handler(tracer_provider=self.tracer_provider)
        invocation = self._build_invocation()  # No parent

        handler.start_llm(invocation)
        handler.stop_llm(invocation)

        self.assertTrue(
            invocation.sample_for_evaluation,
            "Root invocation should be marked when filter disabled",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
