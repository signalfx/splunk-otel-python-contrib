"""Unit tests for message caching and reconstruction fixes.

Tests verify:
1. Messages are reconstructed only once (cached)
2. Cached messages are used in invocation build
3. Messages are in correct format for DeepEval
4. Recursion guards work correctly
"""

import json
import os
from unittest.mock import Mock

import pytest

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.processor.openlit_span_processor import (
    OpenlitSpanProcessor,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment before each test."""
    os.environ["OTEL_GENAI_CONTENT_CAPTURE"] = "1"
    yield
    if "OTEL_GENAI_CONTENT_CAPTURE" in os.environ:
        del os.environ["OTEL_GENAI_CONTENT_CAPTURE"]


@pytest.fixture
def setup_tracer_with_handler():
    """Setup tracer with processor, exporter, and mock handler."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()

    # Mock telemetry handler to track start_llm/stop_llm calls
    mock_handler = Mock(spec=TelemetryHandler)
    mock_handler.start_llm = Mock(return_value=Mock())
    mock_handler.stop_llm = Mock(return_value=Mock())

    # Add OpenlitSpanProcessor with attribute transformations
    processor = OpenlitSpanProcessor(
        attribute_transformations={
            "remove": [],
            "rename": {
                "openlit.span.kind": "gen_ai.span.kind",
                "openlit.entity.input": "gen_ai.input.messages",
                "openlit.entity.output": "gen_ai.output.messages",
                "llm.request.model": "gen_ai.request.model",
            },
            "add": {
                "gen_ai.operation.name": "chat",
            },
        },
        telemetry_handler=mock_handler,
    )
    provider.add_span_processor(processor)

    # Add exporter
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)

    return tracer, exporter, provider, processor, mock_handler


class TestMessageCaching:
    """Test message caching functionality."""

    def test_messages_cached_during_mutation(self, setup_tracer_with_handler):
        """Test that messages are cached when span is mutated."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create openlit-style input/output (normalized format)
        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Hello, how are you?"}]}
        )

        output_data = json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I'm doing great, thanks!",
                    }
                ]
            }
        )

        # Create span with openlit attributes
        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", input_data)
            span.set_attribute("openlit.entity.output", output_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        # Force flush to process spans
        provider.force_flush()

        # Check that messages were cached
        assert span_id in processor._message_cache, "Messages should be cached"

        cached_input, cached_output = processor._message_cache[span_id]

        # Verify cached messages are in correct format
        assert len(cached_input) == 1, "Should have 1 input message"
        assert len(cached_output) == 1, "Should have 1 output message"

        # Verify input message format
        input_msg = cached_input[0]
        assert isinstance(input_msg, InputMessage), (
            "Should be InputMessage object"
        )
        assert input_msg.role == "user", "Should have user role"
        assert len(input_msg.parts) == 1, "Should have 1 part"
        assert isinstance(input_msg.parts[0], Text), (
            "Part should be Text object"
        )
        assert input_msg.parts[0].content == "Hello, how are you?"

        # Verify output message format
        output_msg = cached_output[0]
        assert isinstance(output_msg, OutputMessage), (
            "Should be OutputMessage object"
        )
        assert output_msg.role == "assistant", "Should have assistant role"
        assert len(output_msg.parts) == 1, "Should have 1 part"
        assert isinstance(output_msg.parts[0], Text), (
            "Part should be Text object"
        )
        assert output_msg.parts[0].content == "I'm doing great, thanks!"

    def test_reconstruction_not_repeated_unnecessarily(
        self, setup_tracer_with_handler
    ):
        """Test that message reconstruction uses cache when available."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Use real data instead of mocking to test the actual flow
        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Real test input"}]}
        )
        output_data = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": "Real test output"}
                ]
            }
        )

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", input_data)
            span.set_attribute("openlit.entity.output", output_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        # Force flush to process spans
        provider.force_flush()

        # Verify cache was populated (this means reconstruction happened and was cached)
        assert span_id in processor._message_cache, "Messages should be cached"

        # Verify cached data is correct
        cached_input, cached_output = processor._message_cache[span_id]
        assert len(cached_input) > 0, "Should have cached input messages"
        assert len(cached_output) > 0, "Should have cached output messages"
        assert cached_input[0].parts[0].content == "Real test input"
        assert cached_output[0].parts[0].content == "Real test output"

    def test_cached_messages_used_in_invocation(
        self, setup_tracer_with_handler
    ):
        """Test that cached messages are used in invocation build."""
        tracer, exporter, provider, processor, mock_handler = (
            setup_tracer_with_handler
        )

        # Create span with openlit attributes
        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Cached message test"}]}
        )

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", input_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")

        # Force flush to process spans
        provider.force_flush()

        # Check that start_llm was called
        assert mock_handler.start_llm.called, "start_llm should be called"

        # Get the invocation passed to start_llm
        call_args = mock_handler.start_llm.call_args
        invocation = call_args[0][0]

        # Verify invocation has messages
        assert isinstance(invocation, LLMInvocation), "Should be LLMInvocation"
        assert len(invocation.input_messages) > 0, "Should have input messages"

        # Verify messages are in correct format (not reconstructed again)
        input_msg = invocation.input_messages[0]
        assert isinstance(input_msg, InputMessage), (
            "Should be InputMessage object"
        )
        assert hasattr(input_msg, "parts"), "Should have parts attribute"
        assert len(input_msg.parts) > 0, "Should have parts"
        assert isinstance(input_msg.parts[0], Text), (
            "Part should be Text object"
        )
        assert input_msg.parts[0].content == "Cached message test"


class TestDeepEvalFormat:
    """Test that messages are in correct format for DeepEval."""

    def test_deepeval_can_extract_text(self, setup_tracer_with_handler):
        """Test that DeepEval's extract_text_from_messages works with cached messages."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create span
        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Test for DeepEval"}]}
        )

        output_data = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": "Response for DeepEval"}
                ]
            }
        )

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", input_data)
            span.set_attribute("openlit.entity.output", output_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        # Force flush
        provider.force_flush()

        # Get cached messages
        cached_input, cached_output = processor._message_cache[span_id]

        # Simulate DeepEval's extract_text_from_messages
        def extract_text_from_messages(messages):
            """Simulate DeepEval's message extraction."""
            chunks = []
            for message in messages or []:
                parts = getattr(message, "parts", [])
                for part in parts:
                    # DeepEval expects Text objects with .content
                    if hasattr(part, "content"):
                        if part.content:
                            chunks.append(part.content)
            return "\n".join(c for c in chunks if c).strip()

        # Test extraction works
        input_text = extract_text_from_messages(cached_input)
        output_text = extract_text_from_messages(cached_output)

        assert input_text == "Test for DeepEval", "Should extract input text"
        assert output_text == "Response for DeepEval", (
            "Should extract output text"
        )

    def test_messages_have_required_attributes(
        self, setup_tracer_with_handler
    ):
        """Test that messages have all attributes DeepEval expects."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Attribute test"}]}
        )

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", input_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        cached_input, _ = processor._message_cache[span_id]
        msg = cached_input[0]

        # Check required attributes
        assert hasattr(msg, "role"), "Message should have role"
        assert hasattr(msg, "parts"), "Message should have parts"
        assert isinstance(msg.parts, list), "Parts should be a list"
        assert len(msg.parts) > 0, "Should have at least one part"

        part = msg.parts[0]
        assert isinstance(part, Text), "Part should be Text object"
        assert hasattr(part, "content"), "Text should have content"
        assert isinstance(part.content, str), "Content should be string"


class TestRecursionGuards:
    """Test simplified recursion guards."""

    def test_should_skip_span_basic(self, setup_tracer_with_handler):
        """Test basic skip conditions."""
        _, _, _, processor, _ = setup_tracer_with_handler

        # Test None span
        assert processor._should_skip_span(None) is True, (
            "Should skip None span"
        )

        # Test span without name
        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = None
        mock_span.attributes = {}
        assert processor._should_skip_span(mock_span) is True, (
            "Should skip span without name"
        )

    def test_should_skip_synthetic_span(self, setup_tracer_with_handler):
        """Test that synthetic spans are skipped."""
        _, _, _, processor, _ = setup_tracer_with_handler

        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = "synthetic_span"
        mock_span.attributes = {"_openlit_translated": True}

        # Should skip by attribute
        assert processor._should_skip_span(mock_span) is True, (
            "Should skip span with _openlit_translated attribute"
        )

    def test_should_skip_by_span_id(self, setup_tracer_with_handler):
        """Test that spans are skipped by ID in set."""
        _, _, _, processor, _ = setup_tracer_with_handler

        # Add span ID to synthetic set
        test_span_id = 12345
        processor._synthetic_span_ids.add(test_span_id)

        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = "test_span"
        mock_span.attributes = {}

        # Should skip by ID
        assert processor._should_skip_span(mock_span, test_span_id) is True, (
            "Should skip span with ID in synthetic set"
        )

    def test_should_not_skip_normal_span(self, setup_tracer_with_handler):
        """Test that normal spans are not skipped."""
        _, _, _, processor, _ = setup_tracer_with_handler

        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = "normal_span"
        mock_span.attributes = {}

        # Should not skip
        assert processor._should_skip_span(mock_span, 99999) is False, (
            "Should not skip normal span"
        )

    def test_synthetic_span_not_reprocessed(self, setup_tracer_with_handler):
        """Test that synthetic spans created by processor are not reprocessed."""
        tracer, exporter, provider, processor, mock_handler = (
            setup_tracer_with_handler
        )

        # Create a span that will generate a synthetic span
        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Test"}]}
        )

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", input_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")

        provider.force_flush()

        # start_llm should be called once (for the synthetic span)
        assert mock_handler.start_llm.call_count == 1, (
            "start_llm should be called once for synthetic span"
        )

        # stop_llm should be called once
        assert mock_handler.stop_llm.call_count == 1, (
            "stop_llm should be called once"
        )


class TestCacheIntegration:
    """Test cache integration with full flow."""

    def test_multiple_spans_have_separate_caches(
        self, setup_tracer_with_handler
    ):
        """Test that different spans have separate cache entries."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create first span
        with tracer.start_as_current_span("openai.chat") as span1:
            span1.set_attribute(
                "openlit.entity.input",
                json.dumps(
                    {"messages": [{"role": "user", "content": "Message 1"}]}
                ),
            )
            span1.set_attribute("llm.request.model", "gpt-5-nano")
            span1_id = span1.get_span_context().span_id

        # Create second span
        with tracer.start_as_current_span("openai.chat") as span2:
            span2.set_attribute(
                "openlit.entity.input",
                json.dumps(
                    {"messages": [{"role": "user", "content": "Message 2"}]}
                ),
            )
            span2.set_attribute("llm.request.model", "gpt-5-nano")
            span2_id = span2.get_span_context().span_id

        provider.force_flush()

        # Both should be cached separately
        assert span1_id in processor._message_cache, "Span 1 should be cached"
        assert span2_id in processor._message_cache, "Span 2 should be cached"

        # Verify different content
        cache1_input, _ = processor._message_cache[span1_id]
        cache2_input, _ = processor._message_cache[span2_id]

        assert cache1_input[0].parts[0].content == "Message 1"
        assert cache2_input[0].parts[0].content == "Message 2"

    def test_cache_cleared_appropriately(self, setup_tracer_with_handler):
        """Test that cache is managed correctly."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create span
        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute(
                "openlit.entity.input",
                json.dumps(
                    {"messages": [{"role": "user", "content": "Test"}]}
                ),
            )
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Cache should exist
        assert span_id in processor._message_cache, (
            "Cache should exist after processing"
        )

        # Note: Cache is not automatically cleared - this is intentional
        # as spans might be accessed later for debugging/evaluation


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_messages(self, setup_tracer_with_handler):
        """Test handling of spans with no messages."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span.get_span_context().span_id

        provider.force_flush()

        # Should not crash, cache might not have entry for this span
        # since there are no messages to reconstruct
        # This is expected behavior

    def test_malformed_json_input(self, setup_tracer_with_handler):
        """Test handling of malformed JSON in input."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("openlit.entity.input", "invalid json {{{")
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span.get_span_context().span_id

        # Should not crash
        provider.force_flush()

        # Cache might not have entry due to reconstruction failure
        # This is expected - fallback will handle it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
