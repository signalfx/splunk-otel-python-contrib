"""Test message reconstruction for agent and task spans.

This test module verifies that the updated logic in _mutate_span_if_needed
correctly reconstructs messages for agent and task spans in addition to
LLM operation spans (chat, completion, embedding).
"""

import json
import os
from unittest.mock import Mock

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.processor.traceloop_span_processor import (
    TraceloopSpanProcessor,
)
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text


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
    provider = TracerProvider()
    exporter = InMemorySpanExporter()

    # Create mock handler
    mock_handler = Mock(spec=TelemetryHandler)

    # Mock return value for get_telemetry_handler
    mock_handler.start_llm = Mock()
    mock_handler.stop_llm = Mock()

    # Create processor with transformation rules
    attribute_transformations = {
        "rename": {
            "traceloop.span.kind": "gen_ai.span.kind",
            "traceloop.entity.name": "gen_ai.agent.name",
            "traceloop.entity.input": "gen_ai.input.messages",
            "traceloop.entity.output": "gen_ai.output.messages",
            "traceloop.association.properties.ls_model_name": "gen_ai.request.model",
            "llm.request.model": "gen_ai.request.model",
        },
        "add": {
            "gen_ai.system": "traceloop",
            "gen_ai.operation.name": "chat",
        },
    }

    processor = TraceloopSpanProcessor(
        attribute_transformations=attribute_transformations,
        telemetry_handler=mock_handler,
        mutate_original_span=True,
    )

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(processor)

    tracer = provider.get_tracer(__name__)

    return tracer, exporter, provider, processor, mock_handler


class TestAgentMessageReconstruction:
    """Test message reconstruction for agent spans."""

    def test_agent_span_reconstructs_messages(self, setup_tracer_with_handler):
        """Test that agent spans trigger message reconstruction."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create Traceloop-style input/output for an agent
        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Plan a trip to Paris"}]}
        )

        output_data = json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I'll help you plan an amazing trip to Paris!",
                    }
                ]
            }
        )

        # Create span with agent attributes
        with tracer.start_as_current_span("travel_coordinator") as span:
            span.set_attribute("traceloop.span.kind", "agent")
            span.set_attribute("traceloop.entity.name", "travel_coordinator")
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span_id = span.get_span_context().span_id

        # Force flush to process spans
        provider.force_flush()

        # Verify that messages were cached (indicating reconstruction happened)
        assert span_id in processor._message_cache, (
            "Messages should be cached for agent span"
        )

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
        assert input_msg.parts[0].content == "Plan a trip to Paris", (
            "Input content should match"
        )

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
        assert "Paris" in output_msg.parts[0].content, (
            "Output should mention Paris"
        )

    def test_agent_span_has_genai_attributes(self, setup_tracer_with_handler):
        """Test that agent spans get gen_ai.input.messages and gen_ai.output.messages."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Book a hotel"}]}
        )

        output_data = json.dumps(
            {"messages": [{"role": "assistant", "content": "Hotel booked!"}]}
        )

        with tracer.start_as_current_span("hotel_agent") as span:
            span.set_attribute("traceloop.span.kind", "agent")
            span.set_attribute("traceloop.entity.name", "hotel_agent")
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)

        provider.force_flush()

        # Get the span from exporter
        spans = exporter.get_finished_spans()
        agent_span = next((s for s in spans if s.name == "hotel_agent"), None)

        assert agent_span is not None, "Should find agent span"
        assert agent_span.attributes is not None, "Span should have attributes"

        # Verify gen_ai attributes are present
        assert "gen_ai.input.messages" in agent_span.attributes, (
            "Should have gen_ai.input.messages"
        )
        assert "gen_ai.output.messages" in agent_span.attributes, (
            "Should have gen_ai.output.messages"
        )
        assert "gen_ai.span.kind" in agent_span.attributes, (
            "Should have gen_ai.span.kind"
        )
        assert agent_span.attributes["gen_ai.span.kind"] == "agent", (
            "Should preserve agent kind"
        )


class TestTaskMessageReconstruction:
    """Test message reconstruction for task spans."""

    def test_task_span_reconstructs_messages(self, setup_tracer_with_handler):
        """Test that task spans trigger message reconstruction."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create Traceloop-style input/output for a task
        input_data = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for flights from Seattle to Paris",
                    }
                ]
            }
        )

        output_data = json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Found 5 flights from Seattle to Paris",
                    }
                ]
            }
        )

        # Create span with task attributes
        with tracer.start_as_current_span("flight_search_task") as span:
            span.set_attribute("traceloop.span.kind", "task")
            span.set_attribute("traceloop.entity.name", "flight_search")
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span_id = span.get_span_context().span_id

        # Force flush to process spans
        provider.force_flush()

        # Verify that messages were cached (indicating reconstruction happened)
        assert span_id in processor._message_cache, (
            "Messages should be cached for task span"
        )

        cached_input, cached_output = processor._message_cache[span_id]

        # Verify cached messages are in correct format
        assert len(cached_input) == 1, "Should have 1 input message"
        assert len(cached_output) == 1, "Should have 1 output message"

        # Verify message content
        assert (
            cached_input[0].parts[0].content
            == "Search for flights from Seattle to Paris"
        ), "Input should match"
        assert "5 flights" in cached_output[0].parts[0].content, (
            "Output should mention flights"
        )

    def test_task_span_without_messages_skips_reconstruction(
        self, setup_tracer_with_handler
    ):
        """Test that task spans without messages don't crash."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create span with task attributes but no messages
        with tracer.start_as_current_span("empty_task") as span:
            span.set_attribute("traceloop.span.kind", "task")
            span.set_attribute("traceloop.entity.name", "empty_task")
            span_id = span.get_span_context().span_id

        # Force flush to process spans
        provider.force_flush()

        # Should not crash, and span_id should NOT be in cache
        assert span_id not in processor._message_cache, (
            "Empty task should not be cached"
        )


class TestLLMOperationMessageReconstruction:
    """Test that LLM operations still work as before."""

    def test_chat_operation_reconstructs_messages(
        self, setup_tracer_with_handler
    ):
        """Test that chat operations (LLM calls) still reconstruct messages."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Hello GPT"}]}
        )

        output_data = json.dumps(
            {"messages": [{"role": "assistant", "content": "Hello!"}]}
        )

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Verify messages were cached
        assert span_id in processor._message_cache, (
            "Messages should be cached for chat operation"
        )

        cached_input, cached_output = processor._message_cache[span_id]
        assert len(cached_input) == 1, "Should have 1 input message"
        assert len(cached_output) == 1, "Should have 1 output message"

    def test_completion_operation_reconstructs_messages(
        self, setup_tracer_with_handler
    ):
        """Test that completion operations reconstruct messages."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Complete this:"}]}
        )

        output_data = json.dumps(
            {"messages": [{"role": "assistant", "content": "Completed!"}]}
        )

        with tracer.start_as_current_span("openai.completion") as span:
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span.set_attribute("gen_ai.operation.name", "completion")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        assert span_id in processor._message_cache, (
            "Messages should be cached for completion operation"
        )


class TestNonLLMSpanSkipsReconstruction:
    """Test that non-LLM spans (workflows, tools, etc.) skip message reconstruction."""

    def test_workflow_span_skips_reconstruction(
        self, setup_tracer_with_handler
    ):
        """Test that workflow spans without messages don't trigger reconstruction."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Create workflow span (not agent, not task, not LLM operation)
        with tracer.start_as_current_span("travel_workflow") as span:
            span.set_attribute("traceloop.span.kind", "workflow")
            span.set_attribute("traceloop.workflow.name", "travel_planner")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Workflow spans should NOT trigger message reconstruction
        assert span_id not in processor._message_cache, (
            "Workflow spans should not cache messages"
        )

    def test_tool_span_skips_reconstruction(self, setup_tracer_with_handler):
        """Test that tool spans skip message reconstruction."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        with tracer.start_as_current_span("search_tool") as span:
            span.set_attribute("traceloop.span.kind", "tool")
            span.set_attribute("traceloop.entity.name", "search_tool")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Tool spans should NOT trigger message reconstruction
        assert span_id not in processor._message_cache, (
            "Tool spans should not cache messages"
        )

    def test_unknown_span_skips_reconstruction(
        self, setup_tracer_with_handler
    ):
        """Test that unknown span types skip message reconstruction."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        with tracer.start_as_current_span("random_span") as span:
            # No traceloop attributes, just a random span
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Random spans should NOT trigger message reconstruction
        assert span_id not in processor._message_cache, (
            "Unknown spans should not cache messages"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_agent_with_malformed_json(self, setup_tracer_with_handler):
        """Test that malformed JSON in agent span doesn't crash."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        with tracer.start_as_current_span("broken_agent") as span:
            span.set_attribute("traceloop.span.kind", "agent")
            span.set_attribute("traceloop.entity.name", "broken_agent")
            # Malformed JSON
            span.set_attribute("traceloop.entity.input", "{invalid json}")
            span_id = span.get_span_context().span_id

        # Should not crash
        provider.force_flush()

        # Malformed data should not be cached
        assert span_id not in processor._message_cache, (
            "Malformed JSON should not be cached"
        )

    def test_task_with_empty_messages(self, setup_tracer_with_handler):
        """Test that task with empty message arrays is handled."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        # Empty messages array
        input_data = json.dumps({"messages": []})
        output_data = json.dumps({"messages": []})

        with tracer.start_as_current_span("empty_task") as span:
            span.set_attribute("traceloop.span.kind", "task")
            span.set_attribute("traceloop.entity.name", "empty_task")
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Empty messages should still be cached (as empty lists)
        if span_id in processor._message_cache:
            cached_input, cached_output = processor._message_cache[span_id]
            assert cached_input == [], (
                "Empty input should be cached as empty list"
            )
            assert cached_output == [], (
                "Empty output should be cached as empty list"
            )

    def test_mixed_span_kinds(self, setup_tracer_with_handler):
        """Test different span kinds in same workflow."""
        tracer, exporter, provider, processor, _ = setup_tracer_with_handler

        input_data = json.dumps(
            {"messages": [{"role": "user", "content": "Test"}]}
        )
        output_data = json.dumps(
            {"messages": [{"role": "assistant", "content": "OK"}]}
        )

        span_ids = {}

        # Agent span (should cache)
        with tracer.start_as_current_span("agent_span") as span:
            span.set_attribute("traceloop.span.kind", "agent")
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span_ids["agent"] = span.get_span_context().span_id

        # Task span (should cache)
        with tracer.start_as_current_span("task_span") as span:
            span.set_attribute("traceloop.span.kind", "task")
            span.set_attribute("traceloop.entity.input", input_data)
            span.set_attribute("traceloop.entity.output", output_data)
            span_ids["task"] = span.get_span_context().span_id

        # Workflow span (should NOT cache)
        with tracer.start_as_current_span("workflow_span") as span:
            span.set_attribute("traceloop.span.kind", "workflow")
            span_ids["workflow"] = span.get_span_context().span_id

        provider.force_flush()

        # Verify caching behavior
        assert span_ids["agent"] in processor._message_cache, (
            "Agent span should cache messages"
        )
        assert span_ids["task"] in processor._message_cache, (
            "Task span should cache messages"
        )
        assert span_ids["workflow"] not in processor._message_cache, (
            "Workflow span should not cache messages"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
