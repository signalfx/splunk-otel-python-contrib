"""Test with real Traceloop format from LangChain/LangGraph.

This test uses the actual format that Traceloop SDK produces, with:
- Nested structure: inputs.messages[] and outputs.messages[]
- LangChain serialization: lc, type, id, kwargs
- Metadata: response_metadata, usage_metadata, etc.
"""

import json
import os

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
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
def setup_tracer():
    """Setup tracer with processor and exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()

    # Add TraceloopSpanProcessor with attribute transformations
    processor = TraceloopSpanProcessor(
        attribute_transformations={
            "remove": [],
            "rename": {
                "traceloop.span.kind": "gen_ai.span.kind",
                "traceloop.entity.input": "gen_ai.input.messages",
                "traceloop.entity.output": "gen_ai.output.messages",
                "llm.request.model": "gen_ai.request.model",
            },
            "add": {
                "gen_ai.operation.name": "chat",
            },
        }
    )
    provider.add_span_processor(processor)

    # Add exporter
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)

    return tracer, exporter, provider, processor


class TestRealTraceloopFormat:
    """Test with actual Traceloop SDK format."""

    def test_real_nested_input_format(self, setup_tracer):
        """Test with real Traceloop nested input format."""
        tracer, exporter, provider, processor = setup_tracer

        # Real Traceloop format with inputs.messages[]
        input_data = {
            "inputs": {
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "schema",
                            "messages",
                            "HumanMessage",
                        ],
                        "kwargs": {
                            "content": "hi! I'm Lance",
                            "type": "human",
                        },
                    }
                ]
            },
            "tags": [],
            "metadata": {"custom_field1": "value1", "thread_id": "1"},
            "kwargs": {"name": "ChatbotSummarizationAgent"},
        }

        # Create span with real format
        with tracer.start_as_current_span("ChatbotSummarizationAgent") as span:
            span.set_attribute(
                "traceloop.entity.input", json.dumps(input_data)
            )
            span.set_attribute("llm.request.model", "gemini-1.5-flash")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Verify messages were cached
        assert span_id in processor._message_cache, "Messages should be cached"

        cached_input, _ = processor._message_cache[span_id]

        # Verify correct extraction
        assert len(cached_input) == 1, "Should have 1 input message"
        assert isinstance(cached_input[0], InputMessage), (
            "Should be InputMessage"
        )
        assert cached_input[0].role == "user", (
            "Should map HumanMessage to user"
        )
        assert len(cached_input[0].parts) == 1, "Should have 1 part"
        assert isinstance(cached_input[0].parts[0], Text), (
            "Part should be Text"
        )
        assert cached_input[0].parts[0].content == "hi! I'm Lance", (
            "Should extract content from kwargs"
        )

    def test_real_nested_output_format(self, setup_tracer):
        """Test with real Traceloop nested output format."""
        tracer, exporter, provider, processor = setup_tracer

        # Real Traceloop format with outputs.messages[]
        output_data = {
            "outputs": {
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "schema", "messages", "AIMessage"],
                        "kwargs": {
                            "content": "Hi Lance! Nice to meet you.\n",
                            "response_metadata": {
                                "prompt_feedback": {
                                    "block_reason": 0,
                                    "safety_ratings": [],
                                },
                                "finish_reason": "STOP",
                                "safety_ratings": [],
                            },
                            "type": "ai",
                            "id": "run-d7f042aa-b7a9-48ec-9adc-d59df02be09c-0",
                            "usage_metadata": {
                                "input_tokens": 6,
                                "output_tokens": 10,
                                "total_tokens": 16,
                                "input_token_details": {"cache_read": 0},
                            },
                            "tool_calls": [],
                            "invalid_tool_calls": [],
                        },
                    }
                ]
            },
            "kwargs": {"tags": []},
        }

        # Create span with real format
        with tracer.start_as_current_span("ChatbotSummarizationAgent") as span:
            span.set_attribute(
                "traceloop.entity.output", json.dumps(output_data)
            )
            span.set_attribute("llm.request.model", "gemini-1.5-flash")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Verify messages were cached
        assert span_id in processor._message_cache, "Messages should be cached"

        _, cached_output = processor._message_cache[span_id]

        # Verify correct extraction
        assert len(cached_output) == 1, "Should have 1 output message"
        assert isinstance(cached_output[0], OutputMessage), (
            "Should be OutputMessage"
        )
        assert cached_output[0].role == "assistant", (
            "Should map AIMessage to assistant"
        )
        assert len(cached_output[0].parts) == 1, "Should have 1 part"
        assert isinstance(cached_output[0].parts[0], Text), (
            "Part should be Text"
        )
        assert (
            cached_output[0].parts[0].content
            == "Hi Lance! Nice to meet you.\n"
        ), "Should extract content from kwargs"
        assert cached_output[0].finish_reason == "stop", (
            "Should normalize finish_reason to lowercase"
        )

    def test_real_full_conversation(self, setup_tracer):
        """Test with complete conversation including input and output."""
        tracer, exporter, provider, processor = setup_tracer

        # Real input
        input_data = {
            "inputs": {
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "schema",
                            "messages",
                            "HumanMessage",
                        ],
                        "kwargs": {
                            "content": "What is the capital of France?",
                            "type": "human",
                        },
                    }
                ]
            }
        }

        # Real output
        output_data = {
            "outputs": {
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "schema",
                            "messages",
                            "HumanMessage",
                        ],
                        "kwargs": {
                            "content": "What is the capital of France?",
                            "type": "human",
                            "id": "user-msg-123",
                        },
                    },
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "schema", "messages", "AIMessage"],
                        "kwargs": {
                            "content": "The capital of France is Paris.",
                            "type": "ai",
                            "id": "ai-msg-456",
                            "response_metadata": {"finish_reason": "STOP"},
                        },
                    },
                ]
            }
        }

        # Create span
        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute(
                "traceloop.entity.input", json.dumps(input_data)
            )
            span.set_attribute(
                "traceloop.entity.output", json.dumps(output_data)
            )
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Verify cache
        assert span_id in processor._message_cache
        cached_input, cached_output = processor._message_cache[span_id]

        # Verify input
        assert len(cached_input) == 1
        assert (
            cached_input[0].parts[0].content
            == "What is the capital of France?"
        )

        # Verify output (should have 2 messages: echoed input + AI response)
        assert len(cached_output) == 2
        assert (
            cached_output[0].parts[0].content
            == "What is the capital of France?"
        )
        assert (
            cached_output[1].parts[0].content
            == "The capital of France is Paris."
        )

    def test_deepeval_extraction_with_real_format(self, setup_tracer):
        """Test that DeepEval can extract text from real Traceloop format."""
        tracer, exporter, provider, processor = setup_tracer

        # Real format
        input_data = {
            "inputs": {
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "schema",
                            "messages",
                            "HumanMessage",
                        ],
                        "kwargs": {
                            "content": "Test DeepEval extraction",
                            "type": "human",
                        },
                    }
                ]
            }
        }

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute(
                "traceloop.entity.input", json.dumps(input_data)
            )
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Get cached messages
        cached_input, _ = processor._message_cache[span_id]

        # Simulate DeepEval's extract_text_from_messages
        def extract_text_from_messages(messages):
            chunks = []
            for message in messages or []:
                parts = getattr(message, "parts", [])
                for part in parts:
                    if hasattr(part, "content"):
                        if part.content:
                            chunks.append(part.content)
            return "\n".join(c for c in chunks if c).strip()

        # Test extraction
        extracted_text = extract_text_from_messages(cached_input)
        assert extracted_text == "Test DeepEval extraction", (
            "DeepEval should extract text correctly from real format"
        )

    def test_multiple_messages_in_real_format(self, setup_tracer):
        """Test with multiple messages in real Traceloop format."""
        tracer, exporter, provider, processor = setup_tracer

        # Multiple messages in one conversation
        input_data = {
            "inputs": {
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "schema",
                            "messages",
                            "SystemMessage",
                        ],
                        "kwargs": {
                            "content": "You are a helpful assistant.",
                            "type": "system",
                        },
                    },
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "schema",
                            "messages",
                            "HumanMessage",
                        ],
                        "kwargs": {"content": "Hello!", "type": "human"},
                    },
                ]
            }
        }

        with tracer.start_as_current_span("openai.chat") as span:
            span.set_attribute(
                "traceloop.entity.input", json.dumps(input_data)
            )
            span.set_attribute("llm.request.model", "gpt-5-nano")
            span_id = span.get_span_context().span_id

        provider.force_flush()

        # Verify both messages cached
        cached_input, _ = processor._message_cache[span_id]
        assert len(cached_input) == 2, "Should have 2 input messages"

        # Verify system message
        assert cached_input[0].role == "system"
        assert (
            cached_input[0].parts[0].content == "You are a helpful assistant."
        )

        # Verify human message
        assert cached_input[1].role == "user"
        assert cached_input[1].parts[0].content == "Hello!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
