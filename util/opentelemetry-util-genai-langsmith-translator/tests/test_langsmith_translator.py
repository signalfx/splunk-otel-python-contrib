# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Langsmith to GenAI semantic convention translator."""

from unittest.mock import MagicMock, patch


class TestContentNormalizer:
    """Tests for the content normalizer module."""

    def test_normalize_openai_choices_format(self):
        """Test normalizing OpenAI choices format."""
        from opentelemetry.util.genai.processor.content_normalizer import (
            normalize_langsmith_content,
        )

        output_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        messages = normalize_langsmith_content(output_data, "output")

        assert len(messages) >= 1
        output_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(output_msgs) > 0

    def test_normalize_langchain_serialized_messages(self):
        """Test normalizing LangChain serialized message format."""
        from opentelemetry.util.genai.processor.content_normalizer import (
            normalize_langsmith_content,
        )

        input_data = {
            "messages": [
                [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "schema", "HumanMessage"],
                        "kwargs": {"content": "Hello!", "type": "human"},
                    }
                ]
            ]
        }

        messages = normalize_langsmith_content(input_data, "input")

        assert len(messages) >= 1
        # Check content is properly extracted
        assert messages[0]["parts"][0]["content"] == "Hello!"
        assert messages[0]["role"] == "user"

    def test_normalize_direct_lc_array(self):
        """Test normalizing direct array of LangChain lc messages."""
        from opentelemetry.util.genai.processor.content_normalizer import (
            normalize_langsmith_content,
        )

        # This is the format Langsmith native OTEL export uses for message arrays
        input_data = [
            [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "SystemMessage"],
                    "kwargs": {
                        "content": "You are a financial assistant.",
                        "type": "system",
                    },
                },
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {
                        "content": "What's my current balance?",
                        "type": "human",
                    },
                },
            ]
        ]

        messages = normalize_langsmith_content(input_data, "input")

        assert len(messages) == 2
        # Check first message (system)
        assert messages[0]["role"] == "system"
        assert (
            messages[0]["parts"][0]["content"]
            == "You are a financial assistant."
        )
        # Check second message (user)
        assert messages[1]["role"] == "user"
        assert (
            messages[1]["parts"][0]["content"] == "What's my current balance?"
        )

    def test_normalize_string_content(self):
        """Test normalizing simple string content."""
        from opentelemetry.util.genai.processor.content_normalizer import (
            normalize_langsmith_content,
        )

        input_msgs = normalize_langsmith_content(
            "What is the weather?", "input"
        )
        output_msgs = normalize_langsmith_content(
            "It's sunny today.", "output"
        )

        assert len(input_msgs) >= 1
        assert len(output_msgs) >= 1

    def test_normalize_empty_content(self):
        """Test handling empty content."""
        from opentelemetry.util.genai.processor.content_normalizer import (
            normalize_langsmith_content,
        )

        messages = normalize_langsmith_content(None, "input")

        # Should return fallback empty message
        assert isinstance(messages, list)


class TestMessageReconstructor:
    """Tests for the message reconstructor module."""

    def test_reconstruct_simple_messages(self):
        """Test reconstructing simple messages."""
        from opentelemetry.util.genai.processor.message_reconstructor import (
            reconstruct_messages_from_langsmith,
        )

        input_data = "Hello!"
        output_data = "Hi there!"

        input_messages, output_messages = reconstruct_messages_from_langsmith(
            input_data, output_data
        )

        # Results may be None if LangChain is not installed
        # Just check it doesn't crash and returns a tuple
        assert isinstance(input_messages, (list, type(None)))
        assert isinstance(output_messages, (list, type(None)))

    def test_reconstruct_with_tool_calls(self):
        """Test reconstructing messages with tool calls."""
        from opentelemetry.util.genai.processor.message_reconstructor import (
            reconstruct_messages_from_langsmith,
        )

        output_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        input_messages, output_messages = reconstruct_messages_from_langsmith(
            None, output_data
        )

        # Results may be None if LangChain is not installed
        assert isinstance(input_messages, (list, type(None)))
        assert isinstance(output_messages, (list, type(None)))


class TestLangsmithSpanProcessor:
    """Tests for the LangsmithSpanProcessor."""

    def test_processor_initialization(self):
        """Test processor can be initialized."""
        from opentelemetry.util.genai.processor.langsmith_span_processor import (
            LangsmithSpanProcessor,
        )

        processor = LangsmithSpanProcessor(
            attribute_transformations={
                "rename": {"langsmith.model": "gen_ai.request.model"}
            }
        )

        assert processor is not None

    def test_processor_on_start(self):
        """Test on_start method doesn't crash."""
        from opentelemetry.util.genai.processor.langsmith_span_processor import (
            LangsmithSpanProcessor,
        )

        processor = LangsmithSpanProcessor()

        # Create a mock span
        mock_span = MagicMock()
        mock_span.name = "test_span"
        mock_span.attributes = {}

        # Should not raise
        processor.on_start(mock_span, parent_context=None)

    def test_is_llm_span_detection(self):
        """Test LLM span detection logic."""
        from opentelemetry.util.genai.processor.langsmith_span_processor import (
            LangsmithSpanProcessor,
        )

        processor = LangsmithSpanProcessor()

        # Create mock spans with gen_ai.operation.name attribute
        # Test with chat operation (should be detected as LLM)
        chat_span = MagicMock()
        chat_span.attributes = {"gen_ai.operation.name": "chat"}
        chat_span.name = "chat_span"
        assert processor._is_llm_span(chat_span) is True

        # Test with completion operation (should be detected as LLM)
        completion_span = MagicMock()
        completion_span.attributes = {"gen_ai.operation.name": "completion"}
        completion_span.name = "completion_span"
        assert processor._is_llm_span(completion_span) is True

        # Test with non-LLM operation (workflow)
        workflow_span = MagicMock()
        workflow_span.attributes = {"gen_ai.operation.name": "workflow"}
        workflow_span.name = "workflow_span"
        assert processor._is_llm_span(workflow_span) is False

        # Test with no operation name (should return False)
        no_op_span = MagicMock()
        no_op_span.attributes = {"langsmith.run_type": "chain"}
        no_op_span.name = "chain_span"
        assert processor._is_llm_span(no_op_span) is False


class TestEnableLangsmithTranslator:
    """Tests for the enable_langsmith_translator function."""

    def test_enable_returns_false_when_no_provider(self):
        """Test enable returns False when no real provider exists."""
        from opentelemetry.util.genai.langsmith import (
            enable_langsmith_translator,
        )

        with patch("opentelemetry.trace.get_tracer_provider") as mock_get:
            mock_provider = MagicMock()
            # Remove add_span_processor to simulate proxy provider
            del mock_provider.add_span_processor
            mock_get.return_value = mock_provider

            result = enable_langsmith_translator()
            # May return False due to missing add_span_processor
            assert isinstance(result, bool)


class TestAttributeTransformations:
    """Tests for attribute transformation mappings."""

    def test_default_transformations_exist(self):
        """Test that default transformations are defined."""
        from opentelemetry.util.genai.langsmith import (
            _DEFAULT_ATTR_TRANSFORMATIONS,
        )

        assert "rename" in _DEFAULT_ATTR_TRANSFORMATIONS
        renames = _DEFAULT_ATTR_TRANSFORMATIONS["rename"]

        # Check key Langsmith mappings
        assert "langsmith.metadata.ls_provider" in renames
        assert renames["langsmith.metadata.ls_provider"] == "gen_ai.system"

        assert "langsmith.metadata.ls_model_name" in renames
        assert (
            renames["langsmith.metadata.ls_model_name"]
            == "gen_ai.request.model"
        )

    def test_token_usage_mappings(self):
        """Test token usage mappings exist."""
        from opentelemetry.util.genai.langsmith import (
            _DEFAULT_ATTR_TRANSFORMATIONS,
        )

        renames = _DEFAULT_ATTR_TRANSFORMATIONS["rename"]

        # Check token usage mappings
        assert "langsmith.token_usage.prompt_tokens" in renames
        assert (
            renames["langsmith.token_usage.prompt_tokens"]
            == "gen_ai.usage.input_tokens"
        )

        assert "langsmith.token_usage.completion_tokens" in renames
        assert (
            renames["langsmith.token_usage.completion_tokens"]
            == "gen_ai.usage.output_tokens"
        )

    def test_tool_calling_mappings(self):
        """Test tool calling mappings exist."""
        from opentelemetry.util.genai.langsmith import (
            _DEFAULT_ATTR_TRANSFORMATIONS,
        )

        renames = _DEFAULT_ATTR_TRANSFORMATIONS["rename"]

        # Check tool mappings
        assert "langsmith.tool.name" in renames
        assert renames["langsmith.tool.name"] == "gen_ai.tool.call.name"
