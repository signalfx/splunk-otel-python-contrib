"""Tests for openlit_content_normalizer — new helpers added in HYBIM-598.

Covers:
- _extract_langchain_messages (envelope unwrapping, LC constructor parsing)
- normalize_openlit_content with LC constructor objects directly in list
- normalize_openlit_content with "parts" array containing nested LC messages
- normalize_openlit_content with "outputs" envelope unwrapping
"""

from __future__ import annotations

import json

from opentelemetry.util.genai.processor.openlit_content_normalizer import (
    normalize_openlit_content,
)

# ---------------------------------------------------------------------------
# _extract_langchain_messages (tested indirectly via normalize_openlit_content)
# ---------------------------------------------------------------------------


class TestExtractLangchainMessages:
    """Test LangChain message extraction from JSON envelopes."""

    def test_messages_envelope(self):
        """Messages wrapped in {"messages": [...]} are extracted."""
        raw = json.dumps(
            {
                "messages": [
                    {
                        "lc": 1,
                        "kwargs": {"content": "Hello", "type": "human"},
                    },
                    {
                        "lc": 1,
                        "kwargs": {"content": "Hi there", "type": "ai"},
                    },
                ]
            }
        )
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["parts"][0]["content"] == "Hi there"

    def test_outputs_envelope(self):
        """Messages wrapped in {"outputs": {"messages": [...]}} are extracted."""
        raw = {
            "outputs": {
                "messages": [
                    {
                        "lc": 1,
                        "kwargs": {"content": "Done", "type": "ai"},
                    }
                ]
            }
        }
        result = normalize_openlit_content(raw, "output")
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["parts"][0]["content"] == "Done"
        assert "finish_reason" in result[0]

    def test_inputs_envelope(self):
        """Messages wrapped in {"inputs": {"messages": [...]}} are extracted."""
        raw = {
            "inputs": {
                "messages": [
                    {
                        "lc": 1,
                        "kwargs": {"content": "Question?", "type": "human"},
                    }
                ]
            }
        }
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Question?"

    def test_args_envelope(self):
        """Messages wrapped in {"args": [{"messages": [...]}]} are extracted."""
        raw = {
            "args": [
                {
                    "messages": [
                        {
                            "lc": 1,
                            "kwargs": {
                                "content": "From args",
                                "type": "human",
                            },
                        }
                    ]
                }
            ]
        }
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == "From args"

    def test_non_json_string_returns_text(self):
        """Plain text string is returned as a text part, not parsed as LC."""
        result = normalize_openlit_content("just plain text", "input")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "just plain text"


# ---------------------------------------------------------------------------
# LC constructor objects directly in list
# ---------------------------------------------------------------------------


class TestLangchainConstructorInList:
    """Test handling LangChain constructor objects directly in message list."""

    def test_human_message_constructor(self):
        """LC constructor with type=human maps to role=user."""
        raw = [
            {"lc": 1, "kwargs": {"content": "Hello world", "type": "human"}}
        ]
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Hello world"

    def test_ai_message_constructor(self):
        """LC constructor with type=ai maps to role=assistant."""
        raw = [{"lc": 1, "kwargs": {"content": "Response", "type": "ai"}}]
        result = normalize_openlit_content(raw, "output")
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["finish_reason"] == "stop"

    def test_system_message_constructor(self):
        """LC constructor with type=system maps to role=system."""
        raw = [
            {
                "lc": 1,
                "kwargs": {"content": "You are helpful", "type": "system"},
            }
        ]
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_ai_with_response_metadata_finish_reason(self):
        """finish_reason is extracted from response_metadata."""
        raw = [
            {
                "lc": 1,
                "kwargs": {
                    "content": "Done",
                    "type": "ai",
                    "response_metadata": {"finish_reason": "length"},
                },
            }
        ]
        result = normalize_openlit_content(raw, "output")
        assert result[0]["finish_reason"] == "length"

    def test_mixed_constructors_and_plain_dicts(self):
        """Mix of LC constructors and plain message dicts."""
        raw = [
            {"lc": 1, "kwargs": {"content": "Hi", "type": "human"}},
            {"role": "assistant", "content": "Hello"},
        ]
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Hi"
        assert result[1]["role"] == "assistant"
        assert result[1]["parts"][0]["content"] == "Hello"

    def test_constructor_with_none_content_skipped(self):
        """LC constructor with content=None is skipped."""
        raw = [
            {"lc": 1, "kwargs": {"content": None, "type": "human"}},
            {"lc": 1, "kwargs": {"content": "Real msg", "type": "human"}},
        ]
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == "Real msg"


# ---------------------------------------------------------------------------
# "parts" array format with nested LC messages
# ---------------------------------------------------------------------------


class TestPartsArrayWithNestedLC:
    """Test handling of parts array containing nested LangChain messages."""

    def test_parts_with_lc_messages_in_json(self):
        """Parts containing JSON with LC messages are extracted."""
        inner_json = json.dumps(
            {
                "messages": [
                    {
                        "lc": 1,
                        "kwargs": {"content": "nested msg", "type": "human"},
                    }
                ]
            }
        )
        raw = [
            {
                "role": "user",
                "parts": [{"type": "text", "content": inner_json}],
            }
        ]
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "nested msg"

    def test_parts_with_plain_text_content(self):
        """Parts containing plain text are used as-is."""
        raw = [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "plain text"}],
            }
        ]
        result = normalize_openlit_content(raw, "input")
        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == "plain text"


# ---------------------------------------------------------------------------
# "outputs" envelope unwrapping in normalize_openlit_content
# ---------------------------------------------------------------------------


class TestOutputsEnvelopeUnwrapping:
    """Test that {"outputs": {...}} dicts are correctly unwrapped."""

    def test_outputs_with_messages(self):
        raw = {
            "outputs": {
                "messages": [{"role": "assistant", "content": "response text"}]
            }
        }
        result = normalize_openlit_content(raw, "output")
        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == "response text"

    def test_outputs_with_lc_messages(self):
        raw = {
            "outputs": {
                "messages": [
                    {
                        "lc": 1,
                        "kwargs": {"content": "lc response", "type": "ai"},
                    }
                ]
            }
        }
        result = normalize_openlit_content(raw, "output")
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["parts"][0]["content"] == "lc response"
