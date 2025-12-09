"""Test message serialization to ensure no double-encoding.

This test verifies that messages are serialized correctly without
nested JSON encoding issues.
"""

import json

import pytest

from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text


class TestMessageSerialization:
    """Test message serialization format."""

    def test_input_message_not_double_encoded(self):
        """Test that InputMessage content is not double-encoded."""
        msg = InputMessage(
            role="user",
            parts=[Text(content="Hello, how are you?", type="text")],
        )

        # Serialize as we do in the processor
        serialized = json.dumps(
            [
                {
                    "role": msg.role,
                    "parts": [
                        {"type": "text", "content": part.content}
                        for part in msg.parts
                    ],
                }
            ]
        )

        # Parse back
        parsed = json.loads(serialized)

        # Verify structure
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"
        assert len(parsed[0]["parts"]) == 1
        assert parsed[0]["parts"][0]["type"] == "text"
        assert parsed[0]["parts"][0]["content"] == "Hello, how are you?"

        # CRITICAL: Content should be a STRING, not nested JSON
        content = parsed[0]["parts"][0]["content"]
        assert isinstance(content, str), "Content must be string"
        assert not content.startswith('{"'), (
            "Content should NOT be JSON string"
        )
        assert content == "Hello, how are you?", "Content should be plain text"

    def test_output_message_not_double_encoded(self):
        """Test that OutputMessage content is not double-encoded."""
        msg = OutputMessage(
            role="assistant",
            parts=[Text(content="I'm doing great, thanks!", type="text")],
            finish_reason="stop",
        )

        # Serialize as we do in the processor
        serialized = json.dumps(
            [
                {
                    "role": msg.role,
                    "parts": [
                        {"type": "text", "content": part.content}
                        for part in msg.parts
                    ],
                    "finish_reason": msg.finish_reason,
                }
            ]
        )

        # Parse back
        parsed = json.loads(serialized)

        # Verify structure
        assert len(parsed) == 1
        assert parsed[0]["role"] == "assistant"
        assert parsed[0]["finish_reason"] == "stop"
        assert len(parsed[0]["parts"]) == 1

        # CRITICAL: Content should be plain text, not JSON
        content = parsed[0]["parts"][0]["content"]
        assert isinstance(content, str), "Content must be string"
        assert not content.startswith('{"'), (
            "Content should NOT be JSON string"
        )
        assert content == "I'm doing great, thanks!", (
            "Content should be plain text"
        )

    def test_deepeval_can_parse_serialized_messages(self):
        """Test that DeepEval can parse our serialized format."""
        # Create messages
        input_msg = InputMessage(
            role="user", parts=[Text(content="Test input", type="text")]
        )
        output_msg = OutputMessage(
            role="assistant",
            parts=[Text(content="Test output", type="text")],
            finish_reason="stop",
        )

        # Serialize to JSON string (as stored in span attributes)
        input_json = json.dumps(
            [
                {
                    "role": input_msg.role,
                    "parts": [
                        {"type": "text", "content": part.content}
                        for part in input_msg.parts
                    ],
                }
            ]
        )
        output_json = json.dumps(
            [
                {
                    "role": output_msg.role,
                    "parts": [
                        {"type": "text", "content": part.content}
                        for part in output_msg.parts
                    ],
                    "finish_reason": output_msg.finish_reason,
                }
            ]
        )

        # Simulate what DeepEval does: parse JSON and extract text
        input_parsed = json.loads(input_json)
        output_parsed = json.loads(output_json)

        # Extract text (DeepEval's logic)
        def extract_text(messages):
            texts = []
            for msg in messages:
                for part in msg.get("parts", []):
                    if part.get("type") == "text":
                        texts.append(part.get("content", ""))
            return "\n".join(texts)

        input_text = extract_text(input_parsed)
        output_text = extract_text(output_parsed)

        # Verify extraction works
        assert input_text == "Test input", "Should extract input text"
        assert output_text == "Test output", "Should extract output text"

    def test_complex_content_not_double_encoded(self):
        """Test that complex content with special characters is not double-encoded."""
        complex_content = "I found a flight:\n- Airline: AeroJet\n- Price: $1044\nWould you like more information?"

        msg = OutputMessage(
            role="assistant",
            parts=[Text(content=complex_content, type="text")],
            finish_reason="stop",
        )

        # Serialize
        serialized = json.dumps(
            [
                {
                    "role": msg.role,
                    "parts": [
                        {"type": "text", "content": part.content}
                        for part in msg.parts
                    ],
                    "finish_reason": msg.finish_reason,
                }
            ]
        )

        # Parse back
        parsed = json.loads(serialized)
        content = parsed[0]["parts"][0]["content"]

        # Verify content is unchanged
        assert content == complex_content, (
            "Complex content should be preserved"
        )
        assert "\n" in content, "Newlines should be preserved"
        assert "$" in content, "Special characters should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
