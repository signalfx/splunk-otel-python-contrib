"""Test handling of args wrapper format from LangGraph/openlit."""

import pytest

from opentelemetry.util.genai.processor.content_normalizer import (
    normalize_openlit_content,
)


class TestArgsWrapperFormat:
    """Test that the normalizer handles the args wrapper format."""

    def test_args_wrapper_with_messages(self):
        """Test the actual format shown in debugger."""
        # This is the EXACT format from the debugger screenshot
        input_data = {
            "args": [
                {
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
                                "content": "We're planning a romantic long-week trip to Paris from Seattle next month. We'd love a boutique hotel, business-class flights and a few unique experiences.",
                                "type": "human",
                                "id": "8bb38518-7561-40e0-9c3a-682b825ca00d",
                            },
                        }
                    ],
                    "user_request": "We're planning a romantic long-week trip to Paris from Seattle next month. We'd love a boutique hotel, business-class flights and a few unique experiences.",
                    "session_id": "f158b070-5e18-43f7-99f0-095364ed1211",
                    "origin": "Seattle",
                    "destination": "Paris",
                    "departure": "2025-12-07",
                    "return_date": "2025-12-14",
                    "travellers": 2,
                    "flight_summary": None,
                    "hotel_summary": None,
                    "activities_summary": None,
                    "final_itinerary": None,
                    "current_agent": "start",
                }
            ],
            "kwargs": {},
        }

        # Normalize
        result = normalize_openlit_content(input_data, "input")

        # Verify
        assert len(result) == 1, f"Should have 1 message, got {len(result)}"

        message = result[0]
        assert message["role"] == "user", (
            f"Role should be 'user', got {message['role']}"
        )
        assert len(message["parts"]) == 1, (
            f"Should have 1 part, got {len(message['parts'])}"
        )

        part = message["parts"][0]
        assert part["type"] == "text", (
            f"Part type should be 'text', got {part['type']}"
        )
        assert "Paris" in part["content"], "Content should mention Paris"
        assert "Seattle" in part["content"], "Content should mention Seattle"
        assert "boutique hotel" in part["content"], (
            "Content should mention boutique hotel"
        )

    def test_args_wrapper_with_multiple_messages(self):
        """Test args wrapper with conversation history."""
        input_data = {
            "args": [
                {
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
            ],
            "kwargs": {},
        }

        result = normalize_openlit_content(input_data, "input")

        assert len(result) == 2, f"Should have 2 messages, got {len(result)}"

        # System message
        assert result[0]["role"] == "system"
        assert (
            result[0]["parts"][0]["content"] == "You are a helpful assistant."
        )

        # Human message
        assert result[1]["role"] == "user"
        assert result[1]["parts"][0]["content"] == "Hello!"

    def test_args_wrapper_empty_messages(self):
        """Test args wrapper with empty messages array."""
        input_data = {"args": [{"messages": []}], "kwargs": {}}

        result = normalize_openlit_content(input_data, "input")

        assert result == [], "Should return empty list for empty messages"

    def test_args_wrapper_output_format(self):
        """Test args wrapper for output (response) format."""
        output_data = {
            "args": [
                {
                    "messages": [
                        {
                            "lc": 1,
                            "type": "constructor",
                            "id": [
                                "langchain",
                                "schema",
                                "messages",
                                "AIMessage",
                            ],
                            "kwargs": {
                                "content": "I can help you plan your trip to Paris!",
                                "type": "ai",
                                "response_metadata": {"finish_reason": "stop"},
                            },
                        }
                    ]
                }
            ],
            "kwargs": {},
        }

        result = normalize_openlit_content(output_data, "output")

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "Paris" in result[0]["parts"][0]["content"]
        assert result[0]["finish_reason"] == "stop"

    def test_nested_inputs_still_works(self):
        """Ensure the old nested inputs format still works."""
        # Old format with "inputs" wrapper
        old_format = {
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
                        "kwargs": {"content": "Test message", "type": "human"},
                    }
                ]
            }
        }

        result = normalize_openlit_content(old_format, "input")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Test message"

    def test_direct_messages_still_works(self):
        """Ensure direct messages format still works."""
        # Direct format (no wrapper)
        direct_format = {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": "Direct message", "type": "human"},
                }
            ]
        }

        result = normalize_openlit_content(direct_format, "input")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Direct message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
