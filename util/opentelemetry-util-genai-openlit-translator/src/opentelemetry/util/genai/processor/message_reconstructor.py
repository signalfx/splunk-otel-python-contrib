"""
Reconstruct LangChain message objects from openlit serialized data.

This module enables evaluations to work with Openlit SDK alone,
without requiring LangChain instrumentation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .content_normalizer import normalize_openlit_content

_logger = logging.getLogger(__name__)


def reconstruct_messages_from_openlit(
    input_data: Any, output_data: Any
) -> tuple[Optional[List[Any]], Optional[List[Any]]]:
    """
    Reconstruct LangChain message objects from openlit serialized data.

    Args:
        input_data: Raw openlit.entity.input value (string or dict)
        output_data: Raw openlit.entity.output value (string or dict)

    Returns:
        Tuple of (input_messages, output_messages) as LangChain BaseMessage lists,
        or (None, None) if reconstruction fails or LangChain is not available.

    This function:
    1. Parses the JSON-serialized openlit data
    2. Normalizes it to standard message format
    3. Reconstructs LangChain BaseMessage objects (HumanMessage, AIMessage, etc.)
    4. Returns them for use in evaluations

    If LangChain is not installed, returns (None, None) gracefully.
    """
    try:
        # Import LangChain message classes (optional dependency)
        try:
            from langchain_core.messages import (  # noqa: I001
                AIMessage,  # noqa: F401
                BaseMessage,  # noqa: F401
                FunctionMessage,  # noqa: F401
                HumanMessage,  # noqa: F401
                SystemMessage,  # noqa: F401
                ToolMessage,  # noqa: F401
            )
        except ImportError:
            _logger.debug(
                "LangChain not available; message reconstruction skipped. "
                "Install langchain-core to enable evaluations with openlit."
            )
            return None, None

        input_messages = None
        output_messages = None

        # Reconstruct input messages
        if input_data:
            try:
                # Normalize the openlit data to standard format
                normalized_input = normalize_openlit_content(
                    input_data, "input"
                )
                input_messages = _convert_normalized_to_langchain(
                    normalized_input, "input"
                )
                _logger.debug(
                    f"Reconstructed {len(input_messages)} input messages from openlit data"
                )
            except Exception as e:
                _logger.debug(f"Failed to reconstruct input messages: {e}")

        # Reconstruct output messages
        if output_data:
            try:
                # Normalize the openlit data to standard format
                normalized_output = normalize_openlit_content(
                    output_data, "output"
                )
                output_messages = _convert_normalized_to_langchain(
                    normalized_output, "output"
                )
                _logger.debug(
                    f"Reconstructed {len(output_messages)} output messages from openlit data"
                )
            except Exception as e:
                _logger.debug(f"Failed to reconstruct output messages: {e}")

        return input_messages, output_messages

    except Exception as e:
        _logger.debug(f"Message reconstruction failed: {e}")
        return None, None


def _convert_normalized_to_langchain(
    normalized_messages: List[Dict[str, Any]], direction: str
) -> List[Any]:
    """
    Convert normalized message format to LangChain BaseMessage objects.

    Args:
        normalized_messages: List of normalized messages from normalize_openlit_content
        direction: 'input' or 'output' (for logging/debugging)

    Returns:
        List of LangChain BaseMessage objects

    Normalized message format:
        {
            "role": "user" | "assistant" | "system" | "tool" | "function",
            "parts": [{"type": "text", "content": "..."}, ...],
            "finish_reason": "stop"  # optional, for output messages
        }
    """
    from langchain_core.messages import (  # noqa: I001
        AIMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    langchain_messages = []

    for msg in normalized_messages:
        role = msg.get("role", "user" if direction == "input" else "assistant")
        parts = msg.get("parts", [])

        # Extract content from parts (typically just text parts)
        content_parts = []
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    content_parts.append(part.get("content", ""))
                elif part.get("type") == "tool_call":
                    # For tool calls, keep the structured data
                    content_parts.append(json.dumps(part))
                else:
                    # Unknown part type, serialize it
                    content_parts.append(json.dumps(part))
            else:
                # Non-dict part, stringify it
                content_parts.append(str(part))

        # Join all content parts
        content = "\n".join(content_parts) if content_parts else ""

        # Map role to LangChain message class
        if role == "user":
            langchain_msg = HumanMessage(content=content)
        elif role == "assistant":
            # Include finish_reason in additional_kwargs if present
            additional_kwargs = {}
            if "finish_reason" in msg:
                additional_kwargs["finish_reason"] = msg["finish_reason"]
            langchain_msg = AIMessage(
                content=content,
                additional_kwargs=additional_kwargs
                if additional_kwargs
                else {},
            )
        elif role == "system":
            langchain_msg = SystemMessage(content=content)
        elif role == "tool":
            langchain_msg = ToolMessage(
                content=content,
                tool_call_id=msg.get("tool_call_id", "unknown"),
            )
        elif role == "function":
            langchain_msg = FunctionMessage(
                content=content, name=msg.get("name", "unknown")
            )
        else:
            # Unknown role, default to HumanMessage
            _logger.debug(f"Unknown role '{role}', defaulting to HumanMessage")
            langchain_msg = HumanMessage(content=content)

        # CRITICAL FIX: Add .parts attribute for GenAI evaluation compatibility
        # GenAI evaluations expect message.parts (list of Text/ToolCall objects)
        # but LangChain messages only have .content (str)
        # We add .parts here to bridge the gap without requiring LangChain instrumentation
        try:
            # Import Text from GenAI types
            from opentelemetry.util.genai.types import Text

            text_part = Text(content=content, type="text")
            langchain_msg.parts = [text_part]  # type: ignore[attr-defined]

            _logger.debug(
                f"Added .parts attribute to {type(langchain_msg).__name__} "
                f"for evaluation compatibility"
            )
        except ImportError:
            # GenAI types not available, evaluations won't work but won't crash
            _logger.debug(
                "GenAI types not available; .parts attribute not added. "
                "Evaluations will not work."
            )
        except Exception as e:
            # Unexpected error, log but don't crash
            _logger.debug(f"Failed to add .parts attribute: {e}")

        langchain_messages.append(langchain_msg)

    return langchain_messages


__all__ = [
    "reconstruct_messages_from_openlit",
]
