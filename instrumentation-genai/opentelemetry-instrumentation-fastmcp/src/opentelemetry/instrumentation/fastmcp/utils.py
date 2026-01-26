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

"""Utility functions for FastMCP instrumentation."""

import functools
import json
import logging
import os
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)


def dont_throw(func):
    """Decorator that catches and logs exceptions without re-raising.

    Used to ensure instrumentation doesn't break the instrumented application.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _LOGGER.debug(
                f"FastMCP instrumentation error in {func.__name__}: {e}",
                exc_info=True,
            )
            return None

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _LOGGER.debug(
                f"FastMCP instrumentation error in {func.__name__}: {e}",
                exc_info=True,
            )
            return None

    if asyncio_iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def asyncio_iscoroutinefunction(func) -> bool:
    """Check if a function is a coroutine function."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


def safe_serialize(obj: Any, max_depth: int = 4) -> Optional[str]:
    """Safely serialize an object to JSON string.

    Args:
        obj: Object to serialize
        max_depth: Maximum nesting depth

    Returns:
        JSON string or None if serialization fails
    """
    if obj is None:
        return None

    try:

        def _serialize(o, depth=0):
            if depth > max_depth:
                return "<max_depth_exceeded>"

            if isinstance(o, (str, int, float, bool, type(None))):
                return o
            elif isinstance(o, dict):
                return {k: _serialize(v, depth + 1) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [_serialize(item, depth + 1) for item in o]
            elif hasattr(o, "__dict__"):
                return {
                    k: _serialize(v, depth + 1)
                    for k, v in o.__dict__.items()
                    if not k.startswith("_")
                }
            elif hasattr(o, "text"):
                return {"type": "text", "content": o.text}
            else:
                return str(o)

        return json.dumps(_serialize(obj), ensure_ascii=False)
    except (TypeError, ValueError) as e:
        _LOGGER.debug(f"Serialization error: {e}")
        return None


def truncate_if_needed(value: str, max_length: Optional[int] = None) -> str:
    """Truncate a string if it exceeds the OTEL attribute length limit.

    Args:
        value: String to potentially truncate
        max_length: Optional override for max length

    Returns:
        Original or truncated string
    """
    if max_length is None:
        limit_str = os.getenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT")
        if limit_str:
            try:
                max_length = int(limit_str)
            except ValueError:
                return value

    if max_length and max_length > 0 and len(value) > max_length:
        return value[:max_length]
    return value


def should_capture_content() -> bool:
    """Check if content capture is enabled via environment variables."""
    env_value = os.getenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", ""
    ).lower()
    return env_value in ("true", "1", "yes", "on")


def is_instrumentation_enabled() -> bool:
    """Check if GenAI instrumentation is enabled."""
    env_value = os.getenv("OTEL_INSTRUMENTATION_GENAI_ENABLE", "true").lower()
    return env_value in ("true", "1", "yes", "on")


def extract_tool_info(args: tuple, kwargs: dict) -> tuple[str, Any]:
    """Extract tool name and arguments from call_tool parameters.

    FastMCP's ToolManager.call_tool can be called with different patterns:
    - call_tool(key="tool_name", arguments={...})
    - call_tool("tool_name", {...})

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tuple of (tool_name, arguments)
    """
    tool_name = "unknown_tool"
    tool_arguments = {}

    # Pattern 1: kwargs with 'key' parameter
    if kwargs and "key" in kwargs:
        tool_name = kwargs.get("key", "unknown_tool")
        tool_arguments = kwargs.get("arguments", {})
    # Pattern 2: positional args (tool_name, arguments)
    elif args:
        tool_name = args[0] if args else "unknown_tool"
        tool_arguments = args[1] if len(args) > 1 else {}

    return str(tool_name), tool_arguments


def extract_result_content(result: Any) -> Optional[str]:
    """Extract content from a FastMCP tool result.

    Args:
        result: Tool execution result

    Returns:
        Extracted content string or None
    """
    if result is None:
        return None

    try:
        output_data = []
        # FastMCP 2.12.2+ uses result.content
        result_items = result.content if hasattr(result, "content") else result

        # Handle non-iterable or string results directly
        if isinstance(result_items, str):
            return result_items
        if not hasattr(result_items, "__iter__"):
            return str(result_items)

        for item in result_items:
            if hasattr(item, "text"):
                output_data.append({"type": "text", "content": item.text})
            elif hasattr(item, "__dict__"):
                output_data.append(
                    {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                )
            else:
                output_data.append(str(item))

        return json.dumps(output_data, ensure_ascii=False)
    except Exception as e:
        _LOGGER.debug(f"Error extracting result content: {e}")
        return str(result) if result else None
