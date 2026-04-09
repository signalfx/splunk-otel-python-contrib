# Copyright Splunk Inc.
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

"""Utility functions for Strands instrumentation."""

import json
from typing import Any, Optional

from opentelemetry.util.genai.types import InputMessage, Text


def safe_json_dumps(obj: Any) -> Optional[str]:
    """Safely serialize an object to JSON string.

    Args:
        obj: Object to serialize

    Returns:
        JSON string or None if serialization fails
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return None


def safe_str(value: Any) -> str:
    """Safely convert a value to string.

    Args:
        value: Value to convert

    Returns:
        String representation of value
    """
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def convert_strands_messages(messages: Any) -> list[InputMessage]:
    """Convert Strands message format to GenAI InputMessage format.

    Strands messages can be in various formats:
    - List of dicts with "role" and "content" keys
    - List of message objects with role and content attributes
    - String (treated as single user message)

    Args:
        messages: Strands messages in any supported format

    Returns:
        List of InputMessage objects
    """
    if not messages:
        return []

    result: list[InputMessage] = []

    # Handle string input (single user message)
    if isinstance(messages, str):
        return [InputMessage(role="user", parts=[Text(content=messages)])]

    # Handle list of messages
    if isinstance(messages, list):
        for msg in messages:
            try:
                # Handle dict format
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    result.append(
                        InputMessage(role=role, parts=[Text(content=safe_str(content))])
                    )
                # Handle object with attributes
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    role = getattr(msg, "role", "user")
                    content = getattr(msg, "content", "")
                    result.append(
                        InputMessage(role=role, parts=[Text(content=safe_str(content))])
                    )
                # Handle string in list
                elif isinstance(msg, str):
                    result.append(
                        InputMessage(role="user", parts=[Text(content=msg)])
                    )
            except Exception:
                # Skip malformed messages
                continue

    return result


def extract_model_id(agent: Any) -> Optional[str]:
    """Extract model ID from Strands agent.

    Strands agents may store model configuration in various attributes:
    - agent.model (string)
    - agent.model_id (string)
    - agent.model.model_id (nested)
    - agent.config.model (config object)

    Args:
        agent: Strands Agent instance

    Returns:
        Model ID string or None if not found
    """
    if not agent:
        return None

    try:
        # Direct model attribute (most common)
        if hasattr(agent, "model"):
            model = getattr(agent, "model", None)
            if isinstance(model, str):
                return model
            # Model object with model_id
            if hasattr(model, "model_id"):
                return getattr(model, "model_id", None)

        # Direct model_id attribute
        if hasattr(agent, "model_id"):
            model_id = getattr(agent, "model_id", None)
            if isinstance(model_id, str):
                return model_id

        # Config object
        if hasattr(agent, "config"):
            config = getattr(agent, "config", None)
            if config and hasattr(config, "model"):
                model = getattr(config, "model", None)
                if isinstance(model, str):
                    return model

    except Exception:
        pass

    return None


def extract_tools_list(agent: Any) -> list[str]:
    """Extract list of tool names from Strands agent.

    Args:
        agent: Strands Agent instance

    Returns:
        List of tool names
    """
    if not agent:
        return []

    try:
        if hasattr(agent, "tools"):
            tools = getattr(agent, "tools", None)
            if not tools:
                return []

            tool_names = []
            for tool in tools:
                # Tool object with name attribute
                if hasattr(tool, "name"):
                    name = getattr(tool, "name", None)
                    if name:
                        tool_names.append(safe_str(name))
                # Tool dict with name key
                elif isinstance(tool, dict) and "name" in tool:
                    tool_names.append(safe_str(tool["name"]))
                # Tool string
                elif isinstance(tool, str):
                    tool_names.append(tool)

            return tool_names
    except Exception:
        pass

    return []


def extract_agent_name(agent: Any) -> Optional[str]:
    """Extract name from Strands agent.

    Args:
        agent: Strands Agent instance

    Returns:
        Agent name or None if not found
    """
    if not agent:
        return None

    try:
        # Check various possible name attributes
        for attr in ("name", "agent_name", "id", "agent_id"):
            if hasattr(agent, attr):
                value = getattr(agent, attr, None)
                if value:
                    return safe_str(value)
    except Exception:
        pass

    return None
