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

"""Wrapt wrappers for Bedrock AgentCore Browser instrumentation."""

import logging
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, ToolCall

from .utils import safe_json_dumps, safe_str

_LOGGER = logging.getLogger(__name__)


def wrap_browser_start(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap BrowserClient.start to create ToolCall span.

    Args:
        wrapped: Original start method
        instance: BrowserClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original start
    """
    try:
        # Extract parameters
        browser_id = kwargs.get("browser_id")

        # Create ToolCall
        tool_call = ToolCall(
            name="browser.start",
            arguments=safe_json_dumps({"browser_id": browser_id})
            if browser_id
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )

        # Add browser specific attributes
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "start_session"
        if browser_id:
            tool_call.attributes["bedrock.agentcore.browser.id"] = safe_str(browser_id)

        # Start the tool call
        handler.start_tool_call(tool_call)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Populate result data - session ID should be set after start
            if hasattr(instance, "session_id") and instance.session_id:
                tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                    instance.session_id
                )
                tool_call.tool_result = safe_json_dumps(
                    {"session_id": instance.session_id}
                )

            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_tool_call(
                tool_call, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If tool call creation failed, just call original
        return wrapped(*args, **kwargs)


def wrap_browser_stop(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap BrowserClient.stop to create ToolCall span.

    Args:
        wrapped: Original stop method
        instance: BrowserClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original stop
    """
    try:
        # Create ToolCall
        tool_call = ToolCall(
            name="browser.stop",
            system="bedrock-agentcore",
            tool_type="extension",
        )

        # Add browser specific attributes
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "stop_session"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )

        # Start the tool call
        handler.start_tool_call(tool_call)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Populate result
            if capture_content:
                tool_call.tool_result = safe_json_dumps({"success": result})

            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_tool_call(
                tool_call, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If tool call creation failed, just call original
        return wrapped(*args, **kwargs)


def wrap_browser_take_control(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap BrowserClient.take_control to create ToolCall span.

    Args:
        wrapped: Original take_control method
        instance: BrowserClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original take_control
    """
    try:
        # Create ToolCall
        tool_call = ToolCall(
            name="browser.take_control",
            system="bedrock-agentcore",
            tool_type="extension",
        )

        # Add browser specific attributes
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "take_control"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )

        # Start the tool call
        handler.start_tool_call(tool_call)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_tool_call(
                tool_call, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If tool call creation failed, just call original
        return wrapped(*args, **kwargs)


def wrap_browser_release_control(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap BrowserClient.release_control to create ToolCall span.

    Args:
        wrapped: Original release_control method
        instance: BrowserClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original release_control
    """
    try:
        # Create ToolCall
        tool_call = ToolCall(
            name="browser.release_control",
            system="bedrock-agentcore",
            tool_type="extension",
        )

        # Add browser specific attributes
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "release_control"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )

        # Start the tool call
        handler.start_tool_call(tool_call)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_tool_call(
                tool_call, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If tool call creation failed, just call original
        return wrapped(*args, **kwargs)


def wrap_browser_get_session(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap BrowserClient.get_session to create ToolCall span.

    Args:
        wrapped: Original get_session method
        instance: BrowserClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original get_session
    """
    try:
        # Extract parameters
        browser_id = kwargs.get("browser_id")
        session_id = kwargs.get("session_id")

        # Create ToolCall
        tool_call = ToolCall(
            name="browser.get_session",
            arguments=safe_json_dumps(
                {
                    "browser_id": browser_id,
                    "session_id": session_id,
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )

        # Add browser specific attributes
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "get_session"

        # Start the tool call
        handler.start_tool_call(tool_call)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Populate result data
            if result and isinstance(result, dict):
                session_status = result.get("sessionStatus")
                if session_status:
                    tool_call.attributes["bedrock.agentcore.browser.session_status"] = (
                        safe_str(session_status)
                    )

            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_tool_call(
                tool_call, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If tool call creation failed, just call original
        return wrapped(*args, **kwargs)


def wrap_browser_operation(
    operation_name: str,
) -> Any:
    """Generic wrapper factory for BrowserClient operations that creates ToolCall spans.

    Args:
        operation_name: Name of the operation (e.g., "list_sessions", "create_browser")

    Returns:
        Wrapper function
    """

    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
        capture_content: bool = False,
    ) -> Any:
        try:
            invocation = ToolCall(
                name=f"browser.{operation_name}",
                arguments=safe_json_dumps(kwargs)
                if capture_content and kwargs
                else None,
                system="bedrock-agentcore",
            )

            handler.start_tool_call(invocation)

            try:
                result = wrapped(*args, **kwargs)

                if capture_content and result is not None:
                    invocation.tool_result = (
                        safe_json_dumps(result)
                        if not isinstance(result, str)
                        else result
                    )

                handler.stop_tool_call(invocation)

                return result
            except Exception as e:
                handler.fail_tool_call(
                    invocation, Error(type=type(e).__name__, message=safe_str(e))
                )
                raise
        except Exception:
            return wrapped(*args, **kwargs)

    return wrapper
