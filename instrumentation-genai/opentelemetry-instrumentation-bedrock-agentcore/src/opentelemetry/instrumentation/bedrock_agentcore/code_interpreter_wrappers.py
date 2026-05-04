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

"""Wrapt wrappers for Bedrock AgentCore Code Interpreter instrumentation."""

import logging
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, ToolCall

from .utils import bind_call_arguments, safe_json_dumps, safe_str, truncate_error

_LOGGER = logging.getLogger(__name__)


def wrap_code_interpreter_execute(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap CodeInterpreter.execute_code to create ToolCall span.

    Args:
        wrapped: Original execute_code method
        instance: CodeInterpreter instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture code/output content in spans

    Returns:
        Result of original execute_code
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        code = call_arguments.get("code", "")
        tool_call = ToolCall(
            name="code_interpreter.execute",
            arguments=safe_json_dumps({"code": code[:500]})
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )

        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )

        handler.start_tool_call(tool_call)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_tool_call(
            tool_call, Error(type=type(e), message=truncate_error(e))
        )
        raise

    try:
        if result:
            if isinstance(result, dict):
                errors = result.get("errors", [])

                if capture_content:
                    output = result.get("output", "")
                    tool_call.tool_result = safe_json_dumps(
                        {
                            "output": output[:1000] if output else "",
                            "has_errors": bool(errors),
                            "error_count": len(errors) if errors else 0,
                        }
                    )

                if errors:
                    tool_call.attributes[
                        "bedrock.agentcore.code_interpreter.has_errors"
                    ] = True
            elif capture_content:
                tool_call.tool_result = safe_str(result)[:1000]
    except Exception:
        _LOGGER.debug(
            "Failed to enrich code interpreter execute tool call.", exc_info=True
        )

    handler.stop_tool_call(tool_call)
    return result


def wrap_code_interpreter_install_packages(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap CodeInterpreter.install_packages to create ToolCall span.

    Args:
        wrapped: Original install_packages method
        instance: CodeInterpreter instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture package list/result content in spans

    Returns:
        Result of original install_packages
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        packages = call_arguments.get("packages", [])
        tool_call = ToolCall(
            name="code_interpreter.install_packages",
            arguments=safe_json_dumps({"packages": packages})
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )

        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        tool_call.attributes["bedrock.agentcore.code_interpreter.package_count"] = (
            len(packages) if packages else 0
        )
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )

        handler.start_tool_call(tool_call)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_tool_call(
            tool_call, Error(type=type(e), message=truncate_error(e))
        )
        raise

    try:
        if capture_content and result:
            tool_call.tool_result = safe_json_dumps(result)
    except Exception:
        _LOGGER.debug(
            "Failed to enrich code interpreter install_packages tool call.",
            exc_info=True,
        )

    handler.stop_tool_call(tool_call)
    return result


def wrap_code_interpreter_upload_file(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap CodeInterpreter.upload_file to create ToolCall span.

    Args:
        wrapped: Original upload_file method
        instance: CodeInterpreter instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture filename/result content in spans

    Returns:
        Result of original upload_file
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        filename = call_arguments.get("filename", call_arguments.get("path", ""))
        description = call_arguments.get("description", "")
        tool_call = ToolCall(
            name="code_interpreter.upload_file",
            arguments=safe_json_dumps(
                {
                    "filename": filename,
                    "description": description,
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )

        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        tool_call.attributes["bedrock.agentcore.code_interpreter.filename"] = safe_str(
            filename
        )
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )

        handler.start_tool_call(tool_call)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_tool_call(
            tool_call, Error(type=type(e), message=truncate_error(e))
        )
        raise

    try:
        if capture_content and result:
            tool_call.tool_result = safe_json_dumps(result)
    except Exception:
        _LOGGER.debug(
            "Failed to enrich code interpreter upload_file tool call.",
            exc_info=True,
        )

    handler.stop_tool_call(tool_call)
    return result


def wrap_code_interpreter_start(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap CodeInterpreter.start to create ToolCall span.

    Args:
        wrapped: Original start method
        instance: CodeInterpreter instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original start
    """
    try:
        tool_call = ToolCall(
            name="code_interpreter.start",
            system="bedrock-agentcore",
            tool_type="extension",
        )

        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        tool_call.attributes["bedrock.agentcore.code_interpreter.operation"] = (
            "start_session"
        )

        handler.start_tool_call(tool_call)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_tool_call(
            tool_call, Error(type=type(e), message=truncate_error(e))
        )
        raise

    try:
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )
            tool_call.tool_result = safe_json_dumps({"session_id": instance.session_id})
    except Exception:
        _LOGGER.debug(
            "Failed to enrich code interpreter start tool call.", exc_info=True
        )

    handler.stop_tool_call(tool_call)
    return result


def wrap_code_interpreter_stop(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap CodeInterpreter.stop to create ToolCall span.

    Args:
        wrapped: Original stop method
        instance: CodeInterpreter instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original stop
    """
    try:
        tool_call = ToolCall(
            name="code_interpreter.stop",
            system="bedrock-agentcore",
            tool_type="extension",
        )

        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        tool_call.attributes["bedrock.agentcore.code_interpreter.operation"] = (
            "stop_session"
        )
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )

        handler.start_tool_call(tool_call)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_tool_call(
            tool_call, Error(type=type(e), message=truncate_error(e))
        )
        raise

    handler.stop_tool_call(tool_call)
    return result


def wrap_code_interpreter_operation(
    operation_name: str,
) -> Any:
    """Generic wrapper factory for CodeInterpreter operations that creates ToolCall spans.

    Args:
        operation_name: Name of the operation (e.g., "download_file", "get_session")

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
                name=f"code_interpreter.{operation_name}",
                arguments=safe_json_dumps(kwargs)
                if capture_content and kwargs
                else None,
                system="bedrock-agentcore",
            )
            handler.start_tool_call(invocation)
        except Exception:
            return wrapped(*args, **kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:
            handler.fail_tool_call(
                invocation, Error(type=type(e), message=truncate_error(e))
            )
            raise

        try:
            if capture_content and result is not None:
                invocation.tool_result = (
                    safe_json_dumps(result) if not isinstance(result, str) else result
                )
        except Exception:
            _LOGGER.debug(
                "Failed to enrich code interpreter %s tool call.",
                operation_name,
                exc_info=True,
            )

        handler.stop_tool_call(invocation)
        return result

    return wrapper
