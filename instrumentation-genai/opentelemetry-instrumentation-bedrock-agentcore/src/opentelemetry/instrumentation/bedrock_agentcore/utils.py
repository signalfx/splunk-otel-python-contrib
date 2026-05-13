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

"""Utility functions for Bedrock AgentCore instrumentation."""

import inspect
import json
from os import environ
from typing import Any, Callable, Optional

from opentelemetry.util.genai.types import Error, ToolCall

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)
OTEL_INSTRUMENTATION_GENAI_ENABLE = "OTEL_INSTRUMENTATION_GENAI_ENABLE"


def is_instrumentation_enabled() -> bool:
    return environ.get(OTEL_INSTRUMENTATION_GENAI_ENABLE, "true").lower() == "true"


def is_content_enabled() -> bool:
    return (
        environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false").lower()
        == "true"
    )


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize an object to JSON, with fallback on error.

    Args:
        obj: Object to serialize

    Returns:
        JSON string, or repr(obj) if JSON serialization fails
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return repr(obj)


def safe_str(value: Any) -> str:
    """Safely convert any value to string, never raising exceptions.

    Args:
        value: Any value to convert

    Returns:
        String representation of the value
    """
    try:
        return str(value)
    except Exception:
        return repr(value)


def bind_call_arguments(
    wrapped: Any, instance: Any, args: tuple, kwargs: dict
) -> dict[str, Any]:
    """Bind call arguments to a callable signature.

    Args:
        wrapped: Callable being wrapped.
        instance: Bound instance supplied by wrapt, if any.
        args: Positional call arguments.
        kwargs: Keyword call arguments.

    Returns:
        Mapping from parameter names to values, including defaults.

    Raises:
        TypeError: If the arguments do not match the callable signature.
        ValueError: If the callable does not expose an inspectable signature.
    """
    call_signature = inspect.signature(wrapped)
    parameters = list(call_signature.parameters.values())
    bind_args = args

    if (
        instance is not None
        and parameters
        and parameters[0].name in {"self", "cls"}
        and parameters[0].kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameters[0].name not in kwargs
    ):
        bind_args = (instance, *args)

    bound = call_signature.bind(*bind_args, **kwargs)
    bound.apply_defaults()

    arguments = dict(bound.arguments)
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            extra_kwargs = arguments.pop(parameter.name, {})
            if isinstance(extra_kwargs, dict):
                for key, value in extra_kwargs.items():
                    arguments.setdefault(key, value)

    arguments.pop("self", None)
    arguments.pop("cls", None)
    return arguments


def invoke_tool_call(
    handler: Any,
    tool_call: ToolCall,
    wrapped: Any,
    args: tuple,
    kwargs: dict,
    capture_content: bool,
    enrich_result: Optional[Callable[[ToolCall, Any], None]] = None,
) -> Any:
    handler.start_tool_call(tool_call)
    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_tool_call(
            tool_call, Error(type=type(e), message=truncate_error(e))
        )
        raise
    if capture_content and result is not None:
        serialized = safe_json_dumps(result) if not isinstance(result, str) else result
        tool_call.tool_result = (
            serialized[:_RESULT_MAX_LEN] + "..."
            if len(serialized) > _RESULT_MAX_LEN
            else serialized
        )
    if enrich_result is not None:
        enrich_result(tool_call, result)
    handler.stop_tool_call(tool_call)
    return result


_ERROR_MAX_LEN = 256
_RESULT_MAX_LEN = 1024


def truncate_error(e: Exception) -> str:
    """Return a truncated, safe string from an exception.

    AWS SDK exceptions (ClientError etc.) can include ARNs, account IDs, and
    request metadata. We truncate to a fixed limit to avoid leaking sensitive
    details into spans.

    Args:
        e: Exception to convert

    Returns:
        Truncated string representation, at most _ERROR_MAX_LEN characters
    """
    msg = safe_str(e)
    if len(msg) > _ERROR_MAX_LEN:
        return msg[:_ERROR_MAX_LEN] + "..."
    return msg
