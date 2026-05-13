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

from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ToolCall

from .utils import bind_call_arguments, invoke_tool_call, safe_json_dumps, safe_str


def wrap_browser_start(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        browser_id = call_arguments.get("browser_id")
        tool_call = ToolCall(
            name="browser.start",
            arguments=safe_json_dumps({"browser_id": browser_id})
            if capture_content and browser_id
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "start_session"
        if browser_id:
            tool_call.attributes["bedrock.agentcore.browser.id"] = safe_str(browser_id)
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, _result: Any) -> None:
        if hasattr(instance, "session_id") and instance.session_id:
            tc.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )

    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content, enrich
    )


def wrap_browser_stop(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        tool_call = ToolCall(
            name="browser.stop",
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "stop_session"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_browser_take_control(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        tool_call = ToolCall(
            name="browser.take_control",
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "take_control"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_browser_release_control(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        tool_call = ToolCall(
            name="browser.release_control",
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "release_control"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_browser_get_session(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        tool_call = ToolCall(
            name="browser.get_session",
            arguments=safe_json_dumps(
                {
                    "browser_id": call_arguments.get("browser_id"),
                    "session_id": call_arguments.get("session_id"),
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "get_session"
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, result: Any) -> None:
        if result and isinstance(result, dict):
            session_status = result.get("sessionStatus")
            if session_status:
                tc.attributes["bedrock.agentcore.browser.session_status"] = safe_str(
                    session_status
                )

    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content, enrich
    )


def wrap_browser_generate_ws_headers(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        tool_call = ToolCall(
            name="browser.generate_ws_headers",
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — returns auth credentials
    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content=False
    )


def wrap_browser_generate_live_view_url(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        tool_call = ToolCall(
            name="browser.generate_live_view_url",
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = safe_str(
                instance.session_id
            )
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — returns presigned URL with embedded tokens
    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content=False
    )


def wrap_browser_operation(operation_name: str) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
        capture_content: bool = False,
    ) -> Any:
        try:
            call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
            invocation = ToolCall(
                name=f"browser.{operation_name}",
                arguments=safe_json_dumps(call_arguments) if capture_content else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        return invoke_tool_call(
            handler, invocation, wrapped, args, kwargs, capture_content
        )

    return wrapper
