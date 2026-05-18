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
        session_status = _first_value(result, "status", "sessionStatus")
        if session_status:
            tc.attributes["bedrock.agentcore.browser.session_status"] = safe_str(
                session_status
            )

    return invoke_tool_call(
        handler,
        tool_call,
        wrapped,
        args,
        kwargs,
        capture_content=False,
        enrich_result=enrich,
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


def _first_value(result: Any, *keys: str) -> Any:
    if not isinstance(result, dict):
        return None

    for key in keys:
        value = result.get(key)
        if value is not None and value != "":
            return value

    for container_key in ("browser", "browserSummary"):
        nested = result.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if value is not None and value != "":
                    return value
    return None


def wrap_browser_create_browser(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        name = call_arguments.get("name")
        tool_call = ToolCall(
            name="browser.create_browser",
            arguments=safe_json_dumps({"name": safe_str(name)})
            if capture_content and name is not None
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "create_browser"
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, result: Any) -> None:
        browser_id = _first_value(result, "browserId", "browser_id", "id")
        if browser_id is not None:
            tc.attributes["bedrock.agentcore.browser.id"] = safe_str(browser_id)

    # never capture tool_result — control-plane responses can include infrastructure config
    return invoke_tool_call(
        handler,
        tool_call,
        wrapped,
        args,
        kwargs,
        capture_content=False,
        enrich_result=enrich,
    )


def wrap_browser_get_browser(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        browser_id = call_arguments.get("browser_id", call_arguments.get("browserId"))
        tool_call = ToolCall(
            name="browser.get_browser",
            arguments=safe_json_dumps({"browser_id": safe_str(browser_id)})
            if capture_content and browser_id is not None
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "get_browser"
        if browser_id is not None:
            tool_call.attributes["bedrock.agentcore.browser.id"] = safe_str(browser_id)
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, result: Any) -> None:
        status = _first_value(result, "status", "browserStatus")
        if status is not None:
            tc.attributes["bedrock.agentcore.browser.status"] = safe_str(status)

    # never capture tool_result — control-plane responses can include infrastructure config
    return invoke_tool_call(
        handler,
        tool_call,
        wrapped,
        args,
        kwargs,
        capture_content=False,
        enrich_result=enrich,
    )


def wrap_browser_list_browsers(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        max_results = call_arguments.get(
            "max_results", call_arguments.get("maxResults")
        )
        tool_call = ToolCall(
            name="browser.list_browsers",
            arguments=safe_json_dumps({"max_results": max_results})
            if capture_content and max_results is not None
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "list_browsers"
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, result: Any) -> None:
        browsers = result
        if isinstance(result, dict):
            browsers = result.get("browserSummaries", result.get("browsers"))
        if isinstance(browsers, list):
            tc.attributes["bedrock.agentcore.browser.count"] = len(browsers)

    # never capture tool_result — list responses can include infrastructure config
    return invoke_tool_call(
        handler,
        tool_call,
        wrapped,
        args,
        kwargs,
        capture_content=False,
        enrich_result=enrich,
    )


def wrap_browser_update_stream(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        safe_args = {}
        for source, target in (
            ("browser_id", "browser_id"),
            ("browserId", "browser_id"),
            ("session_id", "session_id"),
            ("sessionId", "session_id"),
            ("stream_id", "stream_id"),
            ("streamId", "stream_id"),
        ):
            value = call_arguments.get(source)
            if value is not None and value != "":
                safe_args[target] = safe_str(value)

        tool_call = ToolCall(
            name="browser.update_stream",
            arguments=safe_json_dumps(safe_args)
            if capture_content and safe_args
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "browser"
        tool_call.attributes["bedrock.agentcore.browser.operation"] = "update_stream"
        browser_id = safe_args.get("browser_id")
        if browser_id:
            tool_call.attributes["bedrock.agentcore.browser.id"] = browser_id
        session_id = safe_args.get("session_id")
        if session_id:
            tool_call.attributes["bedrock.agentcore.browser.session_id"] = session_id
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, result: Any) -> None:
        status = _first_value(result, "status", "streamStatus")
        if status is not None:
            tc.attributes["bedrock.agentcore.browser.stream_status"] = safe_str(status)

    # never capture tool_result — stream config can include S3/KMS ARNs and prefixes
    return invoke_tool_call(
        handler,
        tool_call,
        wrapped,
        args,
        kwargs,
        capture_content=False,
        enrich_result=enrich,
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
