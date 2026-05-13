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

from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ToolCall

from .utils import bind_call_arguments, invoke_tool_call, safe_json_dumps, safe_str


def wrap_code_interpreter_execute(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
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
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, result: Any) -> None:
        if not result or not isinstance(result, dict):
            return
        errors = result.get("errors", [])
        if errors:
            tc.attributes["bedrock.agentcore.code_interpreter.has_errors"] = True
        if capture_content:
            output = result.get("output", "")
            tc.tool_result = safe_json_dumps(
                {
                    "output": output[:1000] if output else "",
                    "has_errors": bool(errors),
                    "error_count": len(errors) if errors else 0,
                }
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


def wrap_code_interpreter_install_packages(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
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
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_code_interpreter_upload_file(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        files = call_arguments.get("files")
        is_multi = files is not None
        operation_name = "upload_files" if is_multi else "upload_file"
        file_count = len(files) if is_multi else 1
        path = call_arguments.get("path", call_arguments.get("filename", ""))
        description = call_arguments.get("description", "")
        tool_call = ToolCall(
            name=f"code_interpreter.{operation_name}",
            arguments=safe_json_dumps(
                {"file_count": file_count}
                if is_multi
                else {"path": path, "description": description}
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        tool_call.attributes["bedrock.agentcore.code_interpreter.file_count"] = (
            file_count
        )
        if not is_multi:
            tool_call.attributes["bedrock.agentcore.code_interpreter.filename"] = (
                safe_str(path)
            )
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_code_interpreter_download_file(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        is_multi = "paths" in call_arguments
        paths = call_arguments.get("paths", call_arguments.get("path", ""))
        tool_call = ToolCall(
            name="code_interpreter.download_files"
            if is_multi
            else "code_interpreter.download_file",
            arguments=safe_json_dumps({"paths": paths} if is_multi else {"path": paths})
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
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — returns raw file content
    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content=False
    )


def wrap_code_interpreter_execute_command(
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
            name="code_interpreter.execute_command",
            arguments=safe_json_dumps({"command": call_arguments.get("command", "")})
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
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — may contain sensitive stdout/stderr
    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content=False
    )


def wrap_code_interpreter_clear_context(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        tool_call = ToolCall(
            name="code_interpreter.clear_context",
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
        if hasattr(instance, "session_id") and instance.session_id:
            tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] = (
                safe_str(instance.session_id)
            )
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — context cleanup responses may include state details
    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content=False
    )


def wrap_code_interpreter_create(
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
            name="code_interpreter.create_code_interpreter",
            arguments=safe_json_dumps(
                {
                    "name": safe_str(call_arguments.get("name", "")),
                    "description": safe_str(call_arguments.get("description", "")),
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
            tool_type="extension",
        )
        tool_call.attributes["bedrock.agentcore.tool.type"] = "code_interpreter"
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_code_interpreter_start(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
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
    except Exception:
        return wrapped(*args, **kwargs)

    def enrich(tc: ToolCall, _result: Any) -> None:
        if hasattr(instance, "session_id") and instance.session_id:
            tc.attributes["bedrock.agentcore.code_interpreter.session_id"] = safe_str(
                instance.session_id
            )

    return invoke_tool_call(
        handler, tool_call, wrapped, args, kwargs, capture_content, enrich
    )


def wrap_code_interpreter_stop(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
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
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, tool_call, wrapped, args, kwargs, capture_content)


def wrap_code_interpreter_operation(operation_name: str) -> Any:
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
                name=f"code_interpreter.{operation_name}",
                arguments=safe_json_dumps(call_arguments) if capture_content else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        return invoke_tool_call(
            handler, invocation, wrapped, args, kwargs, capture_content
        )

    return wrapper
