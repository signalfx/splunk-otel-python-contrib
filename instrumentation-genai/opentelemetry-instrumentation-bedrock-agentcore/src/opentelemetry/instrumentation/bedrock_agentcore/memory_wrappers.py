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

"""Wrapt wrappers for Bedrock AgentCore Memory instrumentation."""

from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, RetrievalInvocation, ToolCall

from .utils import bind_call_arguments, invoke_tool_call, safe_json_dumps, safe_str, truncate_error


def wrap_memory_retrieve(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        query = call_arguments.get("query", "")
        top_k = call_arguments.get("top_k", 3)
        invocation = RetrievalInvocation(
            retriever_type="bedrock-agentcore-memory",
            query=safe_str(query) if capture_content else "",
            top_k=top_k,
        )
        handler.start_retrieval(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_retrieval(invocation, Error(type=type(e), message=truncate_error(e)))
        raise

    if isinstance(result, (list, dict)):
        records = result if isinstance(result, list) else result.get("memoryRecords", [])
        invocation.documents_retrieved = len(records)

    handler.stop_retrieval(invocation)
    return result


def wrap_memory_create_event(
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
            name="memory.create_event",
            arguments=safe_json_dumps({
                "memory_id": safe_str(call_arguments.get("memory_id")),
                "actor_id": safe_str(call_arguments.get("actor_id")),
                "session_id": safe_str(call_arguments.get("session_id")),
            }) if capture_content else None,
            system="bedrock-agentcore",
        )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, invocation, wrapped, args, kwargs, capture_content)


def wrap_memory_create_blob_event(
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
            name="memory.create_blob_event",
            arguments=safe_json_dumps({
                "memory_id": safe_str(call_arguments.get("memory_id")),
                "actor_id": safe_str(call_arguments.get("actor_id")),
                "session_id": safe_str(call_arguments.get("session_id")),
            }) if capture_content else None,
            system="bedrock-agentcore",
        )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, invocation, wrapped, args, kwargs, capture_content)


def wrap_memory_list_events(
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
            name="memory.list_events",
            arguments=safe_json_dumps({"memory_id": safe_str(call_arguments.get("memory_id"))})
            if capture_content else None,
            system="bedrock-agentcore",
        )
    except Exception:
        return wrapped(*args, **kwargs)

    return invoke_tool_call(handler, invocation, wrapped, args, kwargs, capture_content)


def wrap_memory_operation(operation_name: str) -> Any:
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
                name=f"memory.{operation_name}",
                arguments=safe_json_dumps(call_arguments) if capture_content else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        return invoke_tool_call(handler, invocation, wrapped, args, kwargs, capture_content)

    return wrapper
