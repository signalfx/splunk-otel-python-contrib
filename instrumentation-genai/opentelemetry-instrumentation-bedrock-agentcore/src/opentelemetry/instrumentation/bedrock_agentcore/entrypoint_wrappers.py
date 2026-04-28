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

"""Wrapt wrapper for BedrockAgentCoreApp.entrypoint."""

import asyncio
import functools
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, InputMessage, OutputMessage, Text, Workflow

from .utils import safe_json_dumps, safe_str


def _make_input_message(event: Any) -> InputMessage:
    """Convert an entrypoint event payload to an InputMessage for eval context."""
    if isinstance(event, str):
        content = event
    else:
        content = safe_json_dumps(event)
    return InputMessage(role="user", parts=[Text(content=content)])


def _make_output_message(result: Any) -> OutputMessage:
    """Convert an entrypoint return value to an OutputMessage for eval context."""
    if isinstance(result, str):
        content = result
    else:
        content = safe_json_dumps(result)
    return OutputMessage(role="assistant", parts=[Text(content=content)], finish_reason="stop")


def wrap_bedrock_agentcore_app_entrypoint(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap BedrockAgentCoreApp.entrypoint to create Workflow span per invocation.

    entrypoint is a decorator factory: it takes a function and returns a wrapped
    function. We intercept it and further wrap the returned function so that each
    call to the entrypoint function creates a Workflow span at invocation time,
    not at decoration time.

    Args:
        wrapped: Original entrypoint decorator method
        instance: BedrockAgentCoreApp instance
        args: Positional arguments (the function being decorated)
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Decorated function that creates a Workflow span on each call
    """
    try:
        # Call original entrypoint decorator to get the decorated function
        decorated_func = wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", None) or "BedrockAgentCore"

        if asyncio.iscoroutinefunction(decorated_func):

            @functools.wraps(decorated_func)
            async def async_workflow_wrapper(*call_args, **call_kwargs):
                workflow = Workflow(name=workflow_name, system="bedrock-agentcore")
                if call_args:
                    workflow.input_messages = [_make_input_message(call_args[0])]
                handler.start_workflow(workflow)
                try:
                    result = await decorated_func(*call_args, **call_kwargs)
                    if result is not None:
                        workflow.output_messages = [_make_output_message(result)]
                    handler.stop_workflow(workflow)
                    return result
                except Exception as e:
                    handler.fail_workflow(
                        workflow, Error(type=type(e).__name__, message=safe_str(e))
                    )
                    raise

            return async_workflow_wrapper

        @functools.wraps(decorated_func)
        def workflow_wrapper(*call_args, **call_kwargs):
            workflow = Workflow(name=workflow_name, system="bedrock-agentcore")
            if call_args:
                workflow.input_messages = [_make_input_message(call_args[0])]
            handler.start_workflow(workflow)
            try:
                result = decorated_func(*call_args, **call_kwargs)
                if result is not None:
                    workflow.output_messages = [_make_output_message(result)]
                handler.stop_workflow(workflow)
                return result
            except Exception as e:
                handler.fail_workflow(
                    workflow, Error(type=type(e).__name__, message=safe_str(e))
                )
                raise

        return workflow_wrapper
    except Exception:
        return wrapped(*args, **kwargs)
