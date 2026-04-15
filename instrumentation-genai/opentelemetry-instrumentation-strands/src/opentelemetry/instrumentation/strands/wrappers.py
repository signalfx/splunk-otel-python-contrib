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

"""Wrapt wrappers for Strands Agent lifecycle instrumentation."""

import asyncio
import functools
import logging
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Error,
    OutputMessage,
    Text,
    Workflow,
)

from .utils import (
    convert_strands_messages,
    extract_agent_name,
    extract_model_id,
    extract_tools_list,
    safe_str,
)

_LOGGER = logging.getLogger(__name__)

# Store original tracer methods for restoration
_original_tracer_methods: dict[str, Any] = {}


def wrap_agent_init(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    hook_provider: Any,
) -> Any:
    """Wrap Agent.__init__ to inject StrandsHookProvider into hook registry.

    Args:
        wrapped: Original __init__ method
        instance: Agent instance
        args: Positional arguments
        kwargs: Keyword arguments
        hook_provider: StrandsHookProvider instance

    Returns:
        Result of original __init__
    """
    try:
        # Call original __init__
        result = wrapped(*args, **kwargs)

        # Inject hook provider into agent's hook registry (attribute is `hooks`, not `hook_registry`)
        try:
            hook_registry = getattr(instance, "hooks", None)
            if hook_registry and hasattr(hook_provider, "register_hooks"):
                hook_provider.register_hooks(hook_registry)
        except Exception as e:
            _LOGGER.debug("Failed to inject hook provider into agent: %s", e)

        return result
    except Exception as e:
        _LOGGER.debug("Error in wrap_agent_init: %s", e)
        return wrapped(*args, **kwargs)


def wrap_agent_call(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap Agent.__call__ to create AgentInvocation span.

    Args:
        wrapped: Original __call__ method
        instance: Agent instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original __call__
    """
    try:
        # Create AgentInvocation
        invocation = _create_agent_invocation(instance, args, kwargs)

        # Start the invocation
        handler.start_agent(invocation)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Populate result data
            _populate_agent_result(invocation, result)

            # Stop the invocation successfully
            handler.stop_agent(invocation)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_agent(
                invocation, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If invocation creation failed, just call original
        return wrapped(*args, **kwargs)


async def wrap_agent_invoke_async(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap Agent.invoke_async to create AgentInvocation span.

    Args:
        wrapped: Original invoke_async method
        instance: Agent instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original invoke_async
    """
    try:
        # Create AgentInvocation
        invocation = _create_agent_invocation(instance, args, kwargs)

        # Start the invocation
        handler.start_agent(invocation)

        try:
            # Call original async method
            result = await wrapped(*args, **kwargs)

            # Populate result data
            _populate_agent_result(invocation, result)

            # Stop the invocation successfully
            handler.stop_agent(invocation)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_agent(
                invocation, Error(type=error_type, message=error_message)
            )
            raise
    except Exception:
        # If invocation creation failed, just call original
        return await wrapped(*args, **kwargs)


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
                workflow = Workflow(name=workflow_name, system="strands")
                handler.start_workflow(workflow)
                try:
                    result = await decorated_func(*call_args, **call_kwargs)
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
            workflow = Workflow(name=workflow_name, system="strands")
            handler.start_workflow(workflow)
            try:
                result = decorated_func(*call_args, **call_kwargs)
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


def _create_agent_invocation(
    instance: Any,
    args: tuple,
    kwargs: dict,
) -> AgentInvocation:
    """Create AgentInvocation from Agent instance and call arguments.

    Args:
        instance: Agent instance
        args: Positional arguments to __call__
        kwargs: Keyword arguments to __call__

    Returns:
        AgentInvocation instance
    """
    # Extract agent name
    agent_name = extract_agent_name(instance) or "strands_agent"

    # Extract model ID
    model_id = extract_model_id(instance) or "unknown"

    # Extract tools
    tools = extract_tools_list(instance)

    # Extract input message (first positional arg or 'input' kwarg)
    input_messages = []
    if args:
        input_messages = convert_strands_messages(args[0])
    elif "input" in kwargs:
        input_messages = convert_strands_messages(kwargs["input"])
    elif "prompt" in kwargs:
        input_messages = convert_strands_messages(kwargs["prompt"])
    elif "message" in kwargs:
        input_messages = convert_strands_messages(kwargs["message"])

    # Extract system instructions
    system_instructions = getattr(instance, "system_prompt", None) or getattr(
        instance, "instructions", None
    )

    # Create invocation
    invocation = AgentInvocation(
        name=agent_name,
        model=model_id,
        input_messages=input_messages,
        system="strands",
        framework="strands",
        tools=tools,
        system_instructions=safe_str(system_instructions)
        if system_instructions
        else None,
    )

    return invocation


def _populate_agent_result(invocation: AgentInvocation, result: Any) -> None:
    """Populate AgentInvocation with result data.

    Args:
        invocation: AgentInvocation to populate
        result: Result from agent execution
    """
    try:
        # Extract output message
        if result:
            # Handle AgentResult object
            if hasattr(result, "output") or hasattr(result, "content"):
                content = getattr(result, "output", None) or getattr(
                    result, "content", None
                )
                if content:
                    output_message = OutputMessage(
                        role="assistant", parts=[Text(content=safe_str(content))]
                    )
                    invocation.output_messages = [output_message]

            # Handle string result
            elif isinstance(result, str):
                output_message = OutputMessage(
                    role="assistant", parts=[Text(content=result)]
                )
                invocation.output_messages = [output_message]

            # Extract token usage if available
            if hasattr(result, "usage"):
                usage = getattr(result, "usage", None)
                if usage:
                    invocation.usage_input_tokens = getattr(
                        usage, "prompt_tokens", None
                    ) or getattr(usage, "input_tokens", None)
                    invocation.usage_output_tokens = getattr(
                        usage, "completion_tokens", None
                    ) or getattr(usage, "output_tokens", None)
    except Exception as e:
        _LOGGER.debug("Error populating agent result: %s", e)


def suppress_builtin_tracer() -> None:
    """Suppress Strands' built-in OTel tracer by replacing span methods with no-ops.

    Stores original methods for later restoration.
    """
    try:
        from strands.telemetry.tracer import Tracer

        _noop = lambda self, *args, **kwargs: None  # noqa: E731

        _methods = [
            "start_agent_span",
            "end_agent_span",
            "start_tool_call_span",
            "end_tool_call_span",
            "start_model_invoke_span",
            "end_model_invoke_span",
            "start_event_loop_cycle_span",
            "end_event_loop_cycle_span",
            "start_multiagent_span",
            "start_swarm_span",
            "end_swarm_span",
            "end_span_with_error",
            # legacy names kept for safety
            "start_span",
            "end_span",
            "start_llm_span",
            "start_tool_span",
        ]

        for method in _methods:
            if hasattr(Tracer, method):
                _original_tracer_methods[method] = getattr(Tracer, method)
                setattr(Tracer, method, _noop)

        _LOGGER.debug("Suppressed Strands built-in tracer")
    except (ImportError, AttributeError) as e:
        _LOGGER.debug("Failed to suppress Strands built-in tracer: %s", e)


def restore_builtin_tracer() -> None:
    """Restore Strands' built-in OTel tracer methods."""
    try:
        # Import Strands tracer
        from strands.telemetry.tracer import Tracer

        # Restore original methods
        for method_name, original_method in _original_tracer_methods.items():
            setattr(Tracer, method_name, original_method)

        _original_tracer_methods.clear()
        _LOGGER.debug("Restored Strands built-in tracer")
    except (ImportError, AttributeError) as e:
        _LOGGER.debug("Failed to restore Strands built-in tracer: %s", e)
