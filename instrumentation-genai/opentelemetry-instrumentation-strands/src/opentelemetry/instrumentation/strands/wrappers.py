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

import logging
from typing import Any, Optional

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import AgentInvocation, Error, OutputMessage, Text, Workflow

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

        # Inject hook provider into agent's hook registry
        try:
            hook_registry = getattr(instance, "hook_registry", None)
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
            handler.fail_agent(invocation, Error(type=error_type, message=error_message))
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
            handler.fail_agent(invocation, Error(type=error_type, message=error_message))
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
    """Wrap BedrockAgentCoreApp.entrypoint to create Workflow span.

    Args:
        wrapped: Original entrypoint method
        instance: BedrockAgentCoreApp instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original entrypoint
    """
    try:
        # Create Workflow
        workflow_name = getattr(instance, "name", None) or "BedrockAgentCore"
        workflow = Workflow(
            name=workflow_name,
            system="strands",
        )

        # Start the workflow
        handler.start_workflow(workflow)

        try:
            # Call original method
            result = wrapped(*args, **kwargs)

            # Stop the workflow successfully
            handler.stop_workflow(workflow)

            return result
        except Exception as e:
            # Handle error
            error_message = safe_str(e)
            error_type = type(e).__name__
            handler.fail_workflow(workflow, Error(type=error_type, message=error_message))
            raise
    except Exception:
        # If workflow creation failed, just call original
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

    # Create invocation
    invocation = AgentInvocation(
        agent_name=agent_name,
        input_messages=input_messages,
        system="strands",
        request_model=model_id,
    )

    # Add tools as attribute if available
    if tools:
        invocation.attributes["gen_ai.tools"] = tools

    # Extract instructions/system prompt if available
    instructions = getattr(instance, "instructions", None)
    if instructions:
        invocation.attributes["gen_ai.agent.instructions"] = safe_str(instructions)

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
                content = getattr(result, "output", None) or getattr(result, "content", None)
                if content:
                    output_message = OutputMessage(
                        role="assistant",
                        parts=[Text(content=safe_str(content))]
                    )
                    invocation.output_messages = [output_message]

            # Handle string result
            elif isinstance(result, str):
                output_message = OutputMessage(
                    role="assistant",
                    parts=[Text(content=result)]
                )
                invocation.output_messages = [output_message]

            # Extract token usage if available
            if hasattr(result, "usage"):
                usage = getattr(result, "usage", None)
                if usage:
                    invocation.usage_input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                    invocation.usage_output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    except Exception as e:
        _LOGGER.debug("Error populating agent result: %s", e)


def suppress_builtin_tracer() -> None:
    """Suppress Strands' built-in OTel tracer by replacing span methods with no-ops.

    Stores original methods for later restoration.
    """
    try:
        # Import Strands tracer
        from strands.telemetry.tracer import Tracer

        # Store and replace span creation methods
        if hasattr(Tracer, "start_span"):
            _original_tracer_methods["start_span"] = Tracer.start_span
            Tracer.start_span = lambda self, *args, **kwargs: None

        if hasattr(Tracer, "end_span"):
            _original_tracer_methods["end_span"] = Tracer.end_span
            Tracer.end_span = lambda self, *args, **kwargs: None

        if hasattr(Tracer, "start_agent_span"):
            _original_tracer_methods["start_agent_span"] = Tracer.start_agent_span
            Tracer.start_agent_span = lambda self, *args, **kwargs: None

        if hasattr(Tracer, "start_tool_span"):
            _original_tracer_methods["start_tool_span"] = Tracer.start_tool_span
            Tracer.start_tool_span = lambda self, *args, **kwargs: None

        if hasattr(Tracer, "start_llm_span"):
            _original_tracer_methods["start_llm_span"] = Tracer.start_llm_span
            Tracer.start_llm_span = lambda self, *args, **kwargs: None

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
