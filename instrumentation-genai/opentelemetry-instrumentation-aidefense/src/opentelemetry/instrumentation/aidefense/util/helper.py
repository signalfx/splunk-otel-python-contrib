# Copyright The OpenTelemetry Authors
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

"""
AI Defense Instrumentation Helpers

This module provides common utilities for AI Defense SDK instrumentation,
applying the DRY principle to reduce code repetition in wrapper functions.
"""

from typing import Any, Callable, List, Optional

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    Text,
)


# Common constants for AI Defense instrumentation
AI_DEFENSE_MODEL = "cisco-ai-defense"
AI_DEFENSE_SYSTEM = "aidefense"
AI_DEFENSE_FRAMEWORK = "aidefense"
AI_DEFENSE_OPERATION = "chat"

# Content limits for span size
MAX_CONTENT_LENGTH = 1000
MAX_SHORT_CONTENT_LENGTH = 500
MAX_MESSAGES_IN_CONVERSATION = 10


def create_ai_defense_invocation(
    server_address: str,
    input_messages: List[InputMessage],
    operation: str = AI_DEFENSE_OPERATION,
) -> LLMInvocation:
    """
    Create a standardized LLMInvocation for AI Defense SDK operations.

    Args:
        server_address: The AI Defense server address
        input_messages: List of input messages for the invocation
        operation: The operation type (default: "chat")

    Returns:
        LLMInvocation configured for AI Defense
    """
    return LLMInvocation(
        request_model=AI_DEFENSE_MODEL,
        server_address=server_address,
        operation=operation,
        system=AI_DEFENSE_SYSTEM,
        framework=AI_DEFENSE_FRAMEWORK,
        input_messages=input_messages,
    )


def create_input_message(
    role: str,
    content: str,
    max_length: int = MAX_CONTENT_LENGTH,
) -> InputMessage:
    """
    Create an InputMessage with content truncated to max_length.

    Args:
        role: Message role (user, assistant, system, etc.)
        content: Message content
        max_length: Maximum content length (default: 1000)

    Returns:
        InputMessage with truncated content
    """
    truncated_content = str(content)[:max_length]
    return InputMessage(role=role, parts=[Text(content=truncated_content)])


def execute_with_telemetry(
    handler: Optional[TelemetryHandler],
    invocation: LLMInvocation,
    wrapped: Callable,
    args: tuple,
    kwargs: dict,
    result_processor: Optional[Callable[[LLMInvocation, Any], None]] = None,
) -> Any:
    """
    Execute a wrapped function with telemetry handling.

    This helper reduces boilerplate in wrapper functions by handling the
    common pattern of:
    1. Starting the LLM invocation span
    2. Calling the wrapped function
    3. Processing the result
    4. Stopping the span (or failing on exception)

    Args:
        handler: TelemetryHandler instance (can be None for graceful degradation)
        invocation: LLMInvocation to track
        wrapped: The original function to call
        args: Positional arguments for wrapped function
        kwargs: Keyword arguments for wrapped function
        result_processor: Optional callback to process result before stopping span

    Returns:
        Result from the wrapped function

    Raises:
        Any exception from the wrapped function (after recording in telemetry)
    """
    # If no handler, just call the wrapped function
    if handler is None:
        return wrapped(*args, **kwargs)

    # Start the span
    try:
        handler.start_llm(invocation)
    except Exception:
        # If we can't start telemetry, still execute the function
        return wrapped(*args, **kwargs)

    # Execute and handle result/errors
    try:
        result = wrapped(*args, **kwargs)

        try:
            if result_processor:
                result_processor(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def add_event_id_to_current_span(
    event_id: str,
    source: str = "gateway",
) -> bool:
    """
    Add AI Defense event_id to the current active span.

    This is used by Gateway Mode wrappers to add the security event_id
    to the existing LLM span (from LangChain, OpenAI, etc.).

    Args:
        event_id: The AI Defense event ID from the X-Cisco-AI-Defense-Event-Id header
        source: Description of where the event_id came from (for logging)

    Returns:
        True if the attribute was successfully added, False otherwise
    """
    from opentelemetry import trace

    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.security.event_id", event_id)
            return True
    except Exception:
        pass
    return False


def get_server_address(instance: Any) -> str:
    """
    Extract server address from an AI Defense client instance.

    Supports multiple AI Defense SDK versions by checking various config attributes.

    Args:
        instance: AI Defense client instance (ChatInspectionClient or HttpInspectionClient)

    Returns:
        Server address string or empty string if not found
    """
    try:
        # Try newer SDK attribute path (instance.config.runtime_base_url)
        if hasattr(instance, "config"):
            config = instance.config
            if hasattr(config, "runtime_base_url") and config.runtime_base_url:
                return str(config.runtime_base_url)

        # Try older SDK attribute path (instance._config.backend_url)
        if hasattr(instance, "_config"):
            config = instance._config
            if hasattr(config, "backend_url") and config.backend_url:
                return str(config.backend_url)
            if hasattr(config, "region") and config.region:
                return f"aidefense.{config.region}.cisco.com"
    except Exception:
        pass
    return ""

