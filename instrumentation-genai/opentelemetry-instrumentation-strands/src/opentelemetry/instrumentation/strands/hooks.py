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

"""Strands hook provider for LLM and tool call telemetry."""

import logging
from typing import Any, Dict

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, LLMInvocation, OutputMessage, Text, ToolCall

from .utils import convert_strands_messages, safe_json_dumps, safe_str

_LOGGER = logging.getLogger(__name__)


class StrandsHookProvider:
    """Hook provider for Strands SDK lifecycle events.

    Registers callbacks for BeforeModelCallEvent, AfterModelCallEvent,
    BeforeToolCallEvent, and AfterToolCallEvent to capture LLM and tool
    telemetry via TelemetryHandler.
    """

    def __init__(self, handler: TelemetryHandler):
        """Initialize hook provider.

        Args:
            handler: TelemetryHandler instance for emitting telemetry
        """
        self.handler = handler
        # Track active LLM invocations by invocation_state id
        self._active_llm_invocations: Dict[int, LLMInvocation] = {}
        # Track active tool calls by tool_use_id
        self._active_tool_calls: Dict[str, ToolCall] = {}

    def register_hooks(self, registry: Any) -> None:
        """Register hook callbacks with Strands hook registry.

        Args:
            registry: Strands HookRegistry instance
        """
        try:
            # Register LLM hooks
            registry.register("BeforeModelCallEvent", self._on_before_model_call)
            registry.register("AfterModelCallEvent", self._on_after_model_call)
            # Register tool hooks
            registry.register("BeforeToolCallEvent", self._on_before_tool_call)
            registry.register("AfterToolCallEvent", self._on_after_tool_call)
        except Exception as e:
            _LOGGER.debug("Failed to register Strands hooks: %s", e)

    def _on_before_model_call(self, event: Any) -> None:
        """Handle BeforeModelCallEvent - start LLM invocation.

        Args:
            event: BeforeModelCallEvent from Strands SDK
        """
        try:
            # Extract model information
            model_id = safe_str(getattr(event, "model", None) or getattr(event, "model_id", None))

            # Extract input messages
            input_messages = []
            messages = getattr(event, "messages", None)
            if messages:
                input_messages = convert_strands_messages(messages)

            # Extract prompt if available
            prompt = getattr(event, "prompt", None)
            if prompt and not input_messages:
                input_messages = convert_strands_messages(prompt)

            # Create LLM invocation
            invocation = LLMInvocation(
                request_model=model_id or "unknown",
                input_messages=input_messages,
                system="strands",
                provider=self._extract_provider(model_id),
            )

            # Extract additional parameters if available
            if hasattr(event, "temperature"):
                invocation.request_temperature = getattr(event, "temperature", None)
            if hasattr(event, "max_tokens"):
                invocation.request_max_tokens = getattr(event, "max_tokens", None)
            if hasattr(event, "top_p"):
                invocation.request_top_p = getattr(event, "top_p", None)

            # Start the invocation
            self.handler.start_llm(invocation)

            # Track by invocation_state id
            invocation_state = getattr(event, "invocation_state", None)
            if invocation_state:
                state_id = id(invocation_state)
                self._active_llm_invocations[state_id] = invocation

        except Exception as e:
            _LOGGER.debug("Error in _on_before_model_call: %s", e)

    def _on_after_model_call(self, event: Any) -> None:
        """Handle AfterModelCallEvent - stop LLM invocation.

        Args:
            event: AfterModelCallEvent from Strands SDK
        """
        try:
            # Find the matching invocation
            invocation_state = getattr(event, "invocation_state", None)
            if not invocation_state:
                return

            state_id = id(invocation_state)
            invocation = self._active_llm_invocations.get(state_id)
            if not invocation:
                return

            # Check for error
            error = getattr(event, "error", None)
            exception = getattr(event, "exception", None)

            if error or exception:
                # Handle error case
                error_message = safe_str(error or exception)
                error_type = type(exception).__name__ if exception else "unknown"
                self.handler.fail_llm(
                    invocation,
                    Error(type=error_type, message=error_message)
                )
            else:
                # Extract response data
                response = getattr(event, "response", None)
                if response:
                    # Extract model from response
                    response_model = getattr(response, "model", None)
                    if response_model:
                        invocation.response_model = safe_str(response_model)

                    # Extract token usage
                    usage = getattr(response, "usage", None)
                    if usage:
                        invocation.usage_input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                        invocation.usage_output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)

                    # Extract output messages
                    content = getattr(response, "content", None)
                    if content:
                        output_message = OutputMessage(
                            role="assistant",
                            parts=[Text(content=safe_str(content))]
                        )
                        invocation.output_messages = [output_message]

                    # Extract finish reason
                    finish_reason = getattr(response, "finish_reason", None) or getattr(response, "stop_reason", None)
                    if finish_reason and invocation.output_messages:
                        invocation.output_messages[0].finish_reason = safe_str(finish_reason)

                # Stop the invocation successfully
                self.handler.stop_llm(invocation)

            # Clean up tracking
            self._active_llm_invocations.pop(state_id, None)

        except Exception as e:
            _LOGGER.debug("Error in _on_after_model_call: %s", e)

    def _on_before_tool_call(self, event: Any) -> None:
        """Handle BeforeToolCallEvent - start tool call.

        Args:
            event: BeforeToolCallEvent from Strands SDK
        """
        try:
            # Extract tool information
            tool_name = safe_str(getattr(event, "tool_name", None) or getattr(event, "name", None) or "unknown")
            tool_use_id = safe_str(getattr(event, "tool_use_id", None) or getattr(event, "id", None))

            # Extract arguments
            arguments = getattr(event, "arguments", None) or getattr(event, "input", None)
            arguments_str = None
            if arguments:
                arguments_str = safe_json_dumps(arguments)

            # Create tool call
            tool_call = ToolCall(
                name=tool_name,
                id=tool_use_id if tool_use_id else None,
                arguments=arguments_str,
                system="strands",
            )

            # Extract tool description if available
            tool_description = getattr(event, "description", None)
            if tool_description:
                tool_call.tool_description = safe_str(tool_description)

            # Start the tool call
            self.handler.start_tool_call(tool_call)

            # Track by tool_use_id
            if tool_use_id:
                self._active_tool_calls[tool_use_id] = tool_call

        except Exception as e:
            _LOGGER.debug("Error in _on_before_tool_call: %s", e)

    def _on_after_tool_call(self, event: Any) -> None:
        """Handle AfterToolCallEvent - stop tool call.

        Args:
            event: AfterToolCallEvent from Strands SDK
        """
        try:
            # Find the matching tool call
            tool_use_id = safe_str(getattr(event, "tool_use_id", None) or getattr(event, "id", None))
            if not tool_use_id:
                return

            tool_call = self._active_tool_calls.get(tool_use_id)
            if not tool_call:
                return

            # Check for error
            error = getattr(event, "error", None)
            exception = getattr(event, "exception", None)

            if error or exception:
                # Handle error case
                error_message = safe_str(error or exception)
                error_type = type(exception).__name__ if exception else "unknown"
                self.handler.fail_tool_call(
                    tool_call,
                    Error(type=error_type, message=error_message)
                )
            else:
                # Extract result
                result = getattr(event, "result", None)
                if result is not None:
                    tool_call.tool_result = safe_json_dumps(result) if not isinstance(result, str) else result

                # Stop the tool call successfully
                self.handler.stop_tool_call(tool_call)

            # Clean up tracking
            self._active_tool_calls.pop(tool_use_id, None)

        except Exception as e:
            _LOGGER.debug("Error in _on_after_tool_call: %s", e)

    def _extract_provider(self, model_id: str) -> str:
        """Extract provider name from model ID.

        Args:
            model_id: Model identifier (e.g., "anthropic.claude-v2", "bedrock-anthropic.claude-v2")

        Returns:
            Provider name (e.g., "anthropic", "bedrock")
        """
        if not model_id:
            return "unknown"

        # Check for common provider prefixes
        model_lower = model_id.lower()
        if "anthropic" in model_lower:
            return "anthropic"
        elif "bedrock" in model_lower:
            return "bedrock"
        elif "openai" in model_lower:
            return "openai"
        elif "cohere" in model_lower:
            return "cohere"
        elif "ai21" in model_lower:
            return "ai21"

        # Try to extract from dot-separated format
        parts = model_id.split(".")
        if len(parts) > 1:
            return parts[0]

        return "unknown"
