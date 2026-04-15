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
from opentelemetry.util.genai.types import (
    Error,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
)

from .utils import convert_strands_messages, extract_model_id, safe_json_dumps, safe_str

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

        Uses add_callback() with explicit event type classes (not string names).

        Args:
            registry: Strands HookRegistry instance
        """
        try:
            from strands.hooks.events import (
                AfterModelCallEvent,
                AfterToolCallEvent,
                BeforeModelCallEvent,
                BeforeToolCallEvent,
            )

            registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
            registry.add_callback(AfterModelCallEvent, self._on_after_model_call)
            registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)
            registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)
        except Exception as e:
            _LOGGER.debug("Failed to register Strands hooks: %s", e)

    def _on_before_model_call(self, event: Any) -> None:
        """Handle BeforeModelCallEvent - start LLM invocation.

        BeforeModelCallEvent fields: agent, invocation_state (dict).
        Model ID is accessed via event.agent.model.config.model_id.

        Args:
            event: BeforeModelCallEvent from Strands SDK
        """
        try:
            # Extract model ID from the agent attached to the event
            agent = getattr(event, "agent", None)
            model_id = extract_model_id(agent) or ""

            # Extract input messages from invocation_state (contains conversation history)
            input_messages = []
            invocation_state = getattr(event, "invocation_state", None) or {}
            messages = (
                invocation_state.get("messages")
                if isinstance(invocation_state, dict)
                else None
            )
            if messages:
                input_messages = convert_strands_messages(messages)

            # Create LLM invocation
            invocation = LLMInvocation(
                request_model=model_id or "unknown",
                input_messages=input_messages,
                system="aws.bedrock",
                provider=self._extract_provider(model_id),
            )

            # Start the invocation
            self.handler.start_llm(invocation)

            # Track by invocation_state id for correlation with AfterModelCallEvent
            if invocation_state is not None:
                state_id = id(invocation_state)
                self._active_llm_invocations[state_id] = invocation

        except Exception as e:
            _LOGGER.debug("Error in _on_before_model_call: %s", e)

    def _on_after_model_call(self, event: Any) -> None:
        """Handle AfterModelCallEvent - stop LLM invocation.

        AfterModelCallEvent fields: agent, invocation_state, stop_response, exception, retry.
        stop_response is a ModelStopResponse with .message (Message) and .stop_reason.

        Args:
            event: AfterModelCallEvent from Strands SDK
        """
        try:
            # Find the matching invocation
            invocation_state = getattr(event, "invocation_state", None)
            if invocation_state is None:
                return

            state_id = id(invocation_state)
            invocation = self._active_llm_invocations.get(state_id)
            if not invocation:
                return

            exception = getattr(event, "exception", None)

            if exception:
                error_message = safe_str(exception)
                error_type = type(exception).__name__
                self.handler.fail_llm(
                    invocation, Error(type=error_type, message=error_message)
                )
            else:
                # Extract from stop_response (ModelStopResponse dataclass)
                stop_response = getattr(event, "stop_response", None)
                if stop_response:
                    # stop_reason is a StopReason (str-like)
                    stop_reason = getattr(stop_response, "stop_reason", None)

                    # message is a Message TypedDict with "role" and "content" keys
                    message = getattr(stop_response, "message", None)
                    if message:
                        content_blocks = (
                            message.get("content", [])
                            if isinstance(message, dict)
                            else getattr(message, "content", [])
                        )
                        # Flatten content blocks to text
                        text_parts = []
                        for block in content_blocks or []:
                            if isinstance(block, dict):
                                text = block.get("text", "")
                                if text:
                                    text_parts.append(text)
                            elif hasattr(block, "text"):
                                text = getattr(block, "text", "")
                                if text:
                                    text_parts.append(text)
                        combined = " ".join(text_parts)
                        if combined:
                            output_msg = OutputMessage(
                                role="assistant", parts=[Text(content=combined)]
                            )
                            if stop_reason:
                                output_msg.finish_reason = safe_str(stop_reason)
                            invocation.output_messages = [output_msg]

                # Stop the invocation successfully
                self.handler.stop_llm(invocation)

            # Clean up tracking
            self._active_llm_invocations.pop(state_id, None)

        except Exception as e:
            _LOGGER.debug("Error in _on_after_model_call: %s", e)

    def _on_before_tool_call(self, event: Any) -> None:
        """Handle BeforeToolCallEvent - start tool call.

        BeforeToolCallEvent fields: agent, selected_tool, tool_use (ToolUse TypedDict),
        invocation_state, cancel_tool.
        ToolUse keys: name, toolUseId, input.

        Args:
            event: BeforeToolCallEvent from Strands SDK
        """
        try:
            # tool_use is a ToolUse TypedDict: {name, toolUseId, input}
            tool_use = getattr(event, "tool_use", None) or {}
            tool_name = safe_str(
                tool_use.get("name", "unknown")
                if isinstance(tool_use, dict)
                else getattr(tool_use, "name", "unknown")
            )
            tool_use_id = safe_str(
                tool_use.get("toolUseId", "")
                if isinstance(tool_use, dict)
                else getattr(tool_use, "toolUseId", "")
            )

            # input contains the tool arguments
            arguments = (
                tool_use.get("input")
                if isinstance(tool_use, dict)
                else getattr(tool_use, "input", None)
            )
            arguments_str = (
                safe_json_dumps(arguments) if arguments is not None else None
            )

            # Create tool call
            tool_call = ToolCall(
                name=tool_name,
                id=tool_use_id if tool_use_id else None,
                arguments=arguments_str,
                system="strands",
            )

            # Start the tool call
            self.handler.start_tool_call(tool_call)

            # Track by tool_use_id
            if tool_use_id:
                self._active_tool_calls[tool_use_id] = tool_call

        except Exception as e:
            _LOGGER.debug("Error in _on_before_tool_call: %s", e)

    def _on_after_tool_call(self, event: Any) -> None:
        """Handle AfterToolCallEvent - stop tool call.

        AfterToolCallEvent fields: agent, selected_tool, tool_use, invocation_state,
        result (ToolResult TypedDict), exception, cancel_message, retry.
        ToolResult keys: content, status, toolUseId.

        Args:
            event: AfterToolCallEvent from Strands SDK
        """
        try:
            # Get tool_use_id from tool_use TypedDict
            tool_use = getattr(event, "tool_use", None) or {}
            tool_use_id = safe_str(
                tool_use.get("toolUseId", "")
                if isinstance(tool_use, dict)
                else getattr(tool_use, "toolUseId", "")
            )
            if not tool_use_id:
                return

            tool_call = self._active_tool_calls.get(tool_use_id)
            if not tool_call:
                return

            exception = getattr(event, "exception", None)

            if exception:
                error_message = safe_str(exception)
                error_type = type(exception).__name__
                self.handler.fail_tool_call(
                    tool_call, Error(type=error_type, message=error_message)
                )
            else:
                # result is a ToolResult TypedDict: {content, status, toolUseId}
                result = getattr(event, "result", None)
                if result is not None:
                    tool_call.tool_result = (
                        safe_json_dumps(result)
                        if not isinstance(result, str)
                        else result
                    )

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
