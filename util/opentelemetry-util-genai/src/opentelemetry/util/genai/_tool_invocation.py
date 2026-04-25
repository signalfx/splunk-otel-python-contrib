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

"""ToolInvocation -- new-style tool/function call invocation."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from opentelemetry.metrics import MeterProvider

from opentelemetry.util.genai._invocation import GenAIInvocation


class ToolInvocation(GenAIInvocation):
    """Represents a tool call invocation for execute_tool span tracking.

    Use ``handler.start_tool(name)`` or the ``handler.tool(name)``
    context manager rather than constructing this directly.
    """

    def __init__(
        self,
        *,
        # Components (from handler)
        emitter: Any,
        agent_context_stack: List[Tuple[str, Optional[str]]],
        completion_callbacks: list,
        sampler_fn: Callable[[Optional[int]], bool],
        meter_provider: Optional[MeterProvider] = None,
        capture_refresh_fn: Optional[Callable[[], None]] = None,
        # Tool-specific fields
        provider: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        name: str = "",
        arguments: Any = None,
        id: Optional[str] = None,
        tool_type: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_result: Optional[Any] = None,
        error_type: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            emitter=emitter,
            agent_context_stack=agent_context_stack,
            completion_callbacks=completion_callbacks,
            sampler_fn=sampler_fn,
            meter_provider=meter_provider,
            capture_refresh_fn=capture_refresh_fn,
            provider=provider,
            framework=framework,
            system=system,
            attributes=attributes,
        )
        # Tool-specific fields
        self.name = name
        self.arguments = arguments
        self.id = id
        self.type = "tool_call"
        self.tool_type = tool_type
        self.tool_description = tool_description
        self.tool_result = tool_result
        self.error_type = error_type

        self._start()

    # -- Upstream hooks ------------------------------------------------------

    def _get_metric_attributes(self) -> dict[str, Any]:
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )

        from opentelemetry.util.genai.attributes import GEN_AI_FRAMEWORK

        attrs: dict[str, Any] = {}
        if self.framework is not None:
            attrs[GEN_AI_FRAMEWORK] = self.framework
        if self.provider:
            attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = self.provider
        attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] = (
            GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value
        )
        if self.name:
            attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] = self.name
        if self.agent_name:
            attrs[GenAIAttributes.GEN_AI_AGENT_NAME] = self.agent_name
        if self.agent_id:
            attrs[GenAIAttributes.GEN_AI_AGENT_ID] = self.agent_id
        return attrs
