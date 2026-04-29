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

"""WorkflowInvocation -- new-style workflow orchestration invocation."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from opentelemetry.metrics import MeterProvider
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.types import InputMessage, OutputMessage


class WorkflowInvocation(GenAIInvocation):
    """Represents a workflow orchestrating multiple agents and steps.

    Use ``handler.start_workflow(name=...)`` or the
    ``handler.workflow(name=...)`` context manager rather than
    constructing this directly.
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
        # Workflow-specific fields
        provider: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        name: str = "",
        workflow_type: Optional[str] = None,
        description: Optional[str] = None,
        input_messages: Optional[List[InputMessage]] = None,
        output_messages: Optional[List[OutputMessage]] = None,
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
        # Workflow-specific fields
        self.name = name
        self.workflow_type = workflow_type
        self.description = description
        self.input_messages: List[InputMessage] = (
            [] if input_messages is None else input_messages
        )
        self.output_messages: List[OutputMessage] = (
            [] if output_messages is None else output_messages
        )

        self._start()

    def _start(self) -> None:
        """Override to mark conversation root before emitter.on_start."""
        # Refresh content capture settings
        if self._capture_refresh_fn is not None:
            try:
                self._capture_refresh_fn()
            except Exception:  # pragma: no cover
                pass

        self._apply_genai_context()
        self._apply_agent_context()
        self._inherit_parent_span()

        # Auto-mark as conversation root when no parent span exists
        if self.conversation_root is None and self.parent_span is None:
            self.conversation_root = True

        self._emitter.on_start(self)
        self._push_current_span()

    # -- Upstream hooks ------------------------------------------------------

    def _get_metric_attributes(self) -> dict[str, Any]:
        from opentelemetry.util.genai.attributes import GEN_AI_FRAMEWORK

        attrs: dict[str, Any] = {}
        if self.framework is not None:
            attrs[GEN_AI_FRAMEWORK] = self.framework
        if self.name:
            attrs["gen_ai.workflow.name"] = self.name
        if self.workflow_type:
            attrs["gen_ai.workflow.type"] = self.workflow_type
        attrs["gen_ai.operation.name"] = "invoke_workflow"
        if self.provider:
            from opentelemetry.semconv._incubating.attributes import (
                gen_ai_attributes as GenAIAttributes,
            )

            attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = self.provider
        return attrs
