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

"""InferenceInvocation -- new-style LLM chat/completion invocation."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from opentelemetry.metrics import MeterProvider
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.types import InputMessage, OutputMessage


class InferenceInvocation(GenAIInvocation):
    """Represents a single LLM chat/completion call.

    Use ``handler.start_inference(provider)`` or the
    ``handler.inference(provider)`` context manager rather than
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
        # LLM-specific fields
        provider: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        request_model: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
        input_messages: Optional[List[InputMessage]] = None,
        output_messages: Optional[List[OutputMessage]] = None,
        operation: Optional[str] = None,
        response_model_name: Optional[str] = None,
        response_id: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        request_functions: Optional[list] = None,
        tool_definitions: Optional[str] = None,
        request_temperature: Optional[float] = None,
        request_top_p: Optional[float] = None,
        request_top_k: Optional[int] = None,
        request_frequency_penalty: Optional[float] = None,
        request_presence_penalty: Optional[float] = None,
        request_stop_sequences: Optional[List[str]] = None,
        request_max_tokens: Optional[int] = None,
        request_choice_count: Optional[int] = None,
        request_seed: Optional[int] = None,
        request_encoding_formats: Optional[List[str]] = None,
        output_type: Optional[str] = None,
        response_finish_reasons: Optional[List[str]] = None,
        request_service_tier: Optional[str] = None,
        response_service_tier: Optional[str] = None,
        response_system_fingerprint: Optional[str] = None,
        security_event_id: Optional[str] = None,
        request_stream: Optional[bool] = None,
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
        # LLM-specific fields
        self.request_model = request_model
        self.server_address = server_address
        self.server_port = server_port
        self.input_messages: List[InputMessage] = (
            [] if input_messages is None else input_messages
        )
        self.output_messages: List[OutputMessage] = (
            [] if output_messages is None else output_messages
        )
        self.operation = operation or "chat"
        self.response_model_name = response_model_name
        self.response_id = response_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.request_functions: list = (
            [] if request_functions is None else request_functions
        )
        self.tool_definitions = tool_definitions
        self.request_temperature = request_temperature
        self.request_top_p = request_top_p
        self.request_top_k = request_top_k
        self.request_frequency_penalty = request_frequency_penalty
        self.request_presence_penalty = request_presence_penalty
        self.request_stop_sequences: List[str] = (
            [] if request_stop_sequences is None else request_stop_sequences
        )
        self.request_max_tokens = request_max_tokens
        self.request_choice_count = request_choice_count
        self.request_seed = request_seed
        self.request_encoding_formats: List[str] = (
            []
            if request_encoding_formats is None
            else request_encoding_formats
        )
        self.output_type = output_type
        self.response_finish_reasons: List[str] = (
            [] if response_finish_reasons is None else response_finish_reasons
        )
        self.request_service_tier = request_service_tier
        self.response_service_tier = response_service_tier
        self.response_system_fingerprint = response_system_fingerprint
        self.security_event_id = security_event_id
        self.request_stream = request_stream

        self._start()

    # -- Upstream hooks ------------------------------------------------------

    def _get_metric_attributes(self) -> dict[str, Any]:
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        from opentelemetry.semconv.attributes import (
            server_attributes as ServerAttributes,
        )
        from opentelemetry.util.genai.attributes import GEN_AI_FRAMEWORK

        attrs: dict[str, Any] = {}
        if self.framework is not None:
            attrs[GEN_AI_FRAMEWORK] = self.framework
        if self.provider:
            attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = self.provider
        if self.operation:
            attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] = self.operation
        if self.request_model:
            attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] = self.request_model
        if self.response_model_name:
            attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = (
                self.response_model_name
            )
        if self.server_address:
            attrs[ServerAttributes.SERVER_ADDRESS] = self.server_address
        if self.server_port:
            attrs[ServerAttributes.SERVER_PORT] = self.server_port
        if self.agent_name:
            attrs[GenAIAttributes.GEN_AI_AGENT_NAME] = self.agent_name
        if self.agent_id:
            attrs[GenAIAttributes.GEN_AI_AGENT_ID] = self.agent_id
        return attrs

    def _get_metric_token_counts(self) -> dict[str, int]:
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )

        counts: dict[str, int] = {}
        if self.input_tokens is not None:
            counts[GenAIAttributes.GenAiTokenTypeValues.INPUT.value] = (
                self.input_tokens
            )
        if self.output_tokens is not None:
            counts[GenAIAttributes.GenAiTokenTypeValues.OUTPUT.value] = (
                self.output_tokens
            )
        return counts
