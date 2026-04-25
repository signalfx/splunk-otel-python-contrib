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

"""EmbeddingInvocation -- new-style embedding model invocation."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from opentelemetry.metrics import MeterProvider

from opentelemetry.util.genai._invocation import GenAIInvocation


class EmbeddingInvocation(GenAIInvocation):
    """Represents a single embedding model invocation.

    Use ``handler.start_embedding(provider)`` or the
    ``handler.embedding(provider)`` context manager rather than
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
        # Embedding-specific fields
        provider: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        operation_name: Optional[str] = None,
        request_model: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
        input_texts: Optional[list[str]] = None,
        dimension_count: Optional[int] = None,
        input_tokens: Optional[int] = None,
        encoding_formats: Optional[list[str]] = None,
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
        # Embedding-specific fields
        self.operation_name = operation_name or "embeddings"
        self.request_model = request_model or ""
        self.server_address = server_address
        self.server_port = server_port
        self.input_texts: list[str] = [] if input_texts is None else input_texts
        self.dimension_count = dimension_count
        self.input_tokens = input_tokens
        self.encoding_formats: list[str] = (
            [] if encoding_formats is None else encoding_formats
        )
        self.error_type = error_type

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
        if self.operation_name:
            attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] = self.operation_name
        if self.request_model:
            attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] = self.request_model
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
            counts[GenAIAttributes.GenAiTokenTypeValues.INPUT.value] = self.input_tokens
        return counts
