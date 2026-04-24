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
