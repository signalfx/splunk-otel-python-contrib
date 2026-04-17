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

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    MutableSequence,
)

from opentelemetry.instrumentation.vertexai.utils import (
    GenerateContentParams,
    _map_finish_reason,
    convert_content_to_message_parts,
    get_genai_request_attributes,
    get_server_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.util.genai.handler import (
    Error as InvocationError,
)
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
)

if TYPE_CHECKING:
    from google.cloud.aiplatform_v1.types import (
        content,
        prediction_service,
    )
    from google.cloud.aiplatform_v1beta1.types import (
        content as content_v1beta1,
    )
    from google.cloud.aiplatform_v1beta1.types import (
        prediction_service as prediction_service_v1beta1,
    )


# Use parameter signature from
# https://github.com/googleapis/python-aiplatform/blob/v1.76.0/google/cloud/aiplatform_v1/services/prediction_service/client.py#L2088
# to handle named vs positional args robustly
def _extract_params(
    request: prediction_service.GenerateContentRequest
    | prediction_service_v1beta1.GenerateContentRequest
    | dict[Any, Any]
    | None = None,
    *,
    model: str | None = None,
    contents: MutableSequence[content.Content]
    | MutableSequence[content_v1beta1.Content]
    | None = None,
    **_kwargs: Any,
) -> GenerateContentParams:
    # Request vs the named parameters are mututally exclusive or the RPC will fail
    if not request:
        return GenerateContentParams(
            model=model or "",
            contents=contents,
        )

    if isinstance(request, dict):
        return GenerateContentParams(**request)

    return GenerateContentParams(
        model=request.model,
        contents=request.contents,
        system_instruction=request.system_instruction,
        tools=request.tools,
        tool_config=request.tool_config,
        labels=request.labels,
        safety_settings=request.safety_settings,
        generation_config=request.generation_config,
    )


def _build_invocation(
    params: GenerateContentParams,
    api_endpoint: str,
    capture_content: bool,
) -> LLMInvocation:
    """Build an LLMInvocation from Vertex AI request parameters."""
    request_attributes = get_genai_request_attributes(params)
    server_attrs = get_server_attributes(api_endpoint)

    # Build input messages
    input_messages: list[InputMessage] = []
    if capture_content:
        # Vertex AI uses a dedicated system_instruction field (equivalent to OpenAI's
        # role="system") but its Content proto only supports role="user"|"model".
        # We set role="system" so the emitter routes it to gen_ai.system_instructions.
        if params.system_instruction:
            input_messages.append(
                InputMessage(
                    role="system",
                    parts=convert_content_to_message_parts(
                        params.system_instruction
                    ),
                )
            )
        if params.contents:
            for c in params.contents:
                input_messages.append(
                    InputMessage(
                        role=c.role or "user",
                        parts=convert_content_to_message_parts(c),
                    )
                )

    invocation = LLMInvocation(
        request_model=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_MODEL, ""
        ),
        input_messages=input_messages,
        provider="vertex_ai",
        framework="google-cloud-aiplatform",
        server_address=server_attrs.get(ServerAttributes.SERVER_ADDRESS),
        server_port=server_attrs.get(ServerAttributes.SERVER_PORT),
        request_temperature=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE
        ),
        request_top_p=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_TOP_P
        ),
        request_max_tokens=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS
        ),
        request_presence_penalty=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY
        ),
        request_frequency_penalty=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY
        ),
        request_stop_sequences=list(
            request_attributes.get(
                GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, []
            )
        ),
        request_seed=request_attributes.get(
            GenAIAttributes.GEN_AI_REQUEST_SEED
        ),
    )

    # Propagate extra attributes that don't map to LLMInvocation fields
    if GenAIAttributes.GEN_AI_OUTPUT_TYPE in request_attributes:
        invocation.attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
            request_attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE]
        )

    if capture_content and params.tools:
        from google.protobuf import json_format as _jf

        tool_defs = [_jf.MessageToDict(t._pb) for t in params.tools]  # type: ignore[union-attr]
        invocation.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS] = (
            json.dumps(tool_defs)
        )

    return invocation


def _apply_response_to_invocation(
    invocation: LLMInvocation,
    response: prediction_service.GenerateContentResponse
    | prediction_service_v1beta1.GenerateContentResponse,
    capture_content: bool,
) -> None:
    """Apply response data to an existing LLMInvocation."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        invocation.input_tokens = response.usage_metadata.prompt_token_count
        invocation.output_tokens = (
            response.usage_metadata.candidates_token_count
        )

    model = getattr(response, "model_version", None)
    if model:
        invocation.response_model_name = model

    finish_reasons = []
    output_messages: list[OutputMessage] = []
    for candidate in response.candidates:
        fr = _map_finish_reason(candidate.finish_reason)
        finish_reasons.append(fr)
        parts = []
        if capture_content:
            parts = convert_content_to_message_parts(candidate.content)
        output_messages.append(
            OutputMessage(
                role=candidate.content.role or "model",
                parts=parts,
                finish_reason=fr,
            )
        )

    invocation.response_finish_reasons = finish_reasons
    invocation.output_messages = output_messages


def generate_content(capture_content: bool, handler: TelemetryHandler):
    """Wrap the sync `generate_content` method to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        params = _extract_params(*args, **kwargs)
        api_endpoint: str = instance.api_endpoint  # type: ignore[reportUnknownMemberType]
        invocation = _build_invocation(params, api_endpoint, capture_content)
        handler.start_llm(invocation)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as error:
            try:  # pragma: no cover - defensive
                handler.fail_llm(
                    invocation,
                    InvocationError(message=str(error), type=type(error)),
                )
            except Exception:
                pass
            raise

        try:
            _apply_response_to_invocation(
                invocation, response, capture_content
            )
            handler.stop_llm(invocation)
        except Exception as error:  # pragma: no cover - defensive
            try:
                handler.fail_llm(
                    invocation,
                    InvocationError(message=str(error), type=type(error)),
                )
            except Exception:
                pass

        return response

    return traced_method


def agenerate_content(capture_content: bool, handler: TelemetryHandler):
    """Wrap the async `generate_content` method to trace it."""

    async def traced_method(wrapped, instance, args, kwargs):
        params = _extract_params(*args, **kwargs)
        api_endpoint: str = instance.api_endpoint  # type: ignore[reportUnknownMemberType]
        invocation = _build_invocation(params, api_endpoint, capture_content)
        handler.start_llm(invocation)

        try:
            response = await wrapped(*args, **kwargs)
        except Exception as error:
            try:  # pragma: no cover - defensive
                handler.fail_llm(
                    invocation,
                    InvocationError(message=str(error), type=type(error)),
                )
            except Exception:
                pass
            raise

        try:
            _apply_response_to_invocation(
                invocation, response, capture_content
            )
            handler.stop_llm(invocation)
        except Exception as error:  # pragma: no cover - defensive
            try:
                handler.fail_llm(
                    invocation,
                    InvocationError(message=str(error), type=type(error)),
                )
            except Exception:
                pass

        return response

    return traced_method
