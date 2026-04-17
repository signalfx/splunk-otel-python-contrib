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

# type: ignore[reportUnknownDeprecated]

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Mapping,
    Sequence,
)
from urllib.parse import urlparse

from google.protobuf import json_format

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    FinishReason,
    MessagePart,
    Text,
    ToolCallResponse,
)
from opentelemetry.util.genai.utils import get_content_capturing_mode
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from google.cloud.aiplatform_v1.types import (
        content,
        tool,
    )
    from google.cloud.aiplatform_v1beta1.types import (
        content as content_v1beta1,
    )
    from google.cloud.aiplatform_v1beta1.types import (
        tool as tool_v1beta1,
    )


@dataclass(frozen=True)
class GenerateContentParams:
    model: str
    contents: (
        Sequence[content.Content] | Sequence[content_v1beta1.Content] | None
    ) = None
    system_instruction: content.Content | content_v1beta1.Content | None = None
    tools: Sequence[tool.Tool] | Sequence[tool_v1beta1.Tool] | None = None
    tool_config: tool.ToolConfig | tool_v1beta1.ToolConfig | None = None
    labels: Mapping[str, str] | None = None
    safety_settings: (
        Sequence[content.SafetySetting]
        | Sequence[content_v1beta1.SafetySetting]
        | None
    ) = None
    generation_config: (
        content.GenerationConfig | content_v1beta1.GenerationConfig | None
    ) = None


def get_server_attributes(
    endpoint: str,
) -> dict[str, AttributeValue]:
    """Get server.* attributes from the endpoint, which is a hostname with optional port e.g.
    - ``us-central1-aiplatform.googleapis.com``
    - ``us-central1-aiplatform.googleapis.com:5431``
    """
    parsed = urlparse(f"scheme://{endpoint}")

    if not parsed.hostname:
        return {}

    return {
        server_attributes.SERVER_ADDRESS: parsed.hostname,
        server_attributes.SERVER_PORT: parsed.port or 443,
    }


def get_genai_request_attributes(  # pylint: disable=too-many-branches
    params: GenerateContentParams,
    operation_name: GenAIAttributes.GenAiOperationNameValues = GenAIAttributes.GenAiOperationNameValues.CHAT,
) -> dict[str, AttributeValue]:
    model = _get_model_name(params.model)
    generation_config = params.generation_config
    attributes: dict[str, AttributeValue] = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }

    if not generation_config:
        return attributes

    # Check for optional fields
    # https://proto-plus-python.readthedocs.io/en/stable/fields.html#optional-fields
    if "temperature" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
            generation_config.temperature
        )
    if "top_p" in generation_config:
        # There is also a top_k parameter ( The maximum number of tokens to consider when sampling.),
        # but no semconv yet exists for it.
        attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] = (
            generation_config.top_p
        )
    if "max_output_tokens" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
            generation_config.max_output_tokens
        )
    if "presence_penalty" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY] = (
            generation_config.presence_penalty
        )
    if "frequency_penalty" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] = (
            generation_config.frequency_penalty
        )
    if "stop_sequences" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] = (
            generation_config.stop_sequences
        )
    if "seed" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] = (
            generation_config.seed
        )
    if "candidate_count" in generation_config:
        attributes[GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT] = (
            generation_config.candidate_count
        )
    if "response_mime_type" in generation_config:
        if generation_config.response_mime_type == "text/plain":
            attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = "text"
        elif generation_config.response_mime_type == "application/json":
            attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = "json"
        else:
            attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
                generation_config.response_mime_type
            )

    return attributes


_MODEL_STRIP_RE = re.compile(
    r"^projects/(.*)/locations/(.*)/publishers/google/models/"
)


def _get_model_name(model: str) -> str:
    return _MODEL_STRIP_RE.sub("", model)


def is_content_enabled() -> bool:
    """Check if content capturing is enabled via environment variable."""
    return get_content_capturing_mode() != ContentCapturingMode.NO_CONTENT


def convert_content_to_message_parts(
    content: content.Content | content_v1beta1.Content,
) -> list[MessagePart]:
    """Convert Vertex AI Content proto to a list of util-genai MessagePart objects.

    Only Text and ToolCallResponse parts are supported in this version.
    Unsupported part types (inline_data, file_data, function_call) are
    skipped until the corresponding util-genai types are available (HYBIM-604).
    """
    parts: list[MessagePart] = []
    for idx, part in enumerate(content.parts):
        if "function_response" in part:
            part = part.function_response
            parts.append(
                ToolCallResponse(
                    id=f"{part.name}_{idx}",
                    response=json_format.MessageToDict(part._pb.response),  # type: ignore[reportUnknownMemberType]
                )
            )
        elif "function_call" in part:
            # ToolCallRequest not yet in util-genai (HYBIM-604) — skip
            logging.debug(
                "function_call part skipped (ToolCallRequest not yet supported)"
            )
        elif "text" in part:
            parts.append(Text(content=part.text))
        elif "inline_data" in part:
            # Blob not yet in util-genai (HYBIM-604) — skip
            logging.debug("inline_data part skipped (Blob not yet supported)")
        elif "file_data" in part:
            # Uri not yet in util-genai (HYBIM-604) — skip
            logging.debug("file_data part skipped (Uri not yet supported)")
        else:
            logging.warning("Unknown part dropped from telemetry %s", part)
    return parts


def _map_finish_reason(
    finish_reason: content.Candidate.FinishReason
    | content_v1beta1.Candidate.FinishReason
    | None,
) -> FinishReason | str:
    if finish_reason is None:
        return "error"
    EnumType = type(finish_reason)  # pylint: disable=invalid-name
    if (
        finish_reason is EnumType.FINISH_REASON_UNSPECIFIED
        or finish_reason is EnumType.OTHER
    ):
        return "error"
    if finish_reason is EnumType.STOP:
        return "stop"
    if finish_reason is EnumType.MAX_TOKENS:
        return "length"

    # If there is no 1:1 mapping to an OTel preferred enum value, use the exact vertex reason
    return finish_reason.name
