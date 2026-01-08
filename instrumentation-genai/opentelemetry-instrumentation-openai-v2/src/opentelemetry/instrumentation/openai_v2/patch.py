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


import asyncio
import inspect
from timeit import default_timer
from typing import Any, Iterable, Optional

from openai import Stream

from opentelemetry._logs import Logger, LogRecord
from opentelemetry import context as context_api
from opentelemetry.context import get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.util.genai.handler import (
    Error as InvocationError,
)
from opentelemetry.util.genai.types import (
    EmbeddingInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.types import (
    ToolCall as GenAIToolCall,
)

from .instruments import Instruments
from .utils import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    choice_to_event,
    get_llm_request_attributes,
    handle_span_exception,
    is_streaming,
    message_to_event,
    set_span_attribute,
    value_is_set,
)


def _normalize_stop_sequences(stop_values: Any) -> list[str]:
    if stop_values is None:
        return []
    if isinstance(stop_values, str):
        return [stop_values]
    if isinstance(stop_values, Iterable):
        return [
            value
            for value in stop_values
            if isinstance(value, str) and value_is_set(value)
        ]
    return []


def _to_text_parts(content: Any, capture_content: bool) -> list[Text]:
    if content is None:
        return []

    def _text_value(item: Any) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("text") or item.get("content")
        if hasattr(item, "text"):
            return getattr(item, "text")
        try:
            return str(item)
        except Exception:  # pragma: no cover - defensive
            return None

    if isinstance(content, str):
        return [Text(content=content if capture_content else "")]
    if isinstance(content, Iterable) and not isinstance(content, dict):
        parts: list[Text] = []
        for item in content:
            text_value = _text_value(item)
            if text_value is None:
                continue
            parts.append(Text(content=text_value if capture_content else ""))
        return parts

    text_value = _text_value(content)
    return (
        [Text(content=text_value if capture_content else "")]
        if text_value is not None
        else []
    )


def _build_input_messages(
    messages: Iterable[Any], capture_content: bool
) -> list[InputMessage]:
    input_messages: list[InputMessage] = []
    for message in messages or []:
        role = getattr(message, "role", None) or getattr(message, "type", None)
        if role is None and isinstance(message, dict):
            role = message.get("role") or message.get("type")
        role = role or "user"
        parts = _to_text_parts(
            getattr(message, "content", None)
            if not isinstance(message, dict)
            else message.get("content"),
            capture_content,
        )
        if not parts:
            parts = [Text(content="")]
        input_messages.append(InputMessage(role=role, parts=parts))
    return input_messages


def _build_chat_invocation(
    kwargs: dict[str, Any],
    capture_content: bool,
    attributes: Optional[dict[str, Any]] = None,
) -> LLMInvocation:
    def _clean(value: Any) -> Any:
        return value if value_is_set(value) else None

    response_format_value = kwargs.get("response_format")
    response_format = response_format_value
    if isinstance(response_format_value, dict):
        response_format = response_format_value.get("type")

    stop_sequences = _normalize_stop_sequences(kwargs.get("stop"))
    seed = _clean(kwargs.get("seed"))
    # Only add choice_count if it's a meaningful value (not 1)
    choice_count = kwargs.get("n")
    request_choice_count = None
    if isinstance(choice_count, int) and choice_count != 1:
        request_choice_count = choice_count

    invocation = LLMInvocation(
        request_model=_clean(kwargs.get("model", "")) or "",
        input_messages=_build_input_messages(
            kwargs.get("messages", []), capture_content
        ),
        provider="openai",
        framework="openai-sdk",
        system=GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
        request_temperature=_clean(kwargs.get("temperature")),
        request_top_p=_clean(kwargs.get("p") or kwargs.get("top_p")),
        request_max_tokens=_clean(kwargs.get("max_tokens")),
        request_presence_penalty=_clean(kwargs.get("presence_penalty")),
        request_frequency_penalty=_clean(kwargs.get("frequency_penalty")),
        request_choice_count=request_choice_count,
        request_seed=seed,
        request_functions=list(kwargs.get("tools", []) or []),
        request_stop_sequences=stop_sequences,
        request_encoding_formats=[response_format]
        if value_is_set(response_format)
        else [],
    )

    if response_format:
        invocation.attributes[
            GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT
        ] = response_format

    if seed is not None:
        invocation.attributes[GenAIAttributes.GEN_AI_OPENAI_REQUEST_SEED] = (
            seed
        )

    if attributes:
        if ServerAttributes.SERVER_ADDRESS in attributes:
            invocation.server_address = attributes[
                ServerAttributes.SERVER_ADDRESS
            ]
        if ServerAttributes.SERVER_PORT in attributes:
            invocation.server_port = attributes[ServerAttributes.SERVER_PORT]

    service_tier = kwargs.get("service_tier")
    extra_body = kwargs.get("extra_body")
    if service_tier is None and isinstance(extra_body, dict):
        service_tier = extra_body.get("service_tier")
    if value_is_set(service_tier) and service_tier != "auto":
        invocation.request_service_tier = service_tier

    return invocation


def _build_output_messages_from_response(
    result: Any, capture_content: bool
) -> list[OutputMessage]:
    output_messages: list[OutputMessage] = []
    for choice in getattr(result, "choices", []) or []:
        message = getattr(choice, "message", None)
        role = getattr(message, "role", None) if message else None
        parts: list[Any] = []
        content = getattr(message, "content", None) if message else None
        if content is not None:
            parts.extend(_to_text_parts(content, capture_content))

        # Include tool calls in output_messages parts
        tool_calls = None
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        elif message is not None:
            tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for tool_call in tool_calls:
                genai_tool_call, _ = _build_tool_call_invocation(
                    tool_call, capture_content
                )
                parts.append(genai_tool_call)

        finish_reason = getattr(choice, "finish_reason", None) or "error"
        output_messages.append(
            OutputMessage(
                role=role or "assistant",
                parts=parts,
                finish_reason=finish_reason,
            )
        )
    return output_messages


def _apply_chat_response_to_invocation(
    invocation: LLMInvocation, result: Any, capture_content: bool
) -> None:
    if getattr(result, "id", None):
        invocation.response_id = result.id
    if getattr(result, "model", None):
        invocation.response_model_name = result.model
    if getattr(result, "service_tier", None):
        invocation.response_service_tier = result.service_tier
    if getattr(result, "system_fingerprint", None):
        invocation.response_system_fingerprint = result.system_fingerprint
    if getattr(result, "usage", None):
        invocation.input_tokens = result.usage.prompt_tokens
        invocation.output_tokens = getattr(
            result.usage, "completion_tokens", None
        )

    finish_reasons = []
    for choice in getattr(result, "choices", []) or []:
        finish_reasons.append(choice.finish_reason or "error")
    invocation.response_finish_reasons = finish_reasons
    invocation.output_messages = _build_output_messages_from_response(
        result, capture_content
    )


def _parse_response(result: Any) -> Any:
    """Unwrap LegacyAPIResponse objects returned by with_raw_response helpers."""
    if hasattr(result, "parse"):
        return result.parse()
    return result


def _build_tool_call_invocation(
    tool_call: Any, capture_content: bool
) -> tuple[GenAIToolCall, str]:
    """Normalize to genai-util ToolCall and capture tool type."""
    tool_type = getattr(tool_call, "type", None)
    function = getattr(tool_call, "function", None)
    if isinstance(tool_call, dict):
        tool_type = tool_call.get("type", tool_type)
        function = tool_call.get("function", function)

    function_name = None
    arguments = None
    description = None
    if isinstance(function, dict):
        function_name = function.get("name")
        arguments = function.get("arguments")
        description = function.get("description")
    else:
        function_name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        description = getattr(function, "description", None)

    if not capture_content:
        arguments = None

    tool_call_id = getattr(tool_call, "id", None)
    if isinstance(tool_call, dict):
        tool_call_id = tool_call.get("id", tool_call_id)

    tool_call_type = tool_type

    genai_tool_call = GenAIToolCall(
        name=function_name or "unnamed_tool_call",
        id=tool_call_id,
        arguments=arguments,
        provider=GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    )
    genai_tool_call.attributes[GenAIAttributes.GEN_AI_TOOL_NAME] = (
        function_name or ""
    )
    genai_tool_call.attributes[GenAIAttributes.GEN_AI_TOOL_TYPE] = (
        tool_call_type
    )
    if tool_call_id:
        genai_tool_call.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = (
            tool_call_id
        )
    if description:
        genai_tool_call.attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] = (
            description
        )

    return genai_tool_call, tool_call_type


def _tool_call_body(
    tool_call: GenAIToolCall, tool_type: str, capture_content: bool
) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool_call.name}
    if capture_content and tool_call.arguments is not None:
        function["arguments"] = tool_call.arguments

    body: dict[str, Any] = {
        "type": tool_type,
        "function": function,
    }
    if tool_call.id is not None:
        body["id"] = tool_call.id
    return body


def _normalize_input_texts(input_val: Any) -> list[str]:
    """Normalize embedding input to a list of strings."""
    if input_val is None:
        return []
    if isinstance(input_val, str):
        return [input_val]
    if isinstance(input_val, list):
        return [str(item) for item in input_val]
    return [str(input_val)]


def _build_embedding_invocation(
    kwargs: dict[str, Any], attributes: Optional[dict[str, Any]] = None
) -> EmbeddingInvocation:
    invocation = EmbeddingInvocation(
        request_model=kwargs.get("model", "") or "",
        input_texts=_normalize_input_texts(kwargs.get("input")),
        provider="openai",
        framework="openai-sdk",
        system=GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    )

    if "dimensions" in kwargs and value_is_set(kwargs.get("dimensions")):
        invocation.dimension_count = kwargs.get("dimensions")

    encoding_format = kwargs.get("encoding_format")
    if value_is_set(encoding_format):
        invocation.encoding_formats = [encoding_format]

    if attributes:
        if ServerAttributes.SERVER_ADDRESS in attributes:
            invocation.server_address = attributes[
                ServerAttributes.SERVER_ADDRESS
            ]
        if ServerAttributes.SERVER_PORT in attributes:
            invocation.server_port = attributes[ServerAttributes.SERVER_PORT]

    return invocation


def _apply_embedding_response_to_invocation(
    invocation: EmbeddingInvocation, result: Any
) -> None:
    if getattr(result, "usage", None):
        invocation.input_tokens = result.usage.prompt_tokens

    data = getattr(result, "data", None)
    if not data:
        return

    first_embedding = data[0] if len(data) > 0 else None
    if first_embedding is None:
        return

    embedding_vec = getattr(first_embedding, "embedding", None)
    if embedding_vec is not None:
        try:
            invocation.dimension_count = len(embedding_vec)
        except Exception:  # pragma: no cover - defensive
            pass


def chat_completions_create(
    logger: Logger,
    instruments: Instruments,
    capture_content: bool,
    handler,
):
    """Wrap the `create` method of the `ChatCompletion` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        # Check if instrumentation is suppressed (e.g., by LangChain)
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        span_attributes = {**get_llm_request_attributes(kwargs, instance)}
        invocation = _build_chat_invocation(
            kwargs, capture_content, span_attributes
        )
        handler.start_llm(invocation)
        span = getattr(invocation, "span", None)

        for message in kwargs.get("messages", []):
            logger.emit(message_to_event(message, capture_content))

        start = default_timer()
        result = None
        parsed_result = None
        error_type = None
        try:
            result = wrapped(*args, **kwargs)
            parsed_result = _parse_response(result)
            if is_streaming(kwargs):
                return StreamWrapper(
                    parsed_result,
                    invocation,
                    logger,
                    capture_content,
                    handler,
                )

            if span and span.is_recording():
                _set_response_attributes(
                    span, parsed_result, capture_content, handler
                )
            for choice in getattr(parsed_result, "choices", []):
                logger.emit(choice_to_event(choice, capture_content))

            _apply_chat_response_to_invocation(
                invocation, parsed_result, capture_content
            )
            handler.stop_llm(invocation)
            return result

        except Exception as error:
            error_type = type(error).__qualname__
            handler.fail_llm(
                invocation,
                InvocationError(message=str(error), type=type(error)),
            )
            if span:
                handle_span_exception(span, error)
            raise
        finally:
            duration = max((default_timer() - start), 0)
            _record_metrics(
                instruments,
                duration,
                parsed_result or result,
                span_attributes,
                error_type,
                GenAIAttributes.GenAiOperationNameValues.CHAT.value,
            )

    return traced_method


def async_chat_completions_create(
    logger: Logger,
    instruments: Instruments,
    capture_content: bool,
    handler,
):
    """Wrap the `create` method of the `AsyncChatCompletion` class to trace it."""

    async def traced_method(wrapped, instance, args, kwargs):
        # Check if instrumentation is suppressed (e.g., by LangChain)
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span_attributes = {**get_llm_request_attributes(kwargs, instance)}
        invocation = _build_chat_invocation(
            kwargs, capture_content, span_attributes
        )
        handler.start_llm(invocation)
        span = getattr(invocation, "span", None)

        for message in kwargs.get("messages", []):
            logger.emit(message_to_event(message, capture_content))

        start = default_timer()
        result = None
        parsed_result = None
        error_type = None
        try:
            result = await wrapped(*args, **kwargs)
            parsed_result = _parse_response(result)
            if is_streaming(kwargs):
                return StreamWrapper(
                    parsed_result,
                    invocation,
                    logger,
                    capture_content,
                    handler,
                )

            if span and span.is_recording():
                _set_response_attributes(
                    span, parsed_result, capture_content, handler
                )
            for choice in getattr(parsed_result, "choices", []):
                logger.emit(choice_to_event(choice, capture_content))

            _apply_chat_response_to_invocation(
                invocation, parsed_result, capture_content
            )
            handler.stop_llm(invocation)
            return result

        except Exception as error:
            error_type = type(error).__qualname__
            handler.fail_llm(
                invocation,
                InvocationError(message=str(error), type=type(error)),
            )
            if span:
                handle_span_exception(span, error)
            raise
        finally:
            duration = max((default_timer() - start), 0)
            _record_metrics(
                instruments,
                duration,
                parsed_result or result,
                span_attributes,
                error_type,
                GenAIAttributes.GenAiOperationNameValues.CHAT.value,
            )

    return traced_method


def embeddings_create(
    instruments: Instruments,
    capture_content: bool,
    handler,
):
    """Wrap the `create` method of the `Embeddings` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        # Check if instrumentation is suppressed (e.g., by LangChain)
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        span_attributes = get_llm_request_attributes(
            kwargs,
            instance,
            GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        )
        invocation = _build_embedding_invocation(kwargs, span_attributes)
        handler.start_embedding(invocation)
        span = getattr(invocation, "span", None)

        start = default_timer()
        result = None
        parsed_result = None
        error_type = None

        try:
            result = wrapped(*args, **kwargs)
            parsed_result = _parse_response(result)

            if span and span.is_recording():
                _set_embeddings_response_attributes(
                    span,
                    parsed_result,
                    capture_content,
                    kwargs.get("input", ""),
                )

            _apply_embedding_response_to_invocation(invocation, parsed_result)
            handler.stop_embedding(invocation)
            return result

        except Exception as error:
            error_type = type(error).__qualname__
            handler.fail_embedding(
                invocation,
                InvocationError(message=str(error), type=type(error)),
            )
            if span:
                handle_span_exception(span, error)
            raise

        finally:
            duration = max((default_timer() - start), 0)
            _record_metrics(
                instruments,
                duration,
                parsed_result or result,
                span_attributes,
                error_type,
                GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
            )

    return traced_method


def async_embeddings_create(
    instruments: Instruments,
    capture_content: bool,
    handler,
):
    """Wrap the `create` method of the `AsyncEmbeddings` class to trace it."""

    async def traced_method(wrapped, instance, args, kwargs):
        # Check if instrumentation is suppressed (e.g., by LangChain)
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span_attributes = get_llm_request_attributes(
            kwargs,
            instance,
            GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        )
        invocation = _build_embedding_invocation(kwargs, span_attributes)
        handler.start_embedding(invocation)
        span = getattr(invocation, "span", None)

        start = default_timer()
        result = None
        parsed_result = None
        error_type = None

        try:
            result = await wrapped(*args, **kwargs)
            parsed_result = _parse_response(result)

            if span and span.is_recording():
                _set_embeddings_response_attributes(
                    span,
                    parsed_result,
                    capture_content,
                    kwargs.get("input", ""),
                )

            _apply_embedding_response_to_invocation(invocation, parsed_result)
            handler.stop_embedding(invocation)
            return result

        except Exception as error:
            error_type = type(error).__qualname__
            handler.fail_embedding(
                invocation,
                InvocationError(message=str(error), type=type(error)),
            )
            if span:
                handle_span_exception(span, error)
            raise

        finally:
            duration = max((default_timer() - start), 0)
            _record_metrics(
                instruments,
                duration,
                parsed_result or result,
                span_attributes,
                error_type,
                GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
            )

    return traced_method


def _get_embeddings_span_name(span_attributes):
    """Get span name for embeddings operations."""
    return f"{span_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {span_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]}"


def _record_metrics(
    instruments: Instruments,
    duration: float,
    result,
    request_attributes: dict,
    error_type: Optional[str],
    operation_name: str,
):
    common_attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name,
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: request_attributes[
            GenAIAttributes.GEN_AI_REQUEST_MODEL
        ],
    }

    if "gen_ai.embeddings.dimension.count" in request_attributes:
        common_attributes["gen_ai.embeddings.dimension.count"] = (
            request_attributes["gen_ai.embeddings.dimension.count"]
        )

    if error_type:
        common_attributes["error.type"] = error_type

    if result and getattr(result, "model", None):
        common_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = result.model

    if result and getattr(result, "service_tier", None):
        common_attributes[
            GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
        ] = result.service_tier

    if result and getattr(result, "system_fingerprint", None):
        common_attributes["gen_ai.openai.response.system_fingerprint"] = (
            result.system_fingerprint
        )

    if ServerAttributes.SERVER_ADDRESS in request_attributes:
        common_attributes[ServerAttributes.SERVER_ADDRESS] = (
            request_attributes[ServerAttributes.SERVER_ADDRESS]
        )

    if ServerAttributes.SERVER_PORT in request_attributes:
        common_attributes[ServerAttributes.SERVER_PORT] = request_attributes[
            ServerAttributes.SERVER_PORT
        ]

    instruments.operation_duration_histogram.record(
        duration,
        attributes=common_attributes,
    )

    if result and getattr(result, "usage", None):
        # Always record input tokens
        input_attributes = {
            **common_attributes,
            GenAIAttributes.GEN_AI_TOKEN_TYPE: GenAIAttributes.GenAiTokenTypeValues.INPUT.value,
        }
        instruments.token_usage_histogram.record(
            result.usage.prompt_tokens,
            attributes=input_attributes,
        )

        # For embeddings, don't record output tokens as all tokens are input tokens
        if (
            operation_name
            != GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
        ):
            output_attributes = {
                **common_attributes,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: GenAIAttributes.GenAiTokenTypeValues.COMPLETION.value,
            }
            instruments.token_usage_histogram.record(
                result.usage.completion_tokens, attributes=output_attributes
            )


def _set_response_attributes(
    span, result, capture_content: bool, handler=None
):
    if getattr(result, "model", None):
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, result.model
        )

    if getattr(result, "choices", None):
        finish_reasons = []
        for choice in result.choices:
            finish_reasons.append(choice.finish_reason or "error")

        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
            finish_reasons,
        )

    if getattr(result, "id", None):
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, result.id)

    if getattr(result, "service_tier", None):
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
            result.service_tier,
        )

    # Get the usage
    if getattr(result, "usage", None):
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            result.usage.prompt_tokens,
        )
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            result.usage.completion_tokens,
        )


def _set_embeddings_response_attributes(
    span: Span,
    result: Any,
    capture_content: bool,
    input_text: str,
):
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, result.model
    )

    # Set embeddings dimensions if we can determine it from the response
    if getattr(result, "data", None) and len(result.data) > 0:
        first_embedding = result.data[0]
        if getattr(first_embedding, "embedding", None):
            set_span_attribute(
                span,
                "gen_ai.embeddings.dimension.count",
                len(first_embedding.embedding),
            )

    # Get the usage
    if getattr(result, "usage", None):
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            result.usage.prompt_tokens,
        )
        # Don't set output tokens for embeddings as all tokens are input tokens


class ToolCallBuffer:
    """Accumulates streaming tool call chunks without creating spans.

    Tool call spans (execute_tool) should be created by user code or higher-level
    frameworks (e.g., LangChain) that can observe actual tool execution, not during
    response parsing.
    """

    def __init__(
        self,
        index: int,
        tool_call: GenAIToolCall,
        tool_type: str,
        capture_content: bool,
    ):
        self.index = index
        self.tool_call = tool_call
        self.tool_type = tool_type
        self._capture_content = capture_content
        self._argument_chunks: list[str] = []
        self._finalized = False

    def append_arguments(self, arguments):
        if not self._capture_content or arguments is None:
            return
        self._argument_chunks.append(arguments)

    def finalize(self) -> tuple[GenAIToolCall, str]:
        if self._finalized:
            return self.tool_call, self.tool_type

        if self._capture_content and self._argument_chunks:
            self.tool_call.arguments = "".join(self._argument_chunks)
        else:
            if not self._capture_content:
                self.tool_call.arguments = None

        self._finalized = True
        return self.tool_call, self.tool_type


class ChoiceBuffer:
    def __init__(
        self,
        index: int,
        capture_content: bool,
    ):
        self.index = index
        self.finish_reason = None
        self.text_content = []
        self.tool_calls_buffers = []
        self._capture_content = capture_content

    def append_text_content(self, content):
        self.text_content.append(content)

    def append_tool_call(self, tool_call):
        idx = tool_call.index
        # make sure we have enough tool call buffers
        for _ in range(len(self.tool_calls_buffers), idx + 1):
            self.tool_calls_buffers.append(None)

        if not self.tool_calls_buffers[idx]:
            genai_tool_call, tool_type = _build_tool_call_invocation(
                tool_call, self._capture_content
            )
            self.tool_calls_buffers[idx] = ToolCallBuffer(
                idx,
                genai_tool_call,
                tool_type,
                self._capture_content,
            )
            # Preserve initial arguments from the first chunk
            if hasattr(tool_call.function, "arguments"):
                self.tool_calls_buffers[idx].append_arguments(
                    tool_call.function.arguments
                )
        else:
            if hasattr(tool_call.function, "arguments"):
                self.tool_calls_buffers[idx].append_arguments(
                    tool_call.function.arguments
                )


class StreamWrapper:
    span: Span
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    service_tier: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def __init__(
        self,
        stream: Stream,
        invocation: LLMInvocation,
        logger: Logger,
        capture_content: bool,
        handler,
    ):
        self.stream = stream
        self.invocation = invocation
        self.span = getattr(invocation, "span", None)
        self.handler = handler
        self.choice_buffers = []
        self.finish_reasons = []  # Instance-level to avoid cross-request contamination
        self._span_started = False
        self.capture_content = capture_content
        self._telemetry_stopped = False
        self._error: Optional[Exception] = None

        self.logger = logger
        self.setup()

    def setup(self):
        if not self._span_started:
            self._span_started = True

    def _build_output_messages(self) -> list[OutputMessage]:
        output_messages: list[OutputMessage] = []
        for choice in self.choice_buffers:
            parts: list[Any] = []
            if choice.text_content:
                joined = "".join(choice.text_content)
                parts.append(
                    Text(content=joined if self.capture_content else "")
                )
            if choice.tool_calls_buffers:
                for tool_call_state in choice.tool_calls_buffers:
                    if tool_call_state is None:
                        continue
                    tool_call, tool_type = tool_call_state.finalize()
                    tool_call.provider = (
                        GenAIAttributes.GenAiProviderNameValues.OPENAI.value
                    )
                    parts.append(tool_call)

            finish_reason = choice.finish_reason or "error"
            output_messages.append(
                OutputMessage(
                    role="assistant", parts=parts, finish_reason=finish_reason
                )
            )
        return output_messages

    def _populate_invocation(self) -> None:
        self.invocation.response_id = self.response_id
        self.invocation.response_model_name = self.response_model
        self.invocation.response_service_tier = self.service_tier
        self.invocation.response_finish_reasons = self.finish_reasons
        self.invocation.input_tokens = self.prompt_tokens
        self.invocation.output_tokens = self.completion_tokens
        self.invocation.output_messages = self._build_output_messages()

    def cleanup(self):
        if self._span_started:
            # Record attributes on the span before ending the util span.
            if self.span and self.span.is_recording():
                if self.response_model:
                    set_span_attribute(
                        self.span,
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL,
                        self.response_model,
                    )

                if self.response_id:
                    set_span_attribute(
                        self.span,
                        GenAIAttributes.GEN_AI_RESPONSE_ID,
                        self.response_id,
                    )

                set_span_attribute(
                    self.span,
                    GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                    self.prompt_tokens,
                )
                set_span_attribute(
                    self.span,
                    GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                    self.completion_tokens,
                )

                set_span_attribute(
                    self.span,
                    GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
                    self.service_tier,
                )

                set_span_attribute(
                    self.span,
                    GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
                    self.finish_reasons,
                )

            if not self._telemetry_stopped:
                self._populate_invocation()
                if self._error:
                    self.handler.fail_llm(
                        self.invocation,
                        InvocationError(
                            message=str(self._error), type=type(self._error)
                        ),
                    )
                else:
                    self.handler.stop_llm(self.invocation)
                self._telemetry_stopped = True

            for idx, choice in enumerate(self.choice_buffers):
                message = {"role": "assistant"}
                if self.capture_content and choice.text_content:
                    message["content"] = "".join(choice.text_content)
                if choice.tool_calls_buffers:
                    tool_calls = []
                    for tool_call_state in choice.tool_calls_buffers:
                        if tool_call_state is None:
                            continue
                        tool_call, tool_type = tool_call_state.finalize()
                        tool_calls.append(
                            _tool_call_body(
                                tool_call, tool_type, self.capture_content
                            )
                        )
                    message["tool_calls"] = tool_calls

                body = {
                    "index": idx,
                    "finish_reason": choice.finish_reason or "error",
                    "message": message,
                }

                event_attributes = {
                    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiProviderNameValues.OPENAI.value
                }
                context = (
                    set_span_in_context(self.span, get_current())
                    if self.span
                    else get_current()
                )
                self.logger.emit(
                    LogRecord(
                        event_name="gen_ai.choice",
                        attributes=event_attributes,
                        body=body,
                        context=context,
                    )
                )

            self._span_started = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self._error = exc_val
                if self.span:
                    handle_span_exception(self.span, exc_val)
        finally:
            self.cleanup()
        return False  # Propagate the exception

    async def __aenter__(self):
        self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self._error = exc_val
                if self.span:
                    handle_span_exception(self.span, exc_val)
        finally:
            self.cleanup()
        return False  # Propagate the exception

    def close(self):
        result = self.stream.close()
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(result)
            except RuntimeError:
                asyncio.run(result)
        self.cleanup()

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)
            self.process_chunk(chunk)
            return chunk
        except StopIteration:
            self.cleanup()
            raise
        except Exception as error:
            self._error = error
            if self.span:
                handle_span_exception(self.span, error)
            self.cleanup()
            raise

    async def __anext__(self):
        try:
            chunk = await self.stream.__anext__()
            self.process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self.cleanup()
            raise
        except Exception as error:
            self._error = error
            if self.span:
                handle_span_exception(self.span, error)
            self.cleanup()
            raise

    def set_response_model(self, chunk):
        if self.response_model:
            return

        if getattr(chunk, "model", None):
            self.response_model = chunk.model

    def set_response_id(self, chunk):
        if self.response_id:
            return

        if getattr(chunk, "id", None):
            self.response_id = chunk.id

    def set_response_service_tier(self, chunk):
        if self.service_tier:
            return

        if getattr(chunk, "service_tier", None):
            self.service_tier = chunk.service_tier

    def build_streaming_response(self, chunk):
        if getattr(chunk, "choices", None) is None:
            return

        choices = chunk.choices
        for choice in choices:
            if not choice.delta:
                continue

            # make sure we have enough choice buffers
            for idx in range(len(self.choice_buffers), choice.index + 1):
                self.choice_buffers.append(
                    ChoiceBuffer(idx, self.capture_content)
                )

            if choice.finish_reason:
                self.choice_buffers[
                    choice.index
                ].finish_reason = choice.finish_reason
                self.finish_reasons.append(choice.finish_reason)

            if choice.delta.content is not None:
                self.choice_buffers[choice.index].append_text_content(
                    choice.delta.content
                )

            if choice.delta.tool_calls is not None:
                for tool_call in choice.delta.tool_calls:
                    self.choice_buffers[choice.index].append_tool_call(
                        tool_call
                    )

    def set_usage(self, chunk):
        if getattr(chunk, "usage", None):
            self.completion_tokens = chunk.usage.completion_tokens
            self.prompt_tokens = chunk.usage.prompt_tokens

    def process_chunk(self, chunk):
        self.set_response_id(chunk)
        self.set_response_model(chunk)
        self.set_response_service_tier(chunk)
        self.build_streaming_response(chunk)
        self.set_usage(chunk)
