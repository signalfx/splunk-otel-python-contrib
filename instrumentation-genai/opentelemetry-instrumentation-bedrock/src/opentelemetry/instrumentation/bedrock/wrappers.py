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

"""Botocore wrappers for Bedrock Runtime GenAI instrumentation."""

from __future__ import annotations

import io
import timeit
from typing import Any, Callable, Optional
from urllib.parse import urlparse

try:
    from botocore.response import StreamingBody
except (ImportError, ModuleNotFoundError):
    StreamingBody = None

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)
from opentelemetry.util.genai.utils import should_capture_tool_definitions

from .utils import (
    maybe_parse_json,
    parse_json_body,
    safe_json_dumps,
    safe_str,
    truncate_error,
)

_BEDROCK_RUNTIME_SERVICE = "bedrock-runtime"
_SUPPORTED_OPERATIONS = {
    "Converse",
    "ConverseStream",
    "InvokeModel",
    "InvokeModelWithResponseStream",
}
_STREAMING_OPERATIONS = {"ConverseStream", "InvokeModelWithResponseStream"}
_STREAM_BUFFER_LIMIT = 64 * 1024
_TEXT_COMPLETION_OPERATION = "text_completion"


def bedrock_runtime_api_call_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    """Wrap ``botocore.client.BaseClient._make_api_call``."""

    def traced_method(
        wrapped: Any, instance: Any, args: tuple, kwargs: dict
    ) -> Any:
        operation_name, api_params = _extract_api_call_args(args, kwargs)
        if (
            not _is_bedrock_runtime_client(instance)
            or operation_name not in _SUPPORTED_OPERATIONS
        ):
            return wrapped(*args, **kwargs)

        stream_start_time = (
            timeit.default_timer()
            if operation_name in _STREAMING_OPERATIONS
            else None
        )
        try:
            invocation = _build_invocation(
                instance, operation_name, api_params
            )
            handler.start_llm(invocation)
        except Exception:
            return wrapped(*args, **kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            _fail_safely(handler, invocation, error)
            raise

        try:
            if operation_name in _STREAMING_OPERATIONS:
                return _wrap_streaming_result(
                    result,
                    invocation,
                    operation_name,
                    handler,
                    stream_start_time,
                )
            _apply_response(invocation, operation_name, result)
            handler.stop_llm(invocation)
        except Exception:
            _stop_safely(handler, invocation)

        return result

    return traced_method


def _extract_api_call_args(
    args: tuple, kwargs: dict
) -> tuple[Optional[str], dict]:
    operation_name = args[0] if args else kwargs.get("operation_name")
    api_params = args[1] if len(args) > 1 else kwargs.get("api_params")
    return operation_name, api_params if isinstance(api_params, dict) else {}


def _is_bedrock_runtime_client(instance: Any) -> bool:
    meta = getattr(instance, "meta", None)
    service_model = getattr(meta, "service_model", None)
    service_name = getattr(service_model, "service_name", None)
    if service_name is None:
        service_name = getattr(service_model, "service_id", None)
    normalized = safe_str(service_name).lower().replace(" ", "-")
    return normalized == _BEDROCK_RUNTIME_SERVICE


def _build_invocation(
    instance: Any,
    operation_name: str,
    api_params: dict,
) -> LLMInvocation:
    if operation_name in {"Converse", "ConverseStream"}:
        invocation = _build_converse_invocation(
            instance, operation_name, api_params
        )
    else:
        invocation = _build_invoke_model_invocation(
            instance, operation_name, api_params
        )
    if operation_name in _STREAMING_OPERATIONS:
        invocation.request_stream = True
    return invocation


def _base_invocation(
    instance: Any,
    model_id: str,
    operation_name: str,
) -> LLMInvocation:
    server_address, server_port = _server_from_client(instance)
    return LLMInvocation(
        request_model=model_id,
        provider="aws.bedrock",
        framework="boto3",
        system="aws.bedrock",
        server_address=server_address,
        server_port=server_port,
        attributes={"custom_aws_bedrock.operation": operation_name},
    )


def _build_converse_invocation(
    instance: Any,
    operation_name: str,
    api_params: dict,
) -> LLMInvocation:
    model_id = safe_str(api_params.get("modelId") or "")
    invocation = _base_invocation(instance, model_id, operation_name)

    inference_config = api_params.get("inferenceConfig") or {}
    if isinstance(inference_config, dict):
        invocation.request_max_tokens = inference_config.get("maxTokens")
        invocation.request_temperature = inference_config.get("temperature")
        invocation.request_top_p = inference_config.get("topP")
        stop_sequences = inference_config.get("stopSequences")
        if isinstance(stop_sequences, list):
            invocation.request_stop_sequences = [
                safe_str(s) for s in stop_sequences
            ]

    tool_config = api_params.get("toolConfig") or {}
    tools = tool_config.get("tools") if isinstance(tool_config, dict) else None
    invocation.request_functions = _request_functions_from_bedrock_tools(tools)
    if tools and should_capture_tool_definitions():
        invocation.tool_definitions = safe_json_dumps(tools)

    invocation.input_messages = _input_messages_from_converse_request(
        api_params, invocation.provider
    )

    return invocation


def _build_invoke_model_invocation(
    instance: Any,
    operation_name: str,
    api_params: dict,
) -> LLMInvocation:
    model_id = safe_str(api_params.get("modelId") or "")
    invocation = _base_invocation(instance, model_id, operation_name)
    body = parse_json_body(api_params.get("body")) or {}

    _apply_invoke_model_request(invocation, model_id, body)

    tools = body.get("tools")
    invocation.request_functions = _request_functions_from_invoke_tools(tools)
    if tools and should_capture_tool_definitions():
        invocation.tool_definitions = safe_json_dumps(tools)

    invocation.input_messages = _input_messages_from_invoke_body(
        body, invocation.provider
    )

    return invocation


def _apply_response(
    invocation: LLMInvocation,
    operation_name: str,
    result: Any,
) -> None:
    if not isinstance(result, dict):
        return
    invocation.response_id = (
        result.get("ResponseMetadata", {}).get("RequestId")
        or invocation.response_id
    )
    if operation_name == "Converse":
        _apply_converse_response(invocation, result)
    elif operation_name == "InvokeModel":
        _apply_invoke_model_response(invocation, result)


def _apply_converse_response(invocation: LLMInvocation, result: dict) -> None:
    usage = result.get("usage") or {}
    if isinstance(usage, dict):
        invocation.input_tokens = usage.get("inputTokens")
        invocation.output_tokens = usage.get("outputTokens")

    stop_reason = result.get("stopReason")
    mapped_stop_reason = _map_stop_reason(stop_reason)
    if mapped_stop_reason:
        invocation.response_finish_reasons = [mapped_stop_reason]

    message = (result.get("output") or {}).get("message")
    if isinstance(message, dict):
        invocation.output_messages = [
            _message_from_converse_message(
                message,
                invocation.provider,
                finish_reason=mapped_stop_reason,
            )
        ]


def _apply_invoke_model_response(
    invocation: LLMInvocation, result: dict
) -> None:
    body = _parse_result_body(result) or {}
    if not body:
        _apply_token_headers(invocation, result)
        return

    invocation.response_id = body.get("id") or invocation.response_id
    invocation.response_model_name = body.get("model") or body.get("modelId")

    _apply_invoke_usage(invocation, body)
    _apply_token_headers(invocation, result)

    stop_reason = _extract_invoke_finish_reason(body)
    mapped_stop_reason = _map_stop_reason(stop_reason)
    if mapped_stop_reason:
        invocation.response_finish_reasons = [mapped_stop_reason]

    output_message = _output_message_from_invoke_body(
        body,
        invocation.provider,
        mapped_stop_reason,
    )
    if output_message is not None:
        invocation.output_messages = [output_message]


def _apply_invoke_model_request(
    invocation: LLMInvocation,
    model_id: str,
    body: dict,
) -> None:
    family = _model_family(model_id)
    if family == "titan":
        invocation.operation = _TEXT_COMPLETION_OPERATION
        config = body.get("textGenerationConfig") or {}
        if isinstance(config, dict):
            invocation.request_max_tokens = _first_present(
                config, "maxTokenCount", "maxTokens"
            )
            invocation.request_temperature = _first_present(
                config, "temperature"
            )
            invocation.request_top_p = _first_present(config, "topP", "top_p")
            _set_stop_sequences(invocation, config.get("stopSequences"))
        invocation.request_max_tokens = (
            invocation.request_max_tokens
            or _first_present(body, "maxTokens", "max_tokens")
        )
        return

    if family == "nova":
        config = body.get("inferenceConfig") or {}
        if isinstance(config, dict):
            invocation.request_max_tokens = _first_present(
                config, "max_new_tokens", "maxTokens", "max_tokens"
            )
            invocation.request_temperature = _first_present(
                config, "temperature"
            )
            invocation.request_top_p = _first_present(config, "topP", "top_p")
            _set_stop_sequences(invocation, config.get("stopSequences"))
        return

    if family == "llama":
        invocation.request_max_tokens = _first_present(
            body, "max_gen_len", "max_tokens", "maxTokens"
        )
        invocation.request_temperature = _first_present(body, "temperature")
        invocation.request_top_p = _first_present(body, "top_p", "topP")
        return

    if family == "mistral":
        invocation.request_max_tokens = _first_present(
            body, "max_tokens", "maxTokens"
        )
        invocation.request_temperature = _first_present(body, "temperature")
        invocation.request_top_p = _first_present(body, "top_p", "topP")
        _set_stop_sequences(
            invocation, _first_present(body, "stop", "stop_sequences")
        )
        return

    if family in {"command-r", "command"}:
        invocation.request_max_tokens = _first_present(
            body, "max_tokens", "maxTokens"
        )
        invocation.request_temperature = _first_present(body, "temperature")
        invocation.request_top_p = _first_present(body, "p", "top_p", "topP")
        _set_stop_sequences(invocation, body.get("stop_sequences"))
        return

    invocation.request_max_tokens = _first_present(
        body, "max_tokens", "maxTokens", "max_tokens_to_sample"
    )
    invocation.request_temperature = _first_present(body, "temperature")
    invocation.request_top_p = _first_present(body, "top_p", "topP")
    invocation.request_top_k = _first_present(body, "top_k", "topK")
    _set_stop_sequences(
        invocation, _first_present(body, "stop_sequences", "stopSequences")
    )


def _parse_result_body(result: dict) -> Optional[dict[str, Any]]:
    body = result.get("body")
    parsed = parse_json_body(body)
    if parsed is not None:
        return parsed
    if body is None or not hasattr(body, "read"):
        return None

    try:
        body_content = body.read()
    except Exception:
        return None

    try:
        result["body"] = _rebuild_body_stream(body_content)
    except Exception:
        pass

    try:
        close = getattr(body, "close", None)
        if close is not None:
            close()
    except Exception:
        pass

    return parse_json_body(body_content)


def _rebuild_body_stream(body_content: Any) -> Any:
    raw = (
        bytes(body_content)
        if isinstance(body_content, bytearray)
        else body_content
    )
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes):
        raw = safe_str(raw).encode("utf-8")
    if StreamingBody is not None:
        try:
            return StreamingBody(io.BytesIO(raw), len(raw))
        except Exception:
            pass
    return io.BytesIO(raw)


def _apply_invoke_usage(invocation: LLMInvocation, body: dict) -> None:
    usage = body.get("usage") or {}
    if isinstance(usage, dict):
        input_tokens = _first_present(
            usage, "input_tokens", "inputTokens", "prompt_tokens"
        )
        output_tokens = _first_present(
            usage, "output_tokens", "outputTokens", "completion_tokens"
        )
        if input_tokens is not None:
            invocation.input_tokens = input_tokens
        if output_tokens is not None:
            invocation.output_tokens = output_tokens

    if "inputTextTokenCount" in body:
        invocation.input_tokens = body.get("inputTextTokenCount")
    results = body.get("results")
    if isinstance(results, list) and results:
        first_result = results[0]
        if isinstance(first_result, dict) and "tokenCount" in first_result:
            invocation.output_tokens = first_result.get("tokenCount")

    if "prompt_token_count" in body:
        invocation.input_tokens = body.get("prompt_token_count")
    if "generation_token_count" in body:
        invocation.output_tokens = body.get("generation_token_count")


def _extract_invoke_finish_reason(body: dict) -> Any:
    if (
        stop_reason := _first_present(
            body, "stop_reason", "stopReason", "finish_reason", "finishReason"
        )
    ) is not None:
        return stop_reason
    results = body.get("results")
    if isinstance(results, list) and results:
        first_result = results[0]
        if isinstance(first_result, dict):
            return first_result.get("completionReason")
    generations = body.get("generations")
    if isinstance(generations, list) and generations:
        first_generation = generations[0]
        if isinstance(first_generation, dict):
            return first_generation.get("finish_reason")
    outputs = body.get("outputs")
    if isinstance(outputs, list) and outputs:
        first_output = outputs[0]
        if isinstance(first_output, dict):
            return first_output.get("stop_reason")
    return None


def _output_message_from_invoke_body(
    body: dict,
    provider: Optional[str],
    finish_reason: Optional[str],
) -> Optional[OutputMessage]:
    role = safe_str(body.get("role") or "assistant")
    content = body.get("content")
    if isinstance(content, list):
        parts = _parts_from_invoke_content(content, provider)
        if parts:
            return OutputMessage(
                role=role, parts=parts, finish_reason=finish_reason
            )

    output = body.get("output")
    if isinstance(output, dict):
        message = output.get("message")
        if isinstance(message, dict):
            parsed = _message_from_converse_message(
                message,
                provider,
                finish_reason=finish_reason,
            )
            if isinstance(parsed, OutputMessage):
                return parsed

    output_text = _extract_invoke_output_text(body)
    if output_text is None:
        return None
    return OutputMessage(
        role=role,
        parts=[Text(content=output_text)],
        finish_reason=finish_reason,
    )


def _wrap_streaming_result(
    result: Any,
    invocation: LLMInvocation,
    operation_name: str,
    handler: TelemetryHandler,
    stream_start_time: Optional[float],
) -> Any:
    if not isinstance(result, dict):
        _stop_safely(handler, invocation)
        return result

    invocation.response_id = (
        result.get("ResponseMetadata", {}).get("RequestId")
        or invocation.response_id
    )

    stream_key = "stream" if operation_name == "ConverseStream" else "body"
    stream = result.get(stream_key)
    if stream is None:
        _stop_safely(handler, invocation)
        return result

    result[stream_key] = _BedrockStreamWrapper(
        stream=stream,
        invocation=invocation,
        operation_name=operation_name,
        handler=handler,
        stream_start_time=stream_start_time,
    )
    return result


class _BedrockStreamWrapper:
    """Iterator wrapper that finalizes Bedrock streaming LLM telemetry."""

    def __init__(
        self,
        stream: Any,
        invocation: LLMInvocation,
        operation_name: str,
        handler: TelemetryHandler,
        stream_start_time: Optional[float],
    ) -> None:
        self._stream = stream
        self._invocation = invocation
        self._operation_name = operation_name
        self._handler = handler
        self._stream_start_time = stream_start_time
        self._stopped = False
        self._first_chunk_processed = False
        self._role = "assistant"
        self._content_blocks: dict[int, dict[str, Any]] = {}
        self._finish_reason: Optional[str] = None
        self._invoke_text_parts: list[str] = []
        self._invoke_body = bytearray()

    def __iter__(self) -> "_BedrockStreamWrapper":
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._stream)
            self._process_event(event)
            return event
        except StopIteration:
            self._finish()
            raise
        except Exception as error:
            if not self._stopped:
                _fail_safely(self._handler, self._invocation, error)
                self._stopped = True
            raise

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()
        finally:
            self._finish()

    def _process_event(self, event: Any) -> None:
        self._record_ttfc()
        if not isinstance(event, dict):
            return
        if self._operation_name == "ConverseStream":
            self._process_converse_stream_event(event)
        else:
            self._process_invoke_model_stream_event(event)

    def _record_ttfc(self) -> None:
        if self._first_chunk_processed:
            return
        self._first_chunk_processed = True
        if self._stream_start_time is not None:
            self._invocation.attributes[
                "gen_ai.response.time_to_first_chunk"
            ] = timeit.default_timer() - self._stream_start_time

    def _process_converse_stream_event(self, event: dict) -> None:
        if "messageStart" in event:
            role = event["messageStart"].get("role")
            if role:
                self._role = safe_str(role)
            return

        if "contentBlockStart" in event:
            data = event["contentBlockStart"]
            index = data.get("contentBlockIndex", 0)
            start = data.get("start") or {}
            tool_use = (
                start.get("toolUse") if isinstance(start, dict) else None
            )
            if isinstance(tool_use, dict):
                self._content_blocks[index] = {
                    "type": "toolUse",
                    "toolUseId": tool_use.get("toolUseId"),
                    "name": tool_use.get("name"),
                    "input": "",
                }
            return

        if "contentBlockDelta" in event:
            data = event["contentBlockDelta"]
            index = data.get("contentBlockIndex", 0)
            delta = data.get("delta") or {}
            if "text" in delta:
                block = self._content_blocks.setdefault(
                    index, {"type": "text", "text": ""}
                )
                block["text"] = safe_str(block.get("text", "")) + safe_str(
                    delta.get("text", "")
                )
            elif "toolUse" in delta:
                block = self._content_blocks.setdefault(
                    index, {"type": "toolUse", "input": ""}
                )
                tool_delta = delta.get("toolUse") or {}
                block["input"] = safe_str(block.get("input", "")) + safe_str(
                    tool_delta.get("input", "")
                )
            return

        if "messageStop" in event:
            self._finish_reason = _map_stop_reason(
                event["messageStop"].get("stopReason")
            )
            if self._finish_reason:
                self._invocation.response_finish_reasons = [
                    self._finish_reason
                ]
            return

        if "metadata" in event:
            usage = event["metadata"].get("usage") or {}
            if isinstance(usage, dict):
                self._invocation.input_tokens = usage.get("inputTokens")
                self._invocation.output_tokens = usage.get("outputTokens")

    def _process_invoke_model_stream_event(self, event: dict) -> None:
        chunk = event.get("chunk") or {}
        chunk_bytes = chunk.get("bytes") if isinstance(chunk, dict) else None
        parsed_chunk = parse_json_body(chunk_bytes) or {}
        if parsed_chunk:
            family = _model_family(self._invocation.request_model)
            if family == "titan":
                self._process_titan_stream_chunk(parsed_chunk)
            elif family == "nova":
                self._process_converse_stream_event(parsed_chunk)
            elif family == "claude":
                self._process_anthropic_stream_chunk(parsed_chunk)
            else:
                self._process_generic_stream_chunk(parsed_chunk)

        if (
            isinstance(chunk_bytes, (bytes, bytearray))
            and len(self._invoke_body) < _STREAM_BUFFER_LIMIT
        ):
            remaining = _STREAM_BUFFER_LIMIT - len(self._invoke_body)
            self._invoke_body.extend(bytes(chunk_bytes)[:remaining])

    def _finish(self) -> None:
        if self._stopped:
            return
        try:
            if self._operation_name == "ConverseStream":
                parts = _parts_from_stream_blocks(
                    self._content_blocks, self._invocation.provider
                )
                if parts:
                    self._invocation.output_messages = [
                        OutputMessage(
                            role=self._role,
                            parts=parts,
                            finish_reason=self._finish_reason,
                        )
                    ]
            elif (
                self._operation_name == "InvokeModelWithResponseStream"
                and self._invoke_body
            ):
                parts = _parts_from_stream_blocks(
                    self._content_blocks, self._invocation.provider
                )
                if parts:
                    self._invocation.output_messages = [
                        OutputMessage(
                            role=self._role,
                            parts=parts,
                            finish_reason=self._finish_reason,
                        )
                    ]
                elif self._invoke_text_parts:
                    self._invocation.output_messages = [
                        OutputMessage(
                            role=self._role,
                            parts=[
                                Text(content="".join(self._invoke_text_parts))
                            ],
                            finish_reason=self._finish_reason,
                        )
                    ]
                else:
                    body = parse_json_body(bytes(self._invoke_body)) or {}
                    output_message = _output_message_from_invoke_body(
                        body,
                        self._invocation.provider,
                        self._finish_reason,
                    )
                    if output_message is not None:
                        self._invocation.output_messages = [output_message]
            _stop_safely(self._handler, self._invocation)
        finally:
            self._stopped = True

    def _process_titan_stream_chunk(self, chunk: dict) -> None:
        if output_text := chunk.get("outputText"):
            self._invoke_text_parts.append(safe_str(output_text))
        if stop_reason := chunk.get("completionReason"):
            self._finish_reason = _map_stop_reason(stop_reason)
            if self._finish_reason:
                self._invocation.response_finish_reasons = [
                    self._finish_reason
                ]
        self._apply_invocation_metrics(
            chunk.get("amazon-bedrock-invocationMetrics")
        )

    def _process_anthropic_stream_chunk(self, chunk: dict) -> None:
        message_type = chunk.get("type")
        if message_type == "message_start":
            message = chunk.get("message") or {}
            if role := message.get("role"):
                self._role = safe_str(role)
            self._invocation.response_id = (
                message.get("id") or self._invocation.response_id
            )
            self._invocation.response_model_name = (
                message.get("model") or self._invocation.response_model_name
            )
            usage = message.get("usage") or {}
            if isinstance(usage, dict):
                self._invocation.input_tokens = usage.get("input_tokens")
            return

        if message_type == "content_block_start":
            index = chunk.get("index", 0)
            block = chunk.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._content_blocks[index] = {
                    "type": "toolUse",
                    "toolUseId": block.get("id"),
                    "name": block.get("name"),
                    "input": "",
                }
            else:
                self._content_blocks[index] = {
                    "type": "text",
                    "text": safe_str(block.get("text", "")),
                }
            return

        if message_type == "content_block_delta":
            index = chunk.get("index", 0)
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                block = self._content_blocks.setdefault(
                    index, {"type": "text", "text": ""}
                )
                block["text"] = safe_str(block.get("text", "")) + safe_str(
                    delta.get("text", "")
                )
            elif delta.get("type") == "input_json_delta":
                block = self._content_blocks.setdefault(
                    index, {"type": "toolUse", "input": ""}
                )
                block["input"] = safe_str(block.get("input", "")) + safe_str(
                    delta.get("partial_json", "")
                )
            return

        if message_type == "message_delta":
            delta = chunk.get("delta") or {}
            if stop_reason := delta.get("stop_reason"):
                self._finish_reason = _map_stop_reason(stop_reason)
                if self._finish_reason:
                    self._invocation.response_finish_reasons = [
                        self._finish_reason
                    ]
            usage = chunk.get("usage") or {}
            if (
                isinstance(usage, dict)
                and usage.get("output_tokens") is not None
            ):
                self._invocation.output_tokens = usage.get("output_tokens")
            return

        if message_type == "message_stop":
            self._apply_invocation_metrics(
                chunk.get("amazon-bedrock-invocationMetrics")
            )
            return

    def _process_generic_stream_chunk(self, chunk: dict) -> None:
        _apply_invoke_usage(self._invocation, chunk)
        if stop_reason := _extract_invoke_finish_reason(chunk):
            self._finish_reason = _map_stop_reason(stop_reason)
            if self._finish_reason:
                self._invocation.response_finish_reasons = [
                    self._finish_reason
                ]
        output = _extract_invoke_output_text(chunk)
        if output:
            self._invoke_text_parts.append(output)

    def _apply_invocation_metrics(self, invocation_metrics: Any) -> None:
        if not isinstance(invocation_metrics, dict):
            return
        if invocation_metrics.get("inputTokenCount") is not None:
            self._invocation.input_tokens = invocation_metrics.get(
                "inputTokenCount"
            )
        if invocation_metrics.get("outputTokenCount") is not None:
            self._invocation.output_tokens = invocation_metrics.get(
                "outputTokenCount"
            )


def _input_messages_from_converse_request(
    api_params: dict, provider: Optional[str]
) -> list[InputMessage]:
    messages: list[InputMessage] = []
    system_blocks = api_params.get("system")
    if isinstance(system_blocks, list) and system_blocks:
        messages.append(
            InputMessage(
                role="system",
                parts=_parts_from_content_blocks(system_blocks, provider),
            )
        )
    for message in api_params.get("messages") or []:
        if isinstance(message, dict):
            messages.append(_message_from_converse_message(message, provider))
    return messages


def _message_from_converse_message(
    message: dict,
    provider: Optional[str],
    finish_reason: Optional[str] = None,
) -> InputMessage | OutputMessage:
    role = safe_str(message.get("role") or "user")
    parts = _parts_from_content_blocks(message.get("content") or [], provider)
    if not parts:
        parts = [Text(content="")]
    if finish_reason is not None or role == "assistant":
        return OutputMessage(
            role=role, parts=parts, finish_reason=finish_reason
        )
    return InputMessage(role=role, parts=parts)


def _parts_from_content_blocks(
    content_blocks: Any, provider: Optional[str]
) -> list[Any]:
    parts: list[Any] = []
    if not isinstance(content_blocks, list):
        return parts
    for block in content_blocks:
        if not isinstance(block, dict):
            parts.append(Text(content=safe_str(block)))
            continue
        if "text" in block:
            parts.append(Text(content=safe_str(block.get("text", ""))))
        elif "toolUse" in block and isinstance(block["toolUse"], dict):
            tool_use = block["toolUse"]
            parts.append(
                ToolCall(
                    name=safe_str(tool_use.get("name") or "unknown_tool"),
                    id=tool_use.get("toolUseId"),
                    arguments=tool_use.get("input"),
                    provider=provider,
                    system="aws.bedrock",
                    tool_type="function",
                )
            )
        elif "toolResult" in block and isinstance(block["toolResult"], dict):
            tool_result = block["toolResult"]
            parts.append(
                ToolCallResponse(
                    id=tool_result.get("toolUseId"),
                    response=_tool_result_content(tool_result.get("content")),
                )
            )
        else:
            parts.append(Text(content=safe_json_dumps(block)))
    return parts


def _tool_result_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    values: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            values.append(item)
        elif "text" in item:
            values.append(item.get("text"))
        elif "json" in item:
            values.append(item.get("json"))
        else:
            values.append(item)
    if len(values) == 1:
        return values[0]
    return values


def _parts_from_stream_blocks(
    content_blocks: dict[int, dict[str, Any]], provider: Optional[str]
) -> list[Any]:
    parts: list[Any] = []
    for index in sorted(content_blocks):
        block = content_blocks[index]
        if block.get("type") == "toolUse":
            parts.append(
                ToolCall(
                    name=safe_str(block.get("name") or "unknown_tool"),
                    id=block.get("toolUseId"),
                    arguments=maybe_parse_json(block.get("input", "")),
                    provider=provider,
                    system="aws.bedrock",
                    tool_type="function",
                )
            )
        else:
            parts.append(Text(content=safe_str(block.get("text", ""))))
    return parts


def _input_messages_from_invoke_body(
    body: dict, provider: Optional[str]
) -> list[InputMessage]:
    if isinstance(body.get("messages"), list):
        messages: list[InputMessage] = []
        system = body.get("system")
        if isinstance(system, str) and system:
            messages.append(
                InputMessage(role="system", parts=[Text(content=system)])
            )
        for message in body["messages"]:
            if not isinstance(message, dict):
                continue
            role = safe_str(message.get("role") or "user")
            content = message.get("content")
            if isinstance(content, list):
                parts = _parts_from_invoke_content(content, provider)
            else:
                parts = [Text(content=safe_str(content or ""))]
            messages.append(InputMessage(role=role, parts=parts))
        return messages

    prompt = body.get("prompt") or body.get("inputText") or body.get("message")
    if prompt is not None:
        return [
            InputMessage(role="user", parts=[Text(content=safe_str(prompt))])
        ]
    return []


def _parts_from_invoke_content(
    content: list, provider: Optional[str]
) -> list[Any]:
    parts: list[Any] = []
    for item in content:
        if isinstance(item, dict):
            if any(key in item for key in ("text", "toolUse", "toolResult")):
                parts.extend(_parts_from_content_blocks([item], provider))
                continue
            if item.get("type") == "text" and "text" in item:
                parts.append(Text(content=safe_str(item.get("text", ""))))
            elif item.get("type") == "tool_use":
                parts.append(
                    ToolCall(
                        name=safe_str(item.get("name") or "unknown_tool"),
                        id=item.get("id"),
                        arguments=item.get("input"),
                        provider=provider,
                        system="aws.bedrock",
                        tool_type="function",
                    )
                )
            elif item.get("type") == "tool_result":
                parts.append(
                    ToolCallResponse(
                        id=item.get("tool_use_id"),
                        response=item.get("content"),
                    )
                )
            else:
                parts.append(Text(content=safe_json_dumps(item)))
        else:
            parts.append(Text(content=safe_str(item)))
    return parts


def _request_functions_from_bedrock_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    functions: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        spec = tool.get("toolSpec")
        if not isinstance(spec, dict):
            continue
        schema = spec.get("inputSchema") or {}
        parameters = schema.get("json") if isinstance(schema, dict) else None
        functions.append(
            {
                "name": spec.get("name"),
                "description": spec.get("description"),
                "parameters": parameters,
            }
        )
    return functions


def _request_functions_from_invoke_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    functions: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        functions.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("input_schema")
                or tool.get("parameters"),
            }
        )
    return functions


def _extract_invoke_output_text(body: dict) -> Optional[str]:
    if "outputText" in body:
        return safe_str(body.get("outputText"))
    if "completion" in body:
        return safe_str(body.get("completion"))
    if "generation" in body:
        return safe_str(body.get("generation"))
    if "text" in body:
        return safe_str(body.get("text"))
    results = body.get("results")
    if isinstance(results, list) and results:
        first_result = results[0]
        if isinstance(first_result, dict) and "outputText" in first_result:
            return safe_str(first_result.get("outputText"))
    generations = body.get("generations")
    if isinstance(generations, list) and generations:
        first_generation = generations[0]
        if isinstance(first_generation, dict) and "text" in first_generation:
            return safe_str(first_generation.get("text"))
    outputs = body.get("outputs")
    if isinstance(outputs, list) and outputs:
        first_output = outputs[0]
        if isinstance(first_output, dict) and "text" in first_output:
            return safe_str(first_output.get("text"))
    content = body.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(safe_str(item.get("text", "")))
        if text_parts:
            return "".join(text_parts)
    output = body.get("output")
    if isinstance(output, dict):
        message = output.get("message")
        if isinstance(message, dict):
            parts = _parts_from_content_blocks(message.get("content"), None)
            text_parts = [
                part.content for part in parts if isinstance(part, Text)
            ]
            if text_parts:
                return "".join(text_parts)
    return None


def _apply_token_headers(invocation: LLMInvocation, result: dict) -> None:
    headers = result.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    if not isinstance(headers, dict):
        return
    input_tokens = _first_present(
        headers,
        "x-amzn-bedrock-input-token-count",
        "x-amzn-bedrock-invocation-input-token-count",
    )
    output_tokens = _first_present(
        headers,
        "x-amzn-bedrock-output-token-count",
        "x-amzn-bedrock-invocation-output-token-count",
    )
    input_tokens = _coerce_int(input_tokens)
    output_tokens = _coerce_int(output_tokens)
    if invocation.input_tokens is None and input_tokens is not None:
        invocation.input_tokens = input_tokens
    if invocation.output_tokens is None and output_tokens is not None:
        invocation.output_tokens = output_tokens


def _model_family(model_id: str) -> str:
    model = safe_str(model_id)
    if "amazon.titan" in model:
        return "titan"
    if "amazon.nova" in model:
        return "nova"
    if "anthropic.claude" in model:
        return "claude"
    if "cohere.command-r" in model:
        return "command-r"
    if "cohere.command" in model:
        return "command"
    if "meta.llama" in model:
        return "llama"
    if "mistral" in model:
        return "mistral"
    return "unknown"


def _server_from_client(instance: Any) -> tuple[Optional[str], Optional[int]]:
    endpoint_url = getattr(
        getattr(instance, "meta", None), "endpoint_url", None
    )
    if not endpoint_url:
        return None, None
    try:
        parsed = urlparse(endpoint_url)
        return parsed.hostname, parsed.port
    except Exception:
        return None, None


def _map_stop_reason(stop_reason: Any) -> Optional[str]:
    if stop_reason is None:
        return None
    value = safe_str(stop_reason).strip()
    key = value.lower()
    mapping = {
        "end_turn": "stop",
        "finish": "stop",
        "complete": "stop",
        "stop": "stop",
        "stop_sequence": "stop",
        "stop_sequences": "stop",
        "stop_criteria": "stop",
        "tool_use": "tool_calls",
        "tool_calls": "tool_calls",
        "max_tokens": "length",
        "length": "length",
        "content_filtered": "content_filter",
        "content_filter": "content_filter",
        "guardrail_intervened": "content_filter",
        "error": "error",
    }
    return mapping.get(key)


def _first_present(source: dict, *keys: str) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _set_stop_sequences(
    invocation: LLMInvocation, stop_sequences: Any
) -> None:
    if isinstance(stop_sequences, str):
        invocation.request_stop_sequences = [stop_sequences]
    elif isinstance(stop_sequences, list):
        invocation.request_stop_sequences = [
            safe_str(s) for s in stop_sequences
        ]


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stop_safely(handler: TelemetryHandler, invocation: LLMInvocation) -> None:
    try:
        handler.stop_llm(invocation)
    except Exception:
        pass


def _fail_safely(
    handler: TelemetryHandler,
    invocation: LLMInvocation,
    error: Exception,
) -> None:
    try:
        handler.fail_llm(
            invocation,
            Error(type=type(error), message=truncate_error(error)),
        )
    except Exception:
        pass
