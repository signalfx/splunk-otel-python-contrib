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

"""Tests for Bedrock Runtime botocore wrappers."""

import pytest
from opentelemetry import context as context_api
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.util.genai.types import Text, ToolCall, ToolCallResponse

from opentelemetry.instrumentation.bedrock.wrappers import (
    bedrock_runtime_api_call_wrapper,
)

from .conftest import FakeClient, FakeStream


def _call_wrapper(
    handler,
    client,
    operation_name,
    api_params,
    result,
    capture_content=True,
):
    wrapper = bedrock_runtime_api_call_wrapper(capture_content, handler)

    def wrapped(op_name, params):
        assert op_name == operation_name
        assert params == api_params
        return result

    return wrapper(wrapped, client, (operation_name, api_params), {})


def _converse_params():
    return {
        "modelId": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "You are concise."}],
        "messages": [
            {"role": "user", "content": [{"text": "What is the weather?"}]},
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tool-1",
                            "content": [{"json": {"temperature": 22}}],
                        }
                    }
                ],
            },
        ],
        "inferenceConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxTokens": 128,
            "stopSequences": ["</answer>"],
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            }
                        },
                    }
                }
            ]
        },
    }


def _converse_result():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "It is sunny."},
                    {
                        "toolUse": {
                            "toolUseId": "tool-2",
                            "name": "get_forecast",
                            "input": {"city": "Paris"},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 12, "outputTokens": 8, "totalTokens": 20},
        "ResponseMetadata": {"RequestId": "request-123"},
    }


def test_converse_happy_path_maps_request_and_response(stub_handler, fake_client):
    result = _call_wrapper(
        stub_handler,
        fake_client,
        "Converse",
        _converse_params(),
        _converse_result(),
        capture_content=True,
    )

    assert result == _converse_result()
    assert len(stub_handler.started_llm) == 1
    assert len(stub_handler.stopped_llm) == 1
    invocation = stub_handler.stopped_llm[0]

    assert invocation.request_model == "us.anthropic.claude-3-haiku-20240307-v1:0"
    assert invocation.provider == "anthropic"
    assert invocation.system == "aws.bedrock"
    assert invocation.framework == "boto3"
    assert invocation.request_temperature == 0.2
    assert invocation.request_top_p == 0.9
    assert invocation.request_max_tokens == 128
    assert invocation.request_stop_sequences == ["</answer>"]
    assert invocation.request_functions[0]["name"] == "get_weather"
    assert invocation.input_tokens == 12
    assert invocation.output_tokens == 8
    assert invocation.response_id == "request-123"
    assert invocation.response_finish_reasons == ["tool_calls"]
    assert invocation.server_address == "bedrock-runtime.us-west-2.amazonaws.com"

    assert len(invocation.input_messages) == 3
    assert invocation.input_messages[0].role == "system"
    assert isinstance(invocation.input_messages[1].parts[0], Text)
    assert invocation.input_messages[1].parts[0].content == "What is the weather?"
    assert isinstance(invocation.input_messages[2].parts[0], ToolCallResponse)
    assert invocation.input_messages[2].parts[0].response == {"temperature": 22}

    assert len(invocation.output_messages) == 1
    output_parts = invocation.output_messages[0].parts
    assert isinstance(output_parts[0], Text)
    assert output_parts[0].content == "It is sunny."
    assert isinstance(output_parts[1], ToolCall)
    assert output_parts[1].name == "get_forecast"
    assert output_parts[1].arguments == {"city": "Paris"}


def test_converse_content_capture_off_suppresses_messages(stub_handler, fake_client):
    _call_wrapper(
        stub_handler,
        fake_client,
        "Converse",
        _converse_params(),
        _converse_result(),
        capture_content=False,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.input_messages == []
    assert invocation.output_messages == []
    assert invocation.input_tokens == 12
    assert invocation.output_tokens == 8
    assert invocation.request_functions[0]["name"] == "get_weather"


def test_converse_exception_fails_invocation(stub_handler, fake_client):
    wrapper = bedrock_runtime_api_call_wrapper(True, stub_handler)

    def wrapped(_op_name, _params):
        raise RuntimeError("bedrock failed")

    with pytest.raises(RuntimeError, match="bedrock failed"):
        wrapper(wrapped, fake_client, ("Converse", _converse_params()), {})

    assert len(stub_handler.started_llm) == 1
    assert len(stub_handler.stopped_llm) == 0
    assert len(stub_handler.failed_llm) == 1
    _invocation, error = stub_handler.failed_llm[0]
    assert error.type is RuntimeError
    assert error.message == "bedrock failed"


def test_telemetry_setup_error_passes_through_to_bedrock(fake_client):
    class FailingStartHandler:
        def start_llm(self, _invocation):
            raise RuntimeError("telemetry failed")

        def stop_llm(self, _invocation):
            raise AssertionError("stop_llm should not be called")

        def fail_llm(self, _invocation, _error):
            raise AssertionError("fail_llm should not be called")

    wrapper = bedrock_runtime_api_call_wrapper(True, FailingStartHandler())

    result = wrapper(
        lambda _op_name, _params: {"ok": True},
        fake_client,
        ("Converse", _converse_params()),
        {},
    )

    assert result == {"ok": True}


def test_converse_stream_finalizes_on_exhaustion(stub_handler, fake_client):
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Hel"},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "lo"},
            }
        },
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
    ]
    result = {"stream": FakeStream(events), "ResponseMetadata": {"RequestId": "rid"}}

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "ConverseStream",
        _converse_params(),
        result,
        capture_content=True,
    )

    assert len(stub_handler.stopped_llm) == 0
    assert list(wrapped_result["stream"]) == events
    assert len(stub_handler.stopped_llm) == 1
    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_stream is True
    assert invocation.response_id == "rid"
    assert invocation.response_finish_reasons == ["stop"]
    assert invocation.input_tokens == 5
    assert invocation.output_tokens == 2
    assert "gen_ai.response.time_to_first_chunk" in invocation.attributes
    assert invocation.output_messages[0].parts[0].content == "Hello"


def test_invoke_model_maps_known_anthropic_payload(stub_handler, fake_client):
    params = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": (
            b'{"messages":[{"role":"user","content":[{"type":"text",'
            b'"text":"hi"}]}],"max_tokens":32,"temperature":0.1}'
        ),
    }
    result = {
        "body": (
            b'{"id":"msg-1","model":"claude-3-haiku","stop_reason":"end_turn",'
            b'"usage":{"input_tokens":3,"output_tokens":4},'
            b'"content":[{"type":"text","text":"hello"}]}'
        ),
        "ResponseMetadata": {"RequestId": "request-456"},
    }

    _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModel",
        params,
        result,
        capture_content=True,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_max_tokens == 32
    assert invocation.request_temperature == 0.1
    assert invocation.input_messages[0].parts[0].content == "hi"
    assert invocation.response_id == "msg-1"
    assert invocation.response_model_name == "claude-3-haiku"
    assert invocation.response_finish_reasons == ["stop"]
    assert invocation.input_tokens == 3
    assert invocation.output_tokens == 4
    assert invocation.output_messages[0].parts[0].content == "hello"


def test_invoke_model_uses_token_headers_for_unknown_response(
    stub_handler, fake_client
):
    params = {
        "modelId": "amazon.titan-text-express-v1",
        "body": b'{"inputText":"hello","maxTokens":64}',
    }
    result = {
        "body": b"not-json",
        "ResponseMetadata": {
            "RequestId": "request-789",
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "6",
                "x-amzn-bedrock-output-token-count": "9",
            },
        },
    }

    _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModel",
        params,
        result,
        capture_content=True,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_model == "amazon.titan-text-express-v1"
    assert invocation.provider == "amazon"
    assert invocation.request_max_tokens == 64
    assert invocation.input_messages[0].parts[0].content == "hello"
    assert invocation.response_id == "request-789"
    assert invocation.input_tokens == 6
    assert invocation.output_tokens == 9
    assert invocation.output_messages == []


def test_non_bedrock_runtime_call_is_not_instrumented(stub_handler):
    client = FakeClient(service_name="s3")
    result = {"ok": True}

    wrapped_result = _call_wrapper(
        stub_handler,
        client,
        "ListBuckets",
        {},
        result,
        capture_content=True,
    )

    assert wrapped_result == result
    assert stub_handler.started_llm == []
    assert stub_handler.stopped_llm == []


def test_suppression_context_skips_instrumentation(stub_handler, fake_client):
    wrapper = bedrock_runtime_api_call_wrapper(True, stub_handler)
    ctx = context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    token = context_api.attach(ctx)
    try:
        result = wrapper(
            lambda _op_name, _params: {"ok": True},
            fake_client,
            ("Converse", _converse_params()),
            {},
        )
    finally:
        context_api.detach(token)

    assert result == {"ok": True}
    assert stub_handler.started_llm == []
