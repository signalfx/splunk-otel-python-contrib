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

from opentelemetry.instrumentation.bedrock.wrappers import (
    bedrock_runtime_api_call_wrapper,
)
from opentelemetry.util.genai.types import Text, ToolCall, ToolCallResponse

from .conftest import FakeClient, FakeStream


class FakeReadBody:
    def __init__(self, body):
        self._body = body
        self.closed = False

    def read(self):
        return self._body

    def close(self):
        self.closed = True


def _call_wrapper(
    handler,
    client,
    operation_name,
    api_params,
    result,
):
    wrapper = bedrock_runtime_api_call_wrapper(handler)

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


def test_converse_happy_path_maps_request_and_response(
    stub_handler, fake_client
):
    result = _call_wrapper(
        stub_handler,
        fake_client,
        "Converse",
        _converse_params(),
        _converse_result(),
    )

    assert result == _converse_result()
    assert len(stub_handler.started_llm) == 1
    assert len(stub_handler.stopped_llm) == 1
    invocation = stub_handler.stopped_llm[0]

    assert (
        invocation.request_model == "us.anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert invocation.provider == "aws.bedrock"
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
    assert (
        invocation.server_address == "bedrock-runtime.us-west-2.amazonaws.com"
    )

    assert len(invocation.input_messages) == 3
    assert invocation.input_messages[0].role == "system"
    assert isinstance(invocation.input_messages[1].parts[0], Text)
    assert (
        invocation.input_messages[1].parts[0].content == "What is the weather?"
    )
    assert isinstance(invocation.input_messages[2].parts[0], ToolCallResponse)
    assert invocation.input_messages[2].parts[0].response == {
        "temperature": 22
    }

    assert len(invocation.output_messages) == 1
    output_parts = invocation.output_messages[0].parts
    assert isinstance(output_parts[0], Text)
    assert output_parts[0].content == "It is sunny."
    assert isinstance(output_parts[1], ToolCall)
    assert output_parts[1].name == "get_forecast"
    assert output_parts[1].arguments == {"city": "Paris"}


def test_converse_always_populates_invocation_messages(
    stub_handler, fake_client
):
    _call_wrapper(
        stub_handler,
        fake_client,
        "Converse",
        _converse_params(),
        _converse_result(),
    )

    invocation = stub_handler.stopped_llm[0]
    assert len(invocation.input_messages) == 3
    assert invocation.output_messages[0].parts[0].content == "It is sunny."
    assert invocation.input_tokens == 12
    assert invocation.output_tokens == 8
    assert invocation.request_functions[0]["name"] == "get_weather"


def test_converse_omits_unknown_finish_reason(stub_handler, fake_client):
    result = _converse_result()
    result["stopReason"] = "MODEL_SPECIFIC_DONE"

    _call_wrapper(
        stub_handler,
        fake_client,
        "Converse",
        _converse_params(),
        result,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.response_finish_reasons == []
    assert invocation.output_messages[0].finish_reason is None


def test_converse_exception_fails_invocation(stub_handler, fake_client):
    wrapper = bedrock_runtime_api_call_wrapper(stub_handler)

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


def test_fail_llm_error_does_not_mask_bedrock_exception(fake_client):
    class FailingFailHandler:
        def __init__(self):
            self.started_llm = []

        def start_llm(self, invocation):
            self.started_llm.append(invocation)
            return invocation

        def stop_llm(self, _invocation):
            raise AssertionError("stop_llm should not be called")

        def fail_llm(self, _invocation, _error):
            raise RuntimeError("telemetry fail failed")

    wrapper = bedrock_runtime_api_call_wrapper(FailingFailHandler())

    def wrapped(_op_name, _params):
        raise RuntimeError("bedrock failed")

    with pytest.raises(RuntimeError, match="bedrock failed"):
        wrapper(wrapped, fake_client, ("Converse", _converse_params()), {})


def test_telemetry_setup_error_passes_through_to_bedrock(fake_client):
    class FailingStartHandler:
        def start_llm(self, _invocation):
            raise RuntimeError("telemetry failed")

        def stop_llm(self, _invocation):
            raise AssertionError("stop_llm should not be called")

        def fail_llm(self, _invocation, _error):
            raise AssertionError("fail_llm should not be called")

    wrapper = bedrock_runtime_api_call_wrapper(FailingStartHandler())

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
    result = {
        "stream": FakeStream(events),
        "ResponseMetadata": {"RequestId": "rid"},
    }

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "ConverseStream",
        _converse_params(),
        result,
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


def test_stream_stop_error_does_not_escape_to_application(fake_client):
    class FailingStopHandler:
        def __init__(self):
            self.started_llm = []

        def start_llm(self, invocation):
            self.started_llm.append(invocation)
            return invocation

        def stop_llm(self, _invocation):
            raise RuntimeError("telemetry stop failed")

        def fail_llm(self, _invocation, _error):
            raise AssertionError("fail_llm should not be called")

    handler = FailingStopHandler()
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "ok"},
            }
        },
    ]
    result = {"stream": FakeStream(events)}

    wrapped_result = _call_wrapper(
        handler,
        fake_client,
        "ConverseStream",
        _converse_params(),
        result,
    )

    assert list(wrapped_result["stream"]) == events
    assert len(handler.started_llm) == 1


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
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_model == "amazon.titan-text-express-v1"
    assert invocation.provider == "aws.bedrock"
    assert invocation.request_max_tokens == 64
    assert invocation.input_messages[0].parts[0].content == "hello"
    assert invocation.response_id == "request-789"
    assert invocation.input_tokens == 6
    assert invocation.output_tokens == 9
    assert invocation.output_messages == []


def test_invoke_model_titan_reads_and_replaces_streaming_body(
    stub_handler, fake_client
):
    params = {
        "modelId": "amazon.titan-text-express-v1",
        "body": (
            b'{"inputText":"hello titan","textGenerationConfig":'
            b'{"maxTokenCount":64,"temperature":0.3,"topP":0.8,'
            b'"stopSequences":["END"]}}'
        ),
    }
    response_bytes = (
        b'{"inputTextTokenCount":4,"results":[{"outputText":"titan reply",'
        b'"tokenCount":7,"completionReason":"FINISH"}]}'
    )
    original_body = FakeReadBody(response_bytes)
    result = {
        "body": original_body,
        "ResponseMetadata": {"RequestId": "request-titan"},
    }

    _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModel",
        params,
        result,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.operation == "text_completion"
    assert invocation.request_max_tokens == 64
    assert invocation.request_temperature == 0.3
    assert invocation.request_top_p == 0.8
    assert invocation.request_stop_sequences == ["END"]
    assert invocation.input_tokens == 4
    assert invocation.output_tokens == 7
    assert invocation.response_finish_reasons == ["stop"]
    assert invocation.output_messages[0].parts[0].content == "titan reply"
    assert original_body.closed is True
    assert result["body"].read() == response_bytes


def test_invoke_model_nova_maps_tool_use_response(stub_handler, fake_client):
    params = {
        "modelId": "amazon.nova-pro-v1:0",
        "body": (
            b'{"messages":[{"role":"user","content":[{"text":"weather"}]}],'
            b'"inferenceConfig":{"max_new_tokens":96,"temperature":0.4,'
            b'"topP":0.75,"stopSequences":["stop"]}}'
        ),
    }
    result = {
        "body": (
            b'{"usage":{"inputTokens":10,"outputTokens":5},'
            b'"stopReason":"tool_use","output":{"message":{"role":"assistant",'
            b'"content":[{"text":"checking"},{"toolUse":{"toolUseId":"tool-9",'
            b'"name":"get_weather","input":{"city":"Seattle"}}}]}}}'
        )
    }

    _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModel",
        params,
        result,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_max_tokens == 96
    assert invocation.request_temperature == 0.4
    assert invocation.request_top_p == 0.75
    assert invocation.request_stop_sequences == ["stop"]
    assert invocation.input_messages[0].parts[0].content == "weather"
    assert invocation.input_tokens == 10
    assert invocation.output_tokens == 5
    assert invocation.response_finish_reasons == ["tool_calls"]
    parts = invocation.output_messages[0].parts
    assert parts[0].content == "checking"
    assert isinstance(parts[1], ToolCall)
    assert parts[1].id == "tool-9"
    assert parts[1].arguments == {"city": "Seattle"}


@pytest.mark.parametrize(
    (
        "model_id",
        "request_body",
        "response_body",
        "expected_max_tokens",
        "expected_top_p",
        "expected_input_tokens",
        "expected_output_tokens",
        "expected_output",
        "expected_finish_reason",
    ),
    [
        (
            "cohere.command-r-v1:0",
            b'{"message":"hello cohere","max_tokens":16,"p":0.6}',
            b'{"text":"cohere reply","finish_reason":"COMPLETE"}',
            16,
            0.6,
            None,
            None,
            "cohere reply",
            "stop",
        ),
        (
            "meta.llama3-8b-instruct-v1:0",
            b'{"prompt":"hello llama","max_gen_len":32,"top_p":0.7}',
            b'{"generation":"llama reply","prompt_token_count":3,'
            b'"generation_token_count":4,"stop_reason":"stop"}',
            32,
            0.7,
            3,
            4,
            "llama reply",
            "stop",
        ),
        (
            "mistral.mistral-large-2402-v1:0",
            b'{"prompt":"hello mistral","max_tokens":24,"top_p":0.9,"stop":"</s>"}',
            b'{"outputs":[{"text":"mistral reply","stop_reason":"stop"}]}',
            24,
            0.9,
            None,
            None,
            "mistral reply",
            "stop",
        ),
    ],
)
def test_invoke_model_provider_specific_json_shapes(
    stub_handler,
    fake_client,
    model_id,
    request_body,
    response_body,
    expected_max_tokens,
    expected_top_p,
    expected_input_tokens,
    expected_output_tokens,
    expected_output,
    expected_finish_reason,
):
    params = {"modelId": model_id, "body": request_body}
    result = {"body": response_body}

    _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModel",
        params,
        result,
    )

    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_max_tokens == expected_max_tokens
    assert invocation.request_top_p == expected_top_p
    assert invocation.input_tokens == expected_input_tokens
    assert invocation.output_tokens == expected_output_tokens
    assert invocation.response_finish_reasons == [expected_finish_reason]
    assert invocation.output_messages[0].parts[0].content == expected_output


def test_invoke_model_stream_titan_maps_text_and_metrics(
    stub_handler, fake_client
):
    events = [
        {"chunk": {"bytes": b'{"outputText":"Hel"}'}},
        {
            "chunk": {
                "bytes": (
                    b'{"outputText":"lo","completionReason":"FINISH",'
                    b'"amazon-bedrock-invocationMetrics":'
                    b'{"inputTokenCount":3,"outputTokenCount":2}}'
                )
            }
        },
    ]
    result = {
        "body": FakeStream(events),
        "ResponseMetadata": {"RequestId": "rid"},
    }
    params = {
        "modelId": "amazon.titan-text-express-v1",
        "body": b'{"inputText":"hi"}',
    }

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModelWithResponseStream",
        params,
        result,
    )

    assert list(wrapped_result["body"]) == events
    invocation = stub_handler.stopped_llm[0]
    assert invocation.request_stream is True
    assert invocation.input_tokens == 3
    assert invocation.output_tokens == 2
    assert invocation.response_finish_reasons == ["stop"]
    assert invocation.output_messages[0].parts[0].content == "Hello"


def test_invoke_model_stream_claude_maps_text_tool_and_metrics(
    stub_handler, fake_client
):
    events = [
        {
            "chunk": {
                "bytes": (
                    b'{"type":"message_start","message":{"id":"msg-stream",'
                    b'"role":"assistant","model":"claude-3","usage":'
                    b'{"input_tokens":4}}}'
                )
            }
        },
        {
            "chunk": {
                "bytes": (
                    b'{"type":"content_block_start","index":0,'
                    b'"content_block":{"type":"text","text":""}}'
                )
            }
        },
        {
            "chunk": {
                "bytes": (
                    b'{"type":"content_block_delta","index":0,'
                    b'"delta":{"type":"text_delta","text":"Checking"}}'
                )
            }
        },
        {"chunk": {"bytes": b'{"type":"content_block_stop","index":0}'}},
        {
            "chunk": {
                "bytes": (
                    b'{"type":"content_block_start","index":1,'
                    b'"content_block":{"type":"tool_use","id":"tool-1",'
                    b'"name":"get_weather","input":{}}}'
                )
            }
        },
        {
            "chunk": {
                "bytes": (
                    b'{"type":"content_block_delta","index":1,'
                    b'"delta":{"type":"input_json_delta",'
                    b'"partial_json":"{\\"city\\":\\"Paris\\"}"}}'
                )
            }
        },
        {"chunk": {"bytes": b'{"type":"content_block_stop","index":1}'}},
        {
            "chunk": {
                "bytes": (
                    b'{"type":"message_delta","delta":{"stop_reason":"tool_use"},'
                    b'"usage":{"output_tokens":6}}'
                )
            }
        },
        {
            "chunk": {
                "bytes": (
                    b'{"type":"message_stop","amazon-bedrock-invocationMetrics":'
                    b'{"inputTokenCount":4,"outputTokenCount":6}}'
                )
            }
        },
    ]
    result = {
        "body": FakeStream(events),
        "ResponseMetadata": {"RequestId": "rid"},
    }
    params = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": b'{"messages":[{"role":"user","content":"weather"}]}',
    }

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModelWithResponseStream",
        params,
        result,
    )

    assert list(wrapped_result["body"]) == events
    invocation = stub_handler.stopped_llm[0]
    assert invocation.response_id == "msg-stream"
    assert invocation.response_model_name == "claude-3"
    assert invocation.input_tokens == 4
    assert invocation.output_tokens == 6
    assert invocation.response_finish_reasons == ["tool_calls"]
    parts = invocation.output_messages[0].parts
    assert parts[0].content == "Checking"
    assert isinstance(parts[1], ToolCall)
    assert parts[1].name == "get_weather"
    assert parts[1].arguments == {"city": "Paris"}


def test_converse_stream_close_mid_stream_finalizes_span(
    stub_handler, fake_client
):
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "partial"},
            }
        },
    ]
    raw_stream = FakeStream(events)
    result = {
        "stream": raw_stream,
        "ResponseMetadata": {"RequestId": "close-rid"},
    }

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "ConverseStream",
        _converse_params(),
        result,
    )

    assert next(wrapped_result["stream"]) == events[0]
    assert len(stub_handler.stopped_llm) == 0
    wrapped_result["stream"].close()

    assert raw_stream.closed is True
    assert len(stub_handler.stopped_llm) == 1
    invocation = stub_handler.stopped_llm[0]
    assert invocation.response_id == "close-rid"
    assert invocation.output_messages == []


def test_invoke_model_stream_generic_maps_text_and_finish_reason(
    stub_handler, fake_client
):
    events = [
        {"chunk": {"bytes": b'{"generation":"part "}'}},
        {
            "chunk": {
                "bytes": (
                    b'{"generation":"two","prompt_token_count":5,'
                    b'"generation_token_count":3,"stop_reason":"stop"}'
                )
            }
        },
    ]
    result = {
        "body": FakeStream(events),
        "ResponseMetadata": {"RequestId": "generic-rid"},
    }
    params = {
        "modelId": "meta.llama3-8b-instruct-v1:0",
        "body": b'{"prompt":"hello","max_gen_len":32}',
    }

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModelWithResponseStream",
        params,
        result,
    )

    assert list(wrapped_result["body"]) == events
    invocation = stub_handler.stopped_llm[0]
    assert invocation.response_id == "generic-rid"
    assert invocation.input_tokens == 5
    assert invocation.output_tokens == 3
    assert invocation.response_finish_reasons == ["stop"]
    assert invocation.output_messages[0].parts[0].content == "part two"


def test_invoke_model_stream_always_populates_invocation_messages(
    stub_handler, fake_client
):
    events = [
        {"chunk": {"bytes": b'{"outputText":"secret"}'}},
        {
            "chunk": {
                "bytes": (
                    b'{"completionReason":"FINISH",'
                    b'"amazon-bedrock-invocationMetrics":'
                    b'{"inputTokenCount":8,"outputTokenCount":4}}'
                )
            }
        },
    ]
    result = {"body": FakeStream(events)}
    params = {
        "modelId": "amazon.titan-text-express-v1",
        "body": b'{"inputText":"hello"}',
    }

    wrapped_result = _call_wrapper(
        stub_handler,
        fake_client,
        "InvokeModelWithResponseStream",
        params,
        result,
    )

    assert list(wrapped_result["body"]) == events
    invocation = stub_handler.stopped_llm[0]
    assert invocation.input_messages[0].parts[0].content == "hello"
    assert invocation.output_messages[0].parts[0].content == "secret"
    assert invocation.input_tokens == 8
    assert invocation.output_tokens == 4
    assert invocation.response_finish_reasons == ["stop"]


def test_non_bedrock_runtime_call_is_not_instrumented(stub_handler):
    client = FakeClient(service_name="s3")
    result = {"ok": True}

    wrapped_result = _call_wrapper(
        stub_handler,
        client,
        "ListBuckets",
        {},
        result,
    )

    assert wrapped_result == result
    assert stub_handler.started_llm == []
    assert stub_handler.stopped_llm == []
