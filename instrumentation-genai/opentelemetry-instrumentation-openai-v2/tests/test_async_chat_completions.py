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
# pylint: disable=too-many-locals


import pytest
from openai import APIConnectionError, AsyncOpenAI, NotFoundError

from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)

from .test_utils import (
    assert_all_attributes,
    assert_handler_event,
)


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_with_content(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    response = await async_openai_client.chat.completions.create(
        messages=messages_value, model=llm_model_value, stream=False
    )

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], spans[0])
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": messages_value[0]["content"]}
            ],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": response.choices[0].message.content,
                }
            ],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_no_content(
    span_exporter, log_exporter, async_openai_client, instrument_no_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    response = await async_openai_client.chat.completions.create(
        messages=messages_value, model=llm_model_value, stream=False
    )

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


@pytest.mark.asyncio()
async def test_async_chat_completion_bad_endpoint(
    span_exporter, instrument_no_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    client = AsyncOpenAI(base_url="http://localhost:4242")

    with pytest.raises(APIConnectionError):
        await client.chat.completions.create(
            messages=messages_value,
            model=llm_model_value,
            timeout=0.1,
        )

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0], llm_model_value, server_address="localhost"
    )
    assert 4242 == spans[0].attributes[ServerAttributes.SERVER_PORT]
    assert (
        "APIConnectionError" == spans[0].attributes[ErrorAttributes.ERROR_TYPE]
    )


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_404(
    span_exporter, async_openai_client, instrument_no_content
):
    llm_model_value = "this-model-does-not-exist"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    with pytest.raises(NotFoundError):
        await async_openai_client.chat.completions.create(
            messages=messages_value,
            model=llm_model_value,
        )

    spans = span_exporter.get_finished_spans()

    assert_all_attributes(spans[0], llm_model_value)
    assert "NotFoundError" == spans[0].attributes[ErrorAttributes.ERROR_TYPE]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_extra_params(
    span_exporter, async_openai_client, instrument_no_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    response = await async_openai_client.chat.completions.create(
        messages=messages_value,
        model=llm_model_value,
        seed=42,
        temperature=0.5,
        max_tokens=50,
        stream=False,
        extra_body={"service_tier": "default"},
        response_format={"type": "text"},
    )

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        request_service_tier="default",
        response_service_tier=getattr(response, "service_tier", None),
    )
    assert (
        spans[0].attributes[GenAIAttributes.GEN_AI_OPENAI_REQUEST_SEED] == 42
    )
    assert (
        spans[0].attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    )
    assert spans[0].attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert (
        spans[0].attributes[GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER]
        == "default"
    )
    assert (
        spans[0].attributes[
            GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT
        ]
        == "text"
    )


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_multiple_choices(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    response = await async_openai_client.chat.completions.create(
        messages=messages_value, model=llm_model_value, n=2, stream=False
    )

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], spans[0])
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": messages_value[0]["content"]}
            ],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": response.choices[0].message.content,
                }
            ],
            "finish_reason": "stop",
        },
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": response.choices[1].message.content,
                }
            ],
            "finish_reason": "stop",
        },
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_with_raw_repsonse(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]
    response = (
        await async_openai_client.chat.completions.with_raw_response.create(
            messages=messages_value,
            model=llm_model_value,
        )
    )
    response = response.parse()
    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], spans[0])
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": messages_value[0]["content"]}
            ],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": response.choices[0].message.content,
                }
            ],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_tool_calls_with_content(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    await chat_completion_tool_call(
        span_exporter, log_exporter, async_openai_client, True
    )


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_tool_calls_no_content(
    span_exporter, log_exporter, async_openai_client, instrument_no_content
):
    await chat_completion_tool_call(
        span_exporter, log_exporter, async_openai_client, False
    )


async def chat_completion_tool_call(
    span_exporter, log_exporter, async_openai_client, expect_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [
        {"role": "system", "content": "You're a helpful assistant."},
        {
            "role": "user",
            "content": "What's the weather in Seattle and San Francisco today?",
        },
    ]

    response_0 = await async_openai_client.chat.completions.create(
        messages=messages_value,
        model=llm_model_value,
        tool_choice="auto",
        tools=[get_current_weather_tool_definition()],
    )

    # sanity check
    assert "tool_calls" in response_0.choices[0].finish_reason

    # final request
    messages_value.append(
        {
            "role": "assistant",
            "tool_calls": response_0.choices[0].message.to_dict()[
                "tool_calls"
            ],
        }
    )

    tool_call_result_0 = {
        "role": "tool",
        "content": "50 degrees and raining",
        "tool_call_id": response_0.choices[0].message.tool_calls[0].id,
    }
    tool_call_result_1 = {
        "role": "tool",
        "content": "70 degrees and sunny",
        "tool_call_id": response_0.choices[0].message.tool_calls[1].id,
    }

    messages_value.append(tool_call_result_0)
    messages_value.append(tool_call_result_1)

    response_1 = await async_openai_client.chat.completions.create(
        messages=messages_value, model=llm_model_value
    )

    # sanity check
    assert "stop" in response_1.choices[0].finish_reason

    # validate both calls
    spans = span_exporter.get_finished_spans()
    chat_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    ]
    tool_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value
    ]

    assert len(chat_spans) == 2
    assert len(tool_spans) == 2

    # Verify tool spans
    for span in tool_spans:
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME)
            == "get_current_weather"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_TOOL_TYPE) == "function"
        )
        # Verify parent relationship - should be child of first chat span
        assert span.parent is not None
        assert span.parent.span_id == chat_spans[0].context.span_id

    assert_all_attributes(
        chat_spans[0],
        llm_model_value,
        response_0.id,
        response_0.model,
        response_0.usage.prompt_tokens,
        response_0.usage.completion_tokens,
    )
    assert_all_attributes(
        chat_spans[1],
        llm_model_value,
        response_1.id,
        response_1.model,
        response_1.usage.prompt_tokens,
        response_1.usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    if not expect_content:
        assert len(logs) == 0
        return

    assert len(logs) == 2

    body_call_one = assert_handler_event(logs[0], chat_spans[0])
    assert body_call_one["gen_ai.system_instructions"] == [
        {"type": "text", "content": messages_value[0]["content"]}
    ]
    assert body_call_one["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": messages_value[1]["content"]}
            ],
        }
    ]
    call_one_tool_calls = response_0.choices[0].message.tool_calls
    assert body_call_one["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": call_one_tool_calls[0].id,
                    "name": call_one_tool_calls[0].function.name,
                    "arguments": call_one_tool_calls[0].function.arguments,
                },
                {
                    "type": "tool_call",
                    "id": call_one_tool_calls[1].id,
                    "name": call_one_tool_calls[1].function.name,
                    "arguments": call_one_tool_calls[1].function.arguments,
                },
            ],
            "finish_reason": "tool_calls",
        }
    ]

    body_call_two = assert_handler_event(logs[1], chat_spans[1])
    assert body_call_two["gen_ai.system_instructions"] == [
        {"type": "text", "content": messages_value[0]["content"]}
    ]
    assert body_call_two["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": messages_value[1]["content"]}
            ],
        },
        {"role": "assistant", "parts": [{"type": "text", "content": ""}]},
        {
            "role": "tool",
            "parts": [
                {
                    "type": "text",
                    "content": tool_call_result_0["content"],
                }
            ],
        },
        {
            "role": "tool",
            "parts": [
                {
                    "type": "text",
                    "content": tool_call_result_1["content"],
                }
            ],
        },
    ]
    assert body_call_two["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": response_1.choices[0].message.content,
                }
            ],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_streaming(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    llm_model_value = "gpt-4"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    kwargs = {
        "model": llm_model_value,
        "messages": messages_value,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response_stream_usage = None
    response_stream_model = None
    response_stream_id = None
    response_stream_result = ""
    response = await async_openai_client.chat.completions.create(**kwargs)
    async for chunk in response:
        if chunk.choices:
            response_stream_result += chunk.choices[0].delta.content or ""

        # get the last chunk
        if getattr(chunk, "usage", None):
            response_stream_usage = chunk.usage
            response_stream_model = chunk.model
            response_stream_id = chunk.id

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response_stream_id,
        response_stream_model,
        response_stream_usage.prompt_tokens,
        response_stream_usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], spans[0])
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Say this is a test"}],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": response_stream_result}],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_streaming_not_complete(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    llm_model_value = "gpt-4"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    kwargs = {
        "model": llm_model_value,
        "messages": messages_value,
        "stream": True,
    }

    response_stream_model = None
    response_stream_id = None
    response_stream_result = ""
    response = await async_openai_client.chat.completions.create(**kwargs)
    idx = 0
    async for chunk in response:
        if chunk.choices:
            response_stream_result += chunk.choices[0].delta.content or ""
        if idx == 1:
            # fake a stop
            break

        if chunk.model:
            response_stream_model = chunk.model
        if chunk.id:
            response_stream_id = chunk.id
        idx += 1

    response.close()
    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0], llm_model_value, response_stream_id, response_stream_model
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], spans[0])
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Say this is a test"}],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": response_stream_result}],
            "finish_reason": "error",
        }
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_multiple_choices_streaming(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [
        {"role": "system", "content": "You're a helpful assistant."},
        {
            "role": "user",
            "content": "What's the weather in Seattle and San Francisco today?",
        },
    ]

    response_0 = await async_openai_client.chat.completions.create(
        messages=messages_value,
        model=llm_model_value,
        n=2,
        stream=True,
        stream_options={"include_usage": True},
    )

    # two strings for each choice
    response_stream_result = ["", ""]
    finish_reasons = ["", ""]
    async for chunk in response_0:
        if chunk.choices:
            for choice in chunk.choices:
                response_stream_result[choice.index] += (
                    choice.delta.content or ""
                )
                if choice.finish_reason:
                    finish_reasons[choice.index] = choice.finish_reason

        # get the last chunk
        if getattr(chunk, "usage", None):
            response_stream_usage = chunk.usage
            response_stream_model = chunk.model
            response_stream_id = chunk.id

    # sanity check
    assert "stop" == finish_reasons[0]

    spans = span_exporter.get_finished_spans()
    assert_all_attributes(
        spans[0],
        llm_model_value,
        response_stream_id,
        response_stream_model,
        response_stream_usage.prompt_tokens,
        response_stream_usage.completion_tokens,
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], spans[0])
    assert body["gen_ai.system_instructions"] == [
        {"type": "text", "content": messages_value[0]["content"]}
    ]
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": "What's the weather in Seattle and San Francisco today?",
                }
            ],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "".join(response_stream_result[0]),
                }
            ],
            "finish_reason": "stop",
        },
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "".join(response_stream_result[1]),
                }
            ],
            "finish_reason": "stop",
        },
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_multiple_tools_streaming_with_content(
    span_exporter, log_exporter, async_openai_client, instrument_with_content
):
    await async_chat_completion_multiple_tools_streaming(
        span_exporter, log_exporter, async_openai_client, True
    )


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_multiple_tools_streaming_no_content(
    span_exporter, log_exporter, async_openai_client, instrument_no_content
):
    await async_chat_completion_multiple_tools_streaming(
        span_exporter, log_exporter, async_openai_client, False
    )


@pytest.mark.vcr()
@pytest.mark.asyncio()
async def test_async_chat_completion_streaming_unsampled(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_with_content_unsampled,
):
    llm_model_value = "gpt-4"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    kwargs = {
        "model": llm_model_value,
        "messages": messages_value,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response_stream_result = ""
    response = await async_openai_client.chat.completions.create(**kwargs)
    async for chunk in response:
        if chunk.choices:
            response_stream_result += chunk.choices[0].delta.content or ""

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1

    body = assert_handler_event(logs[0], None)
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Say this is a test"}],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": response_stream_result}],
            "finish_reason": "stop",
        }
    ]

    assert logs[0].log_record.trace_id is not None
    assert logs[0].log_record.span_id is not None
    assert logs[0].log_record.trace_flags == 0


async def async_chat_completion_multiple_tools_streaming(
    span_exporter, log_exporter, async_openai_client, expect_content
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [
        {"role": "system", "content": "You're a helpful assistant."},
        {
            "role": "user",
            "content": "What's the weather in Seattle and San Francisco today?",
        },
    ]

    response = await async_openai_client.chat.completions.create(
        messages=messages_value,
        model=llm_model_value,
        tool_choice="auto",
        tools=[get_current_weather_tool_definition()],
        stream=True,
        stream_options={"include_usage": True},
    )

    finish_reason = None
    # two tools
    tool_names = ["", ""]
    tool_call_ids = ["", ""]
    tool_args = ["", ""]
    async for chunk in response:
        if chunk.choices:
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            for tool_call in chunk.choices[0].delta.tool_calls or []:
                t_idx = tool_call.index
                if tool_call.id:
                    tool_call_ids[t_idx] = tool_call.id
                if tool_call.function:
                    if tool_call.function.arguments:
                        tool_args[t_idx] += tool_call.function.arguments
                    if tool_call.function.name:
                        tool_names[t_idx] = tool_call.function.name

        # get the last chunk
        if getattr(chunk, "usage", None):
            response_stream_usage = chunk.usage
            response_stream_model = chunk.model
            response_stream_id = chunk.id

    # sanity check
    assert "tool_calls" == finish_reason

    spans = span_exporter.get_finished_spans()
    chat_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    ]
    tool_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value
    ]

    assert len(chat_spans) == 1
    assert len(tool_spans) == 2

    assert_all_attributes(
        chat_spans[0],
        llm_model_value,
        response_stream_id,
        response_stream_model,
        response_stream_usage.prompt_tokens,
        response_stream_usage.completion_tokens,
    )

    for span in tool_spans:
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME)
            == "get_current_weather"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_TOOL_TYPE) == "function"
        )
        # Verify parent relationship
        assert span.parent is not None
        assert span.parent.span_id == chat_spans[0].context.span_id

    logs = log_exporter.get_finished_logs()
    if not expect_content:
        assert len(logs) == 0
        return

    assert len(logs) == 1

    body = assert_handler_event(logs[0], chat_spans[0])
    assert body["gen_ai.system_instructions"] == [
        {"type": "text", "content": messages_value[0]["content"]}
    ]
    assert body["gen_ai.input.messages"] == [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": "What's the weather in Seattle and San Francisco today?",
                }
            ],
        }
    ]
    assert body["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": tool_call_ids[0],
                    "name": tool_names[0],
                    "arguments": tool_args[0],
                },
                {
                    "type": "tool_call",
                    "id": tool_call_ids[1],
                    "name": tool_names[1],
                    "arguments": tool_args[1],
                },
            ],
            "finish_reason": "tool_calls",
        }
    ]


def get_current_weather_tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Boston, MA",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
