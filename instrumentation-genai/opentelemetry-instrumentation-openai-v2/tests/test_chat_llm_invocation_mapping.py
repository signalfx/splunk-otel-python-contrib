from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.openai_v2.patch import (
    _build_chat_llm_invocation_from_request,
)


def _base_span_attributes(model: str = "gpt-4o") -> dict:
    return {
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }


def test_build_chat_llm_invocation_simple_messages_with_content():
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 42,
        "n": 2,
        "seed": 123,
        "stop": ["STOP"],
        "stream": True,
        "response_format": {"type": "json_object"},
        "service_tier": "default",
    }

    inv = _build_chat_llm_invocation_from_request(
        kwargs, _base_span_attributes(), capture_content=True
    )

    assert inv.provider == "openai"
    assert inv.framework == "openai-sdk"
    assert inv.request_model == "gpt-4o"

    # sampling / request params
    assert inv.request_temperature == 0.5
    assert inv.request_top_p == 0.9
    assert inv.request_max_tokens == 42
    assert inv.request_choice_count == 2
    assert inv.request_seed == 123
    assert inv.request_stop_sequences == ["STOP"]

    # messages
    assert len(inv.input_messages) == 2
    assert inv.input_messages[0].role == "user"
    assert inv.input_messages[0].parts[0].content == "hello"
    assert inv.input_messages[1].role == "assistant"
    assert inv.input_messages[1].parts[0].content == "hi"

    # extras
    assert inv.attributes["openai.stream"] is True
    assert inv.attributes["openai.response_format"] == "json_object"
    assert inv.attributes["openai.request.service_tier"] == "default"


def test_build_chat_llm_invocation_respects_capture_content_flag():
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": {"value": "hello"}},
                    "world",
                ],
            },
        ],
    }

    inv = _build_chat_llm_invocation_from_request(
        kwargs, _base_span_attributes(), capture_content=False
    )

    assert len(inv.input_messages) == 1
    msg = inv.input_messages[0]
    assert msg.role == "user"
    # when capture_content is False we still create a Text part, but content is empty
    assert len(msg.parts) == 1
    assert getattr(msg.parts[0], "content", None) == ""


def test_build_chat_llm_invocation_stop_scalar_and_functions_and_tools():
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [],
        "stop": "END",
        "tools": [
            {"type": "function", "function": {"name": "foo", "description": ""}},
        ],
        "functions": [
            {"name": "bar", "description": ""},
        ],
    }

    span_attrs = _base_span_attributes(model="gpt-4o-mini")

    inv = _build_chat_llm_invocation_from_request(
        kwargs, span_attrs, capture_content=True
    )

    # stop scalar becomes single-element list
    assert inv.request_stop_sequences == ["END"]

    # tools and legacy functions are both surfaced via request_functions
    assert len(inv.request_functions) == 2
    assert inv.request_functions[0]["type"] == "function"
    assert inv.request_functions[0]["function"]["name"] == "foo"
    assert inv.request_functions[1]["type"] == "function"
    assert inv.request_functions[1]["function"]["name"] == "bar"

