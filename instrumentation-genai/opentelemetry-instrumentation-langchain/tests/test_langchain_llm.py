"""Minimal LangChain LLM instrumentation test.

Rewritten from scratch to perform only essential validation of the current
LangChain callback handler integration with util-genai types. Intentional
omission of former expansive coverage (logs, tool flows, exhaustive metrics)
to keep the test stable and low‑maintenance while still proving:

1. A chat invocation succeeds using the recorded VCR cassette.
2. A span is emitted with GenAI semantic convention attributes for a chat op.
3. Core request/response model attributes exist and are plausible.
4. Metrics (duration at minimum) are produced and contain at least one data point.

If token usage data points exist they are sanity‑checked but not required.
"""

from __future__ import annotations

# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false

import json
from typing import Any, List
import pytest
from pytest import MonkeyPatch
from pydantic import SecretStr

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.sdk.trace import ReadableSpan  # test-only type reference
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from opentelemetry.util.genai.attributes import (
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_REQUEST_STREAMING,
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
)


CHAT = gen_ai_attributes.GenAiOperationNameValues.CHAT.value


@pytest.mark.vcr()
def test_langchain_call(
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_with_content: Any,
    monkeypatch: MonkeyPatch,
):
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("APPKEY", "test-app-key")
    model = "gpt-4o-mini"
    llm = ChatOpenAI(
        temperature=0.0,
        max_tokens=100,
        api_key=SecretStr("test-api-key"),
        base_url="https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini",
        model=model,
        default_headers={"api-key": "test-api-key"},
        model_kwargs={"user": json.dumps({"appkey": "test-app-key"})},
    )
    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="What is the capital of France?"),
    ]

    # Act
    response = llm.invoke(messages)

    # Basic functional assertion
    content = response.content
    if isinstance(content, list):  # some providers may return list segments
        content_text = " ".join(str(c) for c in content)
    else:
        content_text = str(content)
    assert "Paris" in content_text

    # Spans
    spans: List[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore[assignment]
    assert spans, "Expected at least one span"
    chat_span = None
    for s in spans:
        attrs_obj = getattr(s, "attributes", None)
        op_name = None
        try:
            if attrs_obj is not None:
                op_name = attrs_obj.get(gen_ai_attributes.GEN_AI_OPERATION_NAME)
        except Exception:
            op_name = None
        if op_name == CHAT:
            chat_span = s
            break
    assert chat_span is not None, "No chat operation span found"

    # Span attribute sanity
    attrs = getattr(chat_span, "attributes", {})
    
    # Print raw span output
    print("\n=== LLM Span (Raw) ===")
    print(chat_span.to_json(indent=2))
    print("======================\n")
    
    assert attrs.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == CHAT
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_MODEL) == model
    # Response model can differ (provider adds version); only assert presence
    assert attrs.get(gen_ai_attributes.GEN_AI_RESPONSE_MODEL) is not None
    # --- New semconv attributes (HYBIM-559) ---
    # gen_ai.request.temperature
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE) == 0.0
    # gen_ai.request.max_tokens
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS) == 100
    # gen_ai.response.finish_reasons (cassette returns "stop")
    assert attrs.get(gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)

    # --- Streaming attributes ---
    # For non-streaming .invoke() calls, request_streaming is False
    # and time_to_first_token is not captured
    streaming_val = attrs.get(GEN_AI_REQUEST_STREAMING)
    ttft_val = attrs.get(GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN)
    # For .invoke(), streaming is False and no TTFT
    assert streaming_val is False, "request_streaming should be False for non-streaming calls"
    assert ttft_val is None, "time_to_first_token should be None for non-streaming calls"

    # If token usage captured ensure they are non-negative integers
    for key in (
        gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
        gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
    ):
        tok_val = attrs.get(key)
        if tok_val is not None:
            assert isinstance(tok_val, int) and tok_val >= 0

    # Metrics – ensure at least duration histogram present with >=1 point
    metrics_data = metric_reader.get_metrics_data()
    found_duration = False
    if metrics_data:
        for rm in getattr(metrics_data, "resource_metrics", []) or []:
            for scope in getattr(rm, "scope_metrics", []) or []:
                for metric in getattr(scope, "metrics", []) or []:
                    if metric.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION:
                        dps = getattr(metric.data, "data_points", [])
                        if dps:
                            assert dps[0].sum >= 0
                            found_duration = True
    assert found_duration, "Duration metric missing"

    # Do not fail test on absence of token usage metrics – optional.


# --------------- Tool definitions test (HYBIM-559) ---------------


@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


@pytest.mark.vcr()
def test_langchain_call_with_tools(
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_with_content: Any,
    monkeypatch: MonkeyPatch,
):
    """Verify gen_ai.tool.definitions appears on chat span when tools are bound
    and both CAPTURE_MESSAGE_CONTENT and CAPTURE_TOOL_DEFINITIONS are enabled."""
    # Set env vars BEFORE any invocation so _refresh_capture_content picks them up
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("APPKEY", "test-app-key")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS", "true")
    # Force handler singleton to re-read env
    import opentelemetry.util.genai.handler as _h

    if hasattr(_h.get_telemetry_handler, "_default_handler"):
        setattr(_h.get_telemetry_handler, "_default_handler", None)
    model = "gpt-4o-mini"
    llm = ChatOpenAI(
        temperature=0.1,
        api_key=SecretStr("test-api-key"),
        base_url="https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini",
        model=model,
        default_headers={"api-key": "test-api-key"},
        model_kwargs={"user": json.dumps({"appkey": "test-app-key"})},
    )
    llm_with_tools = llm.bind_tools([add, multiply])
    _response = llm_with_tools.invoke("Please add 2 and 3, then multiply 2 and 3.")

    spans: List[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore[assignment]
    assert spans, "Expected at least one span"

    # Find the first chat span (the LLM call that had tools bound)
    chat_spans = [
        s
        for s in spans
        if getattr(s, "attributes", {}).get(gen_ai_attributes.GEN_AI_OPERATION_NAME)
        == CHAT
    ]
    assert chat_spans, "No chat operation span found"
    first_chat = chat_spans[0]
    attrs = getattr(first_chat, "attributes", {})

    # gen_ai.request.temperature
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE) == 0.1

    # gen_ai.response.finish_reasons (first call returns "tool_calls")
    finish_reasons = attrs.get(gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS)
    assert finish_reasons == ("tool_calls",)

    # gen_ai.tool.definitions — JSON string containing tool schemas
    tool_defs_raw = attrs.get(GEN_AI_TOOL_DEFINITIONS)
    assert tool_defs_raw is not None, (
        "gen_ai.tool.definitions should be present when both "
        "CAPTURE_MESSAGE_CONTENT and CAPTURE_TOOL_DEFINITIONS are enabled"
    )
    tool_defs = json.loads(tool_defs_raw)
    assert isinstance(tool_defs, list)
    # Full tool definition structure with type: "function" wrapper
    tool_names = set()
    for t in tool_defs:
        if t.get("type") == "function" and "function" in t:
            tool_names.add(t["function"]["name"])
        elif "name" in t:
            tool_names.add(t["name"])
    assert "add" in tool_names
    assert "multiply" in tool_names


# --------------- Streaming test (TTFT) ---------------


@pytest.mark.vcr()
def test_langchain_streaming_call(
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_with_content: Any,
    monkeypatch: MonkeyPatch,
):
    """Verify streaming calls capture gen_ai.request.streaming=True and TTFT."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("APPKEY", "test-app-key")
    model = "gpt-4o-mini"
    llm = ChatOpenAI(
        temperature=0.0,
        max_tokens=50,
        api_key=SecretStr("test-api-key"),
        base_url="https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini",
        model=model,
        default_headers={"api-key": "test-api-key"},
        model_kwargs={"user": json.dumps({"appkey": "test-app-key"})},
        streaming=True,  # Enable streaming
    )
    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="Say hello in one word."),
    ]

    # Act - use stream() to trigger on_llm_new_token callbacks
    response_content = ""
    for chunk in llm.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            response_content += chunk.content

    # Basic functional assertion
    assert response_content, "Expected some response content from streaming"

    # Find the chat span
    spans: List[ReadableSpan] = span_exporter.get_finished_spans()
    assert spans, "Expected at least one span"
    chat_span = None
    for s in spans:
        attrs_obj = getattr(s, "attributes", None)
        if attrs_obj and attrs_obj.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == CHAT:
            chat_span = s
            break
    assert chat_span is not None, "No chat operation span found"

    attrs = getattr(chat_span, "attributes", {})

    # Print raw span output
    print("\n=== Streaming LLM Span (Raw) ===")
    print(chat_span.to_json(indent=2))
    print("================================\n")

    # --- Streaming attributes ---
    streaming_val = attrs.get(GEN_AI_REQUEST_STREAMING)
    ttft_val = attrs.get(GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN)

    # For streaming calls, these should be set
    assert streaming_val is True, f"request_streaming should be True for streaming calls, got {streaming_val}"
    assert ttft_val is not None, "time_to_first_token should be present for streaming calls"
    assert isinstance(ttft_val, (int, float)), f"TTFT should be numeric, got {type(ttft_val)}"
    assert ttft_val >= 0, f"TTFT should be non-negative, got {ttft_val}"
    print(f"TTFT: {ttft_val:.4f} seconds")