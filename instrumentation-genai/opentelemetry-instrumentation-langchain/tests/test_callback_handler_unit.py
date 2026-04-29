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

"""Unit tests for on_llm_end new message type flag behaviour.

These tests exercise the callback_handler.py branching logic without
any real LLM call or VCR cassettes — all inputs are constructed in-memory.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from opentelemetry.util.genai.types import (
    Text,
    ToolCallRequest,
)

# Import the handler under test
from opentelemetry.instrumentation.langchain.callback_handler import (
    LangchainCallbackHandler,
)
from opentelemetry.util.genai.types import LLMInvocation


# ---------------------------------------------------------------------------
# Minimal stubs to satisfy on_llm_end without a real span pipeline
# ---------------------------------------------------------------------------


def _make_handler():
    """Return a LangchainCallbackHandler with a no-op telemetry handler."""
    telemetry_handler = MagicMock()
    handler = LangchainCallbackHandler.__new__(LangchainCallbackHandler)
    # Minimal attribute set required by on_llm_end
    handler._handler = telemetry_handler
    manager = MagicMock()
    handler._invocation_manager = manager
    return handler, manager


def _make_llm_result(content, finish_reason, tool_calls=None):
    """Build a minimal LangChain LLMResult-like object."""
    gen_info = {"finish_reason": finish_reason}
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    generation = SimpleNamespace(
        message=message,
        generation_info=gen_info,
    )
    return SimpleNamespace(
        generations=[[generation]],
        llm_output={},
    )


def _run_on_llm_end(response, monkeypatch, flag_value=None):
    """Drive on_llm_end and return the LLMInvocation that was set up."""
    if flag_value is not None:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES", flag_value
        )
    else:
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES", raising=False
        )

    handler, manager = _make_handler()
    run_id = uuid4()

    inv = LLMInvocation(request_model="gpt-4o")
    manager.get.return_value = inv

    handler.on_llm_end(response, run_id=run_id)
    return inv


# ---------------------------------------------------------------------------
# Flag OFF — text response
# ---------------------------------------------------------------------------


def test_flag_off_text_response(monkeypatch):
    response = _make_llm_result("Paris is the capital.", "stop")
    inv = _run_on_llm_end(response, monkeypatch, flag_value=None)

    assert inv.output_messages is not None
    assert len(inv.output_messages) == 1
    msg = inv.output_messages[0]
    assert msg.role == "assistant"
    assert msg.finish_reason == "stop"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Text)
    assert msg.parts[0].content == "Paris is the capital."


# ---------------------------------------------------------------------------
# Flag OFF — tool_calls finish reason
# ---------------------------------------------------------------------------


def test_flag_off_tool_calls_finish_reason_emits_empty_text(monkeypatch):
    response = _make_llm_result(
        "",
        "tool_calls",
        tool_calls=[{"id": "c1", "name": "search", "args": {"q": "hi"}}],
    )
    inv = _run_on_llm_end(response, monkeypatch, flag_value=None)

    assert inv.output_messages is not None
    msg = inv.output_messages[0]
    assert msg.finish_reason == "tool_calls"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Text)
    assert msg.parts[0].content == ""


# ---------------------------------------------------------------------------
# Flag ON — tool_calls finish reason produces ToolCallRequest parts
# ---------------------------------------------------------------------------


def test_flag_on_tool_calls_produces_tool_call_request(monkeypatch):
    tool_calls = [
        {"id": "call-1", "name": "get_weather", "args": {"city": "SF"}},
        {"id": "call-2", "name": "lookup_price", "args": {"item": "gpt-4o"}},
    ]
    response = _make_llm_result("", "tool_calls", tool_calls=tool_calls)
    inv = _run_on_llm_end(response, monkeypatch, flag_value="true")

    assert inv.output_messages is not None
    msg = inv.output_messages[0]
    assert msg.finish_reason == "tool_calls"
    assert len(msg.parts) == 2

    part0 = msg.parts[0]
    assert isinstance(part0, ToolCallRequest)
    assert part0.name == "get_weather"
    assert part0.id == "call-1"
    assert part0.arguments == {"city": "SF"}

    part1 = msg.parts[1]
    assert isinstance(part1, ToolCallRequest)
    assert part1.name == "lookup_price"
    assert part1.id == "call-2"


def test_flag_on_tool_calls_empty_list_falls_back_to_text(monkeypatch):
    """When tool_calls is empty even with flag on, fallback to Text."""
    response = _make_llm_result("", "tool_calls", tool_calls=[])
    inv = _run_on_llm_end(response, monkeypatch, flag_value="true")

    msg = inv.output_messages[0]
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Text)


# ---------------------------------------------------------------------------
# Flag ON — normal text response still produces Text
# ---------------------------------------------------------------------------


def test_flag_on_text_response_still_produces_text(monkeypatch):
    response = _make_llm_result("London is in England.", "stop")
    inv = _run_on_llm_end(response, monkeypatch, flag_value="true")

    msg = inv.output_messages[0]
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Text)
    assert msg.parts[0].content == "London is in England."


# ---------------------------------------------------------------------------
# finish_reason propagated to response_finish_reasons
# ---------------------------------------------------------------------------


def test_finish_reason_set_on_invocation(monkeypatch):
    response = _make_llm_result("ok", "stop")
    inv = _run_on_llm_end(response, monkeypatch)
    assert inv.response_finish_reasons == ["stop"]


def test_finish_reason_tool_calls_set_on_invocation(monkeypatch):
    response = _make_llm_result("", "tool_calls")
    inv = _run_on_llm_end(response, monkeypatch)
    assert inv.response_finish_reasons == ["tool_calls"]
