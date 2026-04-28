"""
Integration test with Circuit LLM and ReActAgent.

Requires live Circuit API credentials. Skipped in CI.

To run manually:
    export LLM_TOKEN_URL=... LLM_CLIENT_ID=... LLM_CLIENT_SECRET=... LLM_BASE_URL=...
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS=true
    pytest tests/test_circuit_agent.py -v -p no:deepeval -k test_circuit
"""

import asyncio
import json
import os
from typing import Any

import pytest
import requests

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import (
    CustomLLM,
    ChatMessage,
    MessageRole,
    ChatResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.tools import FunctionTool


# ---------------------------------------------------------------------------
# Circuit LLM
# ---------------------------------------------------------------------------


class CircuITLLM(CustomLLM):
    """Custom LLM for Circuit API."""

    api_url: str
    token_manager: Any
    app_key: str | None = None
    model_name: str = "gpt-5-nano"
    temperature: float = 0.0
    max_tokens: int = 4096

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=128000,
            num_output=self.max_tokens,
            is_function_calling_model=True,
        )

    def _do_chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        access_token = self.token_manager.get_token()
        api_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        payload: dict[str, Any] = {
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if self.app_key:
            payload["user"] = json.dumps({"appkey": self.app_key})

        response = requests.post(
            self.api_url,
            headers={"api-key": access_token, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=result,
        )

    @llm_chat_callback()
    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._do_chat(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: list[ChatMessage], **kwargs: Any):
        response = self._do_chat(messages, **kwargs)
        yield response

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Use chat() instead")

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Not supported")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 72°F."


def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"The current time in {timezone} is 3:45 PM."


def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return "Result: 4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_token_manager():
    """Create OAuth2TokenManager from environment variables."""
    # Import from examples utility
    import sys

    examples_path = os.path.join(os.path.dirname(__file__), "..", "examples")
    sys.path.insert(0, examples_path)
    from util import OAuth2TokenManager

    return OAuth2TokenManager(
        token_url=os.environ.get("LLM_TOKEN_URL", ""),
        client_id=os.environ.get("LLM_CLIENT_ID", ""),
        client_secret=os.environ.get("LLM_CLIENT_SECRET", ""),
        scope=os.environ.get("LLM_SCOPE"),
    )


_requires_circuit = pytest.mark.skipif(
    not os.environ.get("LLM_CLIENT_ID"),
    reason="Requires live Circuit API credentials (LLM_CLIENT_ID, LLM_BASE_URL, etc.)",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_requires_circuit
@pytest.mark.asyncio
async def test_circuit_agent_attributes(span_exporter, instrument):
    """End-to-end test: Circuit LLM + ReActAgent captures all expected attributes.

    Validates gen_ai.response.model, gen_ai.response.finish_reasons,
    gen_ai.tool.definitions, gen_ai.request.max_tokens, and token usage.
    """
    orig_tool_flag = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS"
    )
    orig_content_flag = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    )
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS"] = "true"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

    try:
        token_manager = _get_token_manager()
        model = os.environ.get("LLM_MODEL", "gpt-5-nano")
        base_url = os.environ.get("LLM_BASE_URL", "")
        app_key = os.environ.get("LLM_APP_KEY", "")

        llm = CircuITLLM(
            api_url=base_url,
            token_manager=token_manager,
            app_key=app_key,
            model_name=model,
        )
        Settings.llm = llm

        tools = [
            FunctionTool.from_defaults(fn=get_weather),
            FunctionTool.from_defaults(fn=get_time),
            FunctionTool.from_defaults(fn=calculate),
        ]

        agent = ReActAgent(tools=tools, llm=llm, verbose=False)

        handler = agent.run(user_msg="What is 2 + 2?")
        result = await handler
        await asyncio.sleep(0.5)

        assert result.response is not None

        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1

        # Find LLM chat spans
        llm_spans = [
            s
            for s in spans
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "chat"
        ]
        assert len(llm_spans) >= 1, "Expected at least one LLM chat span"

        attrs = dict(llm_spans[0].attributes)

        # Response model from raw response
        assert "gen_ai.response.model" in attrs

        # Finish reasons
        assert "gen_ai.response.finish_reasons" in attrs

        # Token usage
        assert attrs.get("gen_ai.usage.input_tokens") is not None
        assert attrs.get("gen_ai.usage.output_tokens") is not None

        # Max tokens
        assert attrs.get("gen_ai.request.max_tokens") == 4096

        # Tool definitions
        tool_defs_raw = attrs.get("gen_ai.tool.definitions")
        assert tool_defs_raw is not None, "gen_ai.tool.definitions should be set"
        tool_defs = json.loads(tool_defs_raw)
        tool_names = [t["name"] for t in tool_defs]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names

    finally:
        if orig_tool_flag is None:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS", None)
        else:
            os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS"] = (
                orig_tool_flag
            )
        if orig_content_flag is None:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
        else:
            os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
                orig_content_flag
            )
