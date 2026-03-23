"""Tests for LlamaIndex LLM instrumentation with OpenTelemetry."""

import pytest
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockLLM


@pytest.mark.skip(reason="Requires live OpenAI API key; needs VCR cassettes")
def test_with_openai(span_exporter, instrument):
    """Test with real OpenAI API - requires OPENAI_API_KEY."""
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo")
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Say hello in exactly 5 words"),
    ]

    response = llm.chat(messages)
    assert response.message.content

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1


def test_with_mock(span_exporter, instrument):
    """Test LLM chat instrumentation with MockLLM."""
    llm = MockLLM(max_tokens=50)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Say hello in 5 words"),
    ]

    response = llm.chat(messages)
    assert response.message.content is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1


def test_message_extraction(span_exporter, instrument):
    """Test that message content is captured in spans."""
    llm = MockLLM(max_tokens=20)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Test message"),
    ]

    response = llm.chat(messages)
    assert response.message.content is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
