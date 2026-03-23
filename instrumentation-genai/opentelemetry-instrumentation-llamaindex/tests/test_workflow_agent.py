"""
Test Workflow-based agent instrumentation.

This test validates that workflow event streaming captures agent steps
and tool calls using a ReActAgent with a mock LLM.
"""

import asyncio
from typing import List

import pytest
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import MockLLM
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class SequenceMockLLM(MockLLM):
    responses: List[ChatMessage] = []
    response_index: int = 0

    def __init__(self, responses: List[ChatMessage], max_tokens: int = 256):
        super().__init__(max_tokens=max_tokens)
        self.responses = responses
        self.response_index = 0

    def chat(self, messages, **kwargs):
        from llama_index.core.base.llms.types import ChatResponse

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return ChatResponse(message=response)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Done.")
        )

    async def achat(self, messages, **kwargs):
        from llama_index.core.base.llms.types import ChatResponse

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return ChatResponse(message=response)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Done.")
        )

    def stream_chat(self, messages, **kwargs):
        from llama_index.core.base.llms.types import ChatResponse

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            yield ChatResponse(message=response, delta=response.content)
        else:
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."),
                delta="Done.",
            )

    async def astream_chat(self, messages, **kwargs):
        from llama_index.core.base.llms.types import ChatResponse

        async def gen():
            if self.response_index < len(self.responses):
                response = self.responses[self.response_index]
                self.response_index += 1
                yield ChatResponse(message=response, delta=response.content)
            else:
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."),
                    delta="Done.",
                )

        return gen()


@pytest.mark.asyncio
async def test_workflow_agent(span_exporter, instrument):
    """Test ReActAgent workflow instrumentation with mock LLM."""
    mock_responses = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(
                "Thought: I need to multiply 5 by 3 first.\n"
                'Action: multiply\nAction Input: {"a": 5, "b": 3}'
            ),
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(
                "Thought: The result is 15. Now add 2.\n"
                'Action: add\nAction Input: {"a": 15, "b": 2}'
            ),
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=("Thought: The final result is 17.\nAnswer: The result is 17."),
        ),
    ]
    Settings.llm = SequenceMockLLM(responses=mock_responses, max_tokens=256)

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)

    agent = ReActAgent(
        tools=[multiply_tool, add_tool],
        llm=Settings.llm,
        verbose=False,
    )

    handler = agent.run(user_msg="Calculate 5 times 3, then add 2 to the result")
    result = await handler
    await asyncio.sleep(0.5)

    assert result.response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
