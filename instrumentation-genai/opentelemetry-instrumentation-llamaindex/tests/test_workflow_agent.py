"""
Test Workflow-based agent instrumentation.

This test validates that workflow event streaming captures agent steps and tool calls.
"""
import asyncio
import pytest
from typing import List
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from opentelemetry import trace
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def setup_telemetry():
    """Setup OpenTelemetry with console exporter."""
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return tracer_provider


class SequenceMockLLM(MockLLM):
    responses: List[ChatMessage] = []
    response_index: int = 0
    
    def __init__(self, responses: List[ChatMessage], max_tokens: int = 256):
        super().__init__(max_tokens=max_tokens)
        self.responses = responses
        self.response_index = 0

    def chat(self, messages, **kwargs):
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            from llama_index.core.base.llms.types import ChatResponse
            return ChatResponse(message=response)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."))

    async def achat(self, messages, **kwargs):
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            from llama_index.core.base.llms.types import ChatResponse
            return ChatResponse(message=response)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."))

    def stream_chat(self, messages, **kwargs):
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            from llama_index.core.base.llms.types import ChatResponseGen, ChatResponse
            # Yield a single response chunk
            yield ChatResponse(message=response, delta=response.content)
        else:
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."), delta="Done.")

    async def astream_chat(self, messages, **kwargs):
        async def gen():
            if self.response_index < len(self.responses):
                response = self.responses[self.response_index]
                self.response_index += 1
                from llama_index.core.base.llms.types import ChatResponse
                # Yield a single response chunk
                yield ChatResponse(message=response, delta=response.content)
            else:
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."), delta="Done.")
        
        return gen()


@pytest.mark.asyncio
async def test_workflow_agent():
    """Test Workflow-based agent instrumentation."""
    
    print("=" * 80)
    print("Setting up telemetry...")
    print("=" * 80)
    tracer_provider = setup_telemetry()

    # Setup Mock LLM
    mock_responses = [
        # Step 1: Decide to multiply
        ChatMessage(role=MessageRole.ASSISTANT, content="""Thought: I need to multiply 5 by 3 first.
Action: multiply
Action Input: {"a": 5, "b": 3}"""),
        
        # Step 2: Decide to add
        ChatMessage(role=MessageRole.ASSISTANT, content="""Thought: The result is 15. Now I need to add 2 to 15.
Action: add
Action Input: {"a": 15, "b": 2}"""),

        # Step 3: Final Answer
        ChatMessage(role=MessageRole.ASSISTANT, content="""Thought: The final result is 17.
Answer: The result is 17."""),
    ]
    Settings.llm = SequenceMockLLM(responses=mock_responses, max_tokens=256)

    # Instrument
    print("\n" + "=" * 80)
    print("Instrumenting LlamaIndex...")
    print("=" * 80)
    instrumentor = LlamaindexInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    # Create tools
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)

    print("\n" + "=" * 80)
    print("Creating Workflow-based ReActAgent...")
    print("=" * 80)
    agent = ReActAgent(tools=[multiply_tool, add_tool], llm=Settings.llm, verbose=True)

    print("\n" + "=" * 80)
    print("Running agent task (should see AgentInvocation -> ToolCall spans)...")
    print("=" * 80)
    
    handler = agent.run(user_msg="Calculate 5 times 3, then add 2 to the result")
    result = await handler

    # Give background instrumentation task time to complete
    await asyncio.sleep(0.5)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Response: {result.response.content}")

    print("\n" + "=" * 80)
    print("✓ Test completed!")
    print("=" * 80)
    print("\nExpected trace structure:")
    print("  AgentInvocation (gen_ai.agent.name=agent.Agent)")
    print("    ├─ LLMInvocation")
    print("    ├─ ToolCall (gen_ai.tool.name=multiply)")
    print("    ├─ ToolCall (gen_ai.tool.name=add)")
    print("    └─ LLMInvocation (final answer)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_workflow_agent())
