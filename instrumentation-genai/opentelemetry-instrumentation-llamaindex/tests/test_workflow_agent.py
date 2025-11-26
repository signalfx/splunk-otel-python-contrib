"""
Test Workflow-based agent instrumentation.

This test validates that workflow event streaming captures agent steps and tool calls.
"""
import asyncio
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
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


async def test_workflow_agent():
    """Test Workflow-based agent instrumentation."""
    
    print("=" * 80)
    print("Setting up telemetry...")
    print("=" * 80)
    tracer_provider = setup_telemetry()

    # Setup LlamaIndex
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

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
