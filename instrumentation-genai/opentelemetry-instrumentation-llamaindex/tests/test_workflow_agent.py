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
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def setup_telemetry_with_memory():
    """Setup OpenTelemetry with in-memory exporter to capture spans and metrics."""
    # Setup traces
    memory_exporter = InMemorySpanExporter()
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    # Setup metrics
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    return tracer_provider, memory_exporter, metric_reader


def print_span_hierarchy(spans):
    """Print span hierarchy showing parent-child relationships."""
    print("\n" + "=" * 80)
    print("SPAN HIERARCHY")
    print("=" * 80)

    # Build a map of span_id -> span
    span_map = {span.context.span_id: span for span in spans}

    # Find root spans (no parent)
    root_spans = [
        span
        for span in spans
        if span.parent is None or span.parent.span_id not in span_map
    ]

    def print_span_tree(span, indent=0):
        prefix = "  " * indent
        op_name = span.attributes.get("gen_ai.operation.name", span.name)
        span_type = ""

        if "workflow" in span.name.lower():
            span_type = "Workflow"
            details = f"name={span.attributes.get('gen_ai.workflow.name', 'N/A')}"
        elif "agent" in span.name.lower():
            span_type = "AgentInvocation"
            details = f"name={span.attributes.get('gen_ai.agent.name', 'N/A')}"
        elif "tool" in span.name.lower():
            span_type = "ToolCall"
            details = f"name={span.attributes.get('gen_ai.tool.name', 'N/A')}"
        elif "chat" in span.name.lower() or "llm" in span.name.lower():
            span_type = "LLMInvocation"
            details = f"model={span.attributes.get('gen_ai.request.model', 'N/A')}"
        else:
            span_type = span.name
            details = f"operation={op_name}"

        print(f"{prefix}└─ {span_type} ({details})")

        # Find and print children
        children = [
            s for s in spans if s.parent and s.parent.span_id == span.context.span_id
        ]
        for child in children:
            print_span_tree(child, indent + 1)

    for root in root_spans:
        print_span_tree(root)

    print("=" * 80)


def print_metrics(metric_reader):
    """Print captured metrics."""
    print("\n" + "=" * 80)
    print("WORKFLOW METRICS")
    print("=" * 80)

    metrics_data = metric_reader.get_metrics_data()

    if not metrics_data or not metrics_data.resource_metrics:
        print("No metrics captured")
        print("=" * 80)
        return

    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                print(f"\nMetric: {metric.name}")
                print(f"  Description: {metric.description}")
                print(f"  Unit: {metric.unit}")

                for data_point in metric.data.data_points:
                    attrs = dict(data_point.attributes) if data_point.attributes else {}

                    # Format attributes for display
                    attr_str = ", ".join([f"{k}={v}" for k, v in attrs.items()])

                    # Get the value based on metric type
                    if hasattr(data_point, "value"):
                        value = data_point.value
                    elif hasattr(data_point, "sum"):
                        value = data_point.sum
                    elif hasattr(data_point, "count"):
                        value = f"count={data_point.count}, sum={data_point.sum}"
                    else:
                        value = "N/A"

                    print(f"    [{attr_str}] = {value}")

    print("=" * 80)


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
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Done.")
        )

    async def achat(self, messages, **kwargs):
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            from llama_index.core.base.llms.types import ChatResponse

            return ChatResponse(message=response)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Done.")
        )

    def stream_chat(self, messages, **kwargs):
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            from llama_index.core.base.llms.types import ChatResponse

            # Yield a single response chunk
            yield ChatResponse(message=response, delta=response.content)
        else:
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."),
                delta="Done.",
            )

    async def astream_chat(self, messages, **kwargs):
        async def gen():
            if self.response_index < len(self.responses):
                response = self.responses[self.response_index]
                self.response_index += 1
                from llama_index.core.base.llms.types import ChatResponse

                # Yield a single response chunk
                yield ChatResponse(message=response, delta=response.content)
            else:
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="Done."),
                    delta="Done.",
                )

        return gen()


@pytest.mark.asyncio
async def test_workflow_agent(monkeypatch):
    """Test Workflow-based agent instrumentation."""

    # Enable metric emitter
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event")

    print("=" * 80)
    print("Setting up telemetry...")
    print("=" * 80)
    tracer_provider, memory_exporter, metric_reader = setup_telemetry_with_memory()

    # Setup Mock LLM
    mock_responses = [
        # Step 1: Decide to multiply
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""Thought: I need to multiply 5 by 3 first.
Action: multiply
Action Input: {"a": 5, "b": 3}""",
        ),
        # Step 2: Decide to add
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""Thought: The result is 15. Now I need to add 2 to 15.
Action: add
Action Input: {"a": 15, "b": 2}""",
        ),
        # Step 3: Final Answer
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""Thought: The final result is 17.
Answer: The result is 17.""",
        ),
    ]
    Settings.llm = SequenceMockLLM(responses=mock_responses, max_tokens=256)

    # Instrument
    print("\n" + "=" * 80)
    print("Instrumenting LlamaIndex...")
    print("=" * 80)
    instrumentor = LlamaindexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=metrics.get_meter_provider()
    )

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

    # Print the actual span hierarchy
    spans = memory_exporter.get_finished_spans()
    print(f"\nTotal spans captured: {len(spans)}")
    print_span_hierarchy(spans)

    # Print captured metrics
    print_metrics(metric_reader)

    print("\n" + "=" * 80)
    print("✓ Test completed!")
    print("=" * 80)
    print("\nExpected trace structure:")
    print("  Workflow (ReActAgent Workflow)")
    print("    └─ AgentInvocation (agent.ReActAgent)")
    print("        ├─ LLMInvocation")
    print("        ├─ ToolCall (multiply)")
    print("        ├─ ToolCall (add)")
    print("        └─ LLMInvocation (final answer)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_workflow_agent())
