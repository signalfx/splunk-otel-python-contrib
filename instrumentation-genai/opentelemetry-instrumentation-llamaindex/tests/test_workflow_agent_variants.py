import asyncio

import pytest
from llama_index.core.agent.workflow import AgentWorkflow, CodeActAgent, FunctionAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class StaticChatLLM(MockLLM):
    def __init__(self, content: str):
        super().__init__(max_tokens=128)
        self._response = ChatMessage(role=MessageRole.ASSISTANT, content=content)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            model_name="mock-static",
        )

    def chat(self, messages, **kwargs):
        return ChatResponse(message=self._response)

    async def achat(self, messages, **kwargs):
        return ChatResponse(message=self._response)

    def stream_chat(self, messages, **kwargs):
        yield ChatResponse(message=self._response, delta=self._response.content)


class FunctionCallingLLM(StaticChatLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name="mock-function",
        )

    async def achat_with_tools(self, chat_history, tools, **kwargs):
        return await self.achat(chat_history, **kwargs)

    async def astream_chat_with_tools(self, chat_history, tools, **kwargs):
        return await self.astream_chat(chat_history, **kwargs)

    def get_tool_calls_from_response(self, response, **kwargs):
        return []


def setup_telemetry():
    memory_exporter = InMemorySpanExporter()
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    metric_reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    return tracer_provider, memory_exporter


def _assert_agent_and_workflow_spans(spans):
    op_names = {span.attributes.get("gen_ai.operation.name") for span in spans}
    assert "invoke_agent" in op_names
    assert "invoke_workflow" in op_names


@pytest.fixture(scope="module")
def instrumented_telemetry():
    tracer_provider, memory_exporter = setup_telemetry()

    instrumentor = LlamaindexInstrumentor()
    instrumentor._is_instrumented_by_opentelemetry = False
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=metrics.get_meter_provider()
    )

    return memory_exporter


@pytest.mark.asyncio
async def test_function_agent_emits_spans(monkeypatch, instrumented_telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    memory_exporter = instrumented_telemetry
    memory_exporter.clear()

    agent = FunctionAgent(
        name="FunctionAgent",
        llm=FunctionCallingLLM("Done."),
        tools=[],
        streaming=False,
    )

    handler = agent.run(user_msg="Say hello.")
    await handler
    await asyncio.sleep(0.1)

    spans = memory_exporter.get_finished_spans()
    _assert_agent_and_workflow_spans(spans)


@pytest.mark.asyncio
async def test_codeact_agent_emits_spans(monkeypatch, instrumented_telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    memory_exporter = instrumented_telemetry
    memory_exporter.clear()

    async def code_execute_fn(code: str):
        return {"result": code}

    agent = CodeActAgent(
        code_execute_fn=code_execute_fn,
        llm=StaticChatLLM("No code needed."),
        streaming=False,
    )

    handler = agent.run(user_msg="Explain what Python is.")
    await handler
    await asyncio.sleep(0.1)

    spans = memory_exporter.get_finished_spans()
    _assert_agent_and_workflow_spans(spans)


@pytest.mark.asyncio
async def test_agent_workflow_emits_agent_span(monkeypatch, instrumented_telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    memory_exporter = instrumented_telemetry
    memory_exporter.clear()

    agent = FunctionAgent(
        name="SingleAgent",
        llm=FunctionCallingLLM("Done."),
        tools=[],
        streaming=False,
    )
    workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)

    await workflow.run(user_msg="Hello.")
    await asyncio.sleep(0.1)

    op_names = {
        span.attributes.get("gen_ai.operation.name")
        for span in memory_exporter.get_finished_spans()
    }
    assert "invoke_agent" in op_names
    assert "invoke_workflow" in op_names
