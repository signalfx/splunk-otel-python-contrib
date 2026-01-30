import asyncio
import os

import pytest
from llama_index.core.agent.workflow import AgentWorkflow, CodeActAgent, FunctionAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.llm import ToolSelection
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
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
        return ChatResponse(
            message=self._response,
            raw={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )

    async def achat(self, messages, **kwargs):
        return ChatResponse(
            message=self._response,
            raw={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )

    def stream_chat(self, messages, **kwargs):
        yield ChatResponse(
            message=self._response,
            delta=self._response.content,
            raw={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )


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


class HandoffSequenceLLM(FunctionCallingLLM):
    def __init__(self, content: str, tool_calls):
        super().__init__(content)
        self._tool_calls = list(tool_calls)

    def get_tool_calls_from_response(self, response, **kwargs):
        if not self._tool_calls:
            return []
        entry = self._tool_calls.pop(0)
        if entry is None:
            return []
        tool_name, tool_kwargs = entry
        return [
            ToolSelection(
                tool_id="handoff-1",
                tool_name=tool_name,
                tool_kwargs=tool_kwargs,
            )
        ]


def setup_telemetry():
    memory_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    meter_provider = MeterProvider()

    return tracer_provider, meter_provider, memory_exporter


def _assert_agent_and_workflow_spans(spans):
    op_names = {span.attributes.get("gen_ai.operation.name") for span in spans}
    assert "invoke_agent" in op_names
    assert "invoke_workflow" in op_names


def _assert_agent_and_workflow_attributes(spans):
    workflow_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_workflow"
    ]
    agent_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    ]
    assert any(span.attributes.get("gen_ai.workflow.name") for span in workflow_spans)
    assert any(span.attributes.get("gen_ai.agent.name") for span in agent_spans)


@pytest.fixture(scope="module")
def instrumented_telemetry():
    tracer_provider, meter_provider, memory_exporter = setup_telemetry()
    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"

    instrumentor = LlamaindexInstrumentor()
    instrumentor._is_instrumented_by_opentelemetry = False
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    return memory_exporter


@pytest.mark.asyncio
async def test_function_agent_emits_spans(monkeypatch, instrumented_telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")
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
    _assert_agent_and_workflow_attributes(spans)


@pytest.mark.asyncio
async def test_codeact_agent_emits_spans(monkeypatch, instrumented_telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")
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
    _assert_agent_and_workflow_attributes(spans)


@pytest.mark.asyncio
async def test_agent_workflow_emits_agent_span(monkeypatch, instrumented_telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")
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
    _assert_agent_and_workflow_attributes(memory_exporter.get_finished_spans())


@pytest.mark.asyncio
async def test_multi_agent_workflow_example_emits_spans(
    monkeypatch, instrumented_telemetry
):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")
    memory_exporter = instrumented_telemetry
    memory_exporter.clear()

    llm = HandoffSequenceLLM(
        "Done.",
        [
            ("handoff", {"to_agent": "WriteAgent", "reason": "Draft ready."}),
            ("handoff", {"to_agent": "ReviewAgent", "reason": "Needs review."}),
            None,
        ],
    )

    research_agent = FunctionAgent(
        name="ResearchAgent",
        description="Search the web and record notes.",
        system_prompt="You are a researcher. Hand off to WriteAgent when ready.",
        llm=llm,
        tools=[],
        can_handoff_to=["WriteAgent"],
        streaming=False,
    )
    write_agent = FunctionAgent(
        name="WriteAgent",
        description="Writes a markdown report from the notes.",
        system_prompt="You are a writer. Ask ReviewAgent for feedback when done.",
        llm=llm,
        tools=[],
        can_handoff_to=["ReviewAgent", "ResearchAgent"],
        streaming=False,
    )
    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Reviews a report and gives feedback.",
        system_prompt="You are a reviewer.",
        llm=llm,
        tools=[],
        can_handoff_to=["WriteAgent"],
        streaming=False,
    )

    agent_workflow = AgentWorkflow(
        agents=[research_agent, write_agent, review_agent],
        root_agent=research_agent.name,
        initial_state={
            "research_notes": {},
            "report_content": "Not written yet.",
            "review": "Review required.",
        },
    )

    await agent_workflow.run(user_msg="Write me a report on the history of the web.")
    await asyncio.sleep(0.1)

    op_names = {
        span.attributes.get("gen_ai.operation.name")
        for span in memory_exporter.get_finished_spans()
    }
    assert "invoke_agent" in op_names
    assert "invoke_workflow" in op_names
    spans = memory_exporter.get_finished_spans()
    _assert_agent_and_workflow_attributes(spans)
    agent_names = {
        span.attributes.get("gen_ai.agent.name")
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    }
    assert {"ResearchAgent", "WriteAgent", "ReviewAgent"} <= agent_names
