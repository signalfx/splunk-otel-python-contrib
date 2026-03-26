import asyncio
from collections import Counter

import pytest
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    CodeActAgent,
    FunctionAgent,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


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
            raw={
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                }
            },
        )

    async def achat(self, messages, **kwargs):
        return ChatResponse(
            message=self._response,
            raw={
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                }
            },
        )

    def stream_chat(self, messages, **kwargs):
        yield ChatResponse(
            message=self._response,
            delta=self._response.content,
            raw={
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                }
            },
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


def _span_tree_text(spans):
    by_id = {span.context.span_id: span for span in spans}
    children = {}
    roots = []
    for span in spans:
        parent_id = span.parent.span_id if span.parent else None
        if parent_id in by_id:
            children.setdefault(parent_id, []).append(span)
        else:
            roots.append(span)

    def _line(span, depth):
        op = span.attributes.get("gen_ai.operation.name") or span.name
        parent_id = span.parent.span_id if span.parent else None
        return (
            f"{'  ' * depth}{op}({span.name}) "
            f"span_id={span.context.span_id:x} parent_id={parent_id}"
        )

    def _walk(span, depth, out):
        out.append(_line(span, depth))
        for child in sorted(
            children.get(span.context.span_id, []),
            key=lambda s: s.start_time,
        ):
            _walk(child, depth + 1, out)

    lines = []
    for root in sorted(roots, key=lambda s: s.start_time):
        _walk(root, 0, lines)
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_function_agent_emits_spans(span_exporter, instrument):
    agent = FunctionAgent(
        name="FunctionAgent",
        llm=FunctionCallingLLM("Done."),
        tools=[],
        streaming=False,
    )

    handler = agent.run(user_msg="Say hello.")
    await handler
    await asyncio.sleep(0.1)

    spans = span_exporter.get_finished_spans()
    _assert_agent_and_workflow_spans(spans)
    _assert_agent_and_workflow_attributes(spans)


@pytest.mark.asyncio
async def test_codeact_agent_emits_spans(span_exporter, instrument):
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

    spans = span_exporter.get_finished_spans()
    _assert_agent_and_workflow_spans(spans)
    _assert_agent_and_workflow_attributes(spans)


@pytest.mark.asyncio
async def test_agent_workflow_emits_agent_span(span_exporter, instrument):
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
        for span in span_exporter.get_finished_spans()
    }
    assert "invoke_agent" in op_names
    assert "invoke_workflow" in op_names
    _assert_agent_and_workflow_attributes(span_exporter.get_finished_spans())


@pytest.mark.asyncio
async def test_multi_agent_workflow_example_emits_spans(span_exporter, instrument):
    llm = HandoffSequenceLLM(
        "Done.",
        [
            (
                "handoff",
                {
                    "to_agent": "WriteAgent",
                    "reason": "Draft ready.",
                },
            ),
            (
                "handoff",
                {
                    "to_agent": "ReviewAgent",
                    "reason": "Needs review.",
                },
            ),
            None,
        ],
    )

    research_agent = FunctionAgent(
        name="ResearchAgent",
        description="Search the web and record notes.",
        system_prompt="You are a researcher.",
        llm=llm,
        tools=[],
        can_handoff_to=["WriteAgent"],
        streaming=False,
    )
    write_agent = FunctionAgent(
        name="WriteAgent",
        description="Writes a markdown report.",
        system_prompt="You are a writer.",
        llm=llm,
        tools=[],
        can_handoff_to=["ReviewAgent", "ResearchAgent"],
        streaming=False,
    )
    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Reviews a report.",
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

    spans = span_exporter.get_finished_spans()
    op_names = {span.attributes.get("gen_ai.operation.name") for span in spans}
    assert "invoke_agent" in op_names
    assert "invoke_workflow" in op_names
    _assert_agent_and_workflow_attributes(spans)

    agent_names = {
        span.attributes.get("gen_ai.agent.name")
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    }
    assert {"ResearchAgent", "WriteAgent", "ReviewAgent"} <= agent_names

    workflow_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_workflow"
    ]
    assert len(workflow_spans) == 1
    workflow_span_id = workflow_spans[0].context.span_id

    concrete_agent_counts = Counter(
        span.attributes.get("gen_ai.agent.name")
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
        and span.attributes.get("gen_ai.agent.name")
        in {"ResearchAgent", "WriteAgent", "ReviewAgent"}
    )
    assert concrete_agent_counts == {
        "ResearchAgent": 1,
        "WriteAgent": 1,
        "ReviewAgent": 1,
    }

    for span in spans:
        if span.attributes.get(
            "gen_ai.operation.name"
        ) == "invoke_agent" and span.attributes.get("gen_ai.agent.name") in {
            "ResearchAgent",
            "WriteAgent",
            "ReviewAgent",
        }:
            assert span.parent is not None
            assert span.parent.span_id == workflow_span_id

    concrete_agent_span_ids = {
        span.context.span_id
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
        and span.attributes.get("gen_ai.agent.name")
        in {"ResearchAgent", "WriteAgent", "ReviewAgent"}
    }
    chat_spans = [
        span for span in spans if span.attributes.get("gen_ai.operation.name") == "chat"
    ]
    if chat_spans:
        prefixed_agent_name_chats = [
            span
            for span in chat_spans
            if str(span.attributes.get("gen_ai.agent.name", "")).startswith("agent.")
        ]
        assert not prefixed_agent_name_chats, (
            "Found chat spans with prefixed gen_ai.agent.name.\n"
            f"{_span_tree_text(spans)}"
        )
        wrong_parent_chat_spans = [
            span
            for span in chat_spans
            if (span.parent is None)
            or (span.parent.span_id not in concrete_agent_span_ids)
        ]
        assert not wrong_parent_chat_spans, (
            "Found chat spans not attached to concrete agent spans.\n"
            f"{_span_tree_text(spans)}"
        )


@pytest.mark.asyncio
async def test_custom_workflow_single_agent(span_exporter, instrument):
    agent = FunctionAgent(
        name="CustomSingleAgent",
        llm=FunctionCallingLLM("Done."),
        tools=[],
        streaming=False,
    )

    class CustomWorkflow(Workflow):
        @step
        async def run_agent(self, ev: StartEvent) -> StopEvent:
            handler = agent.run(user_msg="Help me plan a short trip.")
            result = await handler
            return StopEvent(result=str(result))

    workflow = CustomWorkflow(timeout=30, verbose=False)
    await workflow.run()
    await asyncio.sleep(0.1)

    spans = span_exporter.get_finished_spans()
    workflow_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_workflow"
    ]
    assert len(workflow_spans) == 1
    workflow_span_id = workflow_spans[0].context.span_id

    function_agent_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
        and span.attributes.get("gen_ai.agent.name") == "FunctionAgent"
    ]
    assert len(function_agent_spans) == 1
    assert function_agent_spans[0].parent is not None
    assert function_agent_spans[0].parent.span_id == workflow_span_id
