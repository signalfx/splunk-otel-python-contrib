"""Integration tests for manual handler instrumentation inside LangGraph nodes.

Verifies that when user code calls handler.start_tool_call() or
handler.start_step() from inside a LangGraph async node body, the resulting
spans are children of the auto-instrumented agent span (same trace), NOT
orphaned on a separate trace.

This is the core use-case: auto-instrumented LangChain/LangGraph creates
agent spans automatically, and user code adds manual tool/step spans
inside those agent nodes.

Root cause: LangChain Core's callback manager dispatches sync handlers via
copy_context().run() in a thread pool (langchain_core/callbacks/manager.py,
_ahandle_event_for_handler line ~385).  This isolates any ContextVar
modifications made in on_chain_start from the caller's context.  The fix is
setting ``run_inline = True`` on LangchainCallbackHandler so callbacks run
directly in the caller's context without copy_context() isolation.
"""

from __future__ import annotations

import asyncio
import os
from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.util.genai.handler import get_telemetry_handler, TelemetryHandler
from opentelemetry.util.genai.types import ToolCall, Step


class _CollectingExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class State(TypedDict):
    value: str


def _fully_unwrap_callback_manager():
    """Unwrap BaseCallbackManager.__init__ to restore original for test isolation."""
    try:
        from wrapt import ObjectProxy
        import langchain_core.callbacks

        func = langchain_core.callbacks.BaseCallbackManager.__init__
        while isinstance(func, ObjectProxy) and hasattr(func, "__wrapped__"):
            func = func.__wrapped__
        langchain_core.callbacks.BaseCallbackManager.__init__ = func
    except Exception:
        pass


# -- Async node functions that use manual handler instrumentation --


async def node_with_tool_call(state: State) -> State:
    """An async LangGraph node that manually creates a ToolCall span."""
    handler = get_telemetry_handler()
    tool = ToolCall(
        name="my_tool",
        tool_type="function",
        tool_description="a test tool",
        arguments={"input": state["value"]},
        agent_name="test-agent",
    )
    handler.start_tool_call(tool)
    await asyncio.sleep(0.001)
    tool.tool_result = "tool result"
    handler.stop_tool_call(tool)
    return {"value": state["value"] + " -> tool_done"}


async def node_with_step(state: State) -> State:
    """An async LangGraph node that manually creates a Step span."""
    handler = get_telemetry_handler()
    step = Step(
        name="my_step",
        objective="test step inside agent",
        step_type="processing",
        source="agent",
        assigned_agent="test-agent",
    )
    handler.start_step(step)
    await asyncio.sleep(0.001)
    handler.stop_step(step)
    return {"value": state["value"] + " -> step_done"}


# -- Fixtures --


def _setup_env():
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE"] = "SPAN_ONLY"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"


def _teardown_env():
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", None)
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)


@pytest.fixture()
def instrumented_tool_graph():
    """Graph with one async node that creates a manual ToolCall."""
    _setup_env()
    TelemetryHandler._reset_for_testing()

    exporter = _CollectingExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = LangchainInstrumentor()
    instrumentor.instrument(tracer_provider=tp)

    builder = StateGraph(State)
    builder.add_node(
        "agent_node", node_with_tool_call, metadata={"agent_name": "test-agent"}
    )
    builder.add_edge("__start__", "agent_node")
    builder.add_edge("agent_node", "__end__")
    graph = builder.compile()

    yield graph, exporter, tp

    instrumentor.uninstrument()
    _fully_unwrap_callback_manager()
    tp.shutdown()
    _teardown_env()
    TelemetryHandler._reset_for_testing()


@pytest.fixture()
def instrumented_step_graph():
    """Graph with one async node that creates a manual Step."""
    _setup_env()
    TelemetryHandler._reset_for_testing()

    exporter = _CollectingExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = LangchainInstrumentor()
    instrumentor.instrument(tracer_provider=tp)

    builder = StateGraph(State)
    builder.add_node("step_node", node_with_step, metadata={"agent_name": "test-agent"})
    builder.add_edge("__start__", "step_node")
    builder.add_edge("step_node", "__end__")
    graph = builder.compile()

    yield graph, exporter, tp

    instrumentor.uninstrument()
    _fully_unwrap_callback_manager()
    tp.shutdown()
    _teardown_env()
    TelemetryHandler._reset_for_testing()


@pytest.fixture()
def instrumented_multi_node_graph():
    """Graph with two sequential async nodes, each with manual tool calls."""
    _setup_env()
    TelemetryHandler._reset_for_testing()

    exporter = _CollectingExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = LangchainInstrumentor()
    instrumentor.instrument(tracer_provider=tp)

    async def node_a(state: State) -> State:
        handler = get_telemetry_handler()
        tool = ToolCall(
            name="tool_a",
            tool_type="function",
            arguments={"input": "a"},
            agent_name="agent-a",
        )
        handler.start_tool_call(tool)
        await asyncio.sleep(0.001)
        tool.tool_result = "result_a"
        handler.stop_tool_call(tool)
        return {"value": state["value"] + " -> a"}

    async def node_b(state: State) -> State:
        handler = get_telemetry_handler()
        tool = ToolCall(
            name="tool_b",
            tool_type="function",
            arguments={"input": "b"},
            agent_name="agent-b",
        )
        handler.start_tool_call(tool)
        await asyncio.sleep(0.001)
        tool.tool_result = "result_b"
        handler.stop_tool_call(tool)
        return {"value": state["value"] + " -> b"}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a, metadata={"agent_name": "agent-a"})
    builder.add_node("node_b", node_b, metadata={"agent_name": "agent-b"})
    builder.add_edge("__start__", "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "__end__")
    graph = builder.compile()

    yield graph, exporter, tp

    instrumentor.uninstrument()
    _fully_unwrap_callback_manager()
    tp.shutdown()
    _teardown_env()
    TelemetryHandler._reset_for_testing()


# -- Tests --


class TestLangGraphManualInstrumentation:
    """Tests that manual handler.start_tool_call() / start_step() inside
    async LangGraph nodes produce spans on the same trace as the
    auto-instrumented agent spans."""

    @pytest.mark.asyncio
    async def test_tool_call_is_child_of_agent_span(self, instrumented_tool_graph):
        """A manual ToolCall inside an async LangGraph node should be a child
        of the auto-instrumented agent span, not on a separate trace."""
        graph, exporter, tp = instrumented_tool_graph

        await graph.ainvoke({"value": "start"})
        tp.force_flush()

        spans = exporter.spans
        assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"

        # All spans should share the same trace ID
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1, (
            f"All spans should share one trace, got {len(trace_ids)} traces. "
            f"Span details: "
            f"{[(s.name, format(s.context.trace_id, '032x')) for s in spans]}"
        )

        # Find the tool call span
        tool_spans = [s for s in spans if "my_tool" in s.name]
        assert len(tool_spans) == 1, (
            f"Expected 1 tool span, got {len(tool_spans)}. "
            f"All span names: {[s.name for s in spans]}"
        )
        assert tool_spans[0].parent is not None, (
            "Tool call span should have a parent (the agent span), "
            "but parent is None — span is orphaned on its own trace"
        )

    @pytest.mark.asyncio
    async def test_step_is_child_of_agent_span(self, instrumented_step_graph):
        """A manual Step inside an async LangGraph node should be a child
        of the auto-instrumented agent span."""
        graph, exporter, tp = instrumented_step_graph

        await graph.ainvoke({"value": "start"})
        tp.force_flush()

        spans = exporter.spans
        assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"

        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1, (
            f"All spans should share one trace, got {len(trace_ids)} traces"
        )

        step_spans = [s for s in spans if "my_step" in s.name]
        assert len(step_spans) == 1, (
            f"Expected 1 step span, got {len(step_spans)}. "
            f"All span names: {[s.name for s in spans]}"
        )
        assert step_spans[0].parent is not None, (
            "Step span should have a parent (the agent span)"
        )

    @pytest.mark.asyncio
    async def test_multi_node_tool_calls_on_same_trace(
        self, instrumented_multi_node_graph
    ):
        """Tool calls in sequential async LangGraph nodes should all be on
        the same trace, each parented to their respective agent node span."""
        graph, exporter, tp = instrumented_multi_node_graph

        await graph.ainvoke({"value": "start"})
        tp.force_flush()

        spans = exporter.spans

        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1, (
            f"All spans should share one trace, got {len(trace_ids)} traces"
        )

        tool_a = [s for s in spans if "tool_a" in s.name]
        tool_b = [s for s in spans if "tool_b" in s.name]
        assert len(tool_a) == 1, f"Expected 1 tool_a span, got {len(tool_a)}"
        assert len(tool_b) == 1, f"Expected 1 tool_b span, got {len(tool_b)}"

        assert tool_a[0].parent is not None, "tool_a should have a parent"
        assert tool_b[0].parent is not None, "tool_b should have a parent"

        # Tool spans should be parented to different agent nodes
        assert tool_a[0].parent.span_id != tool_b[0].parent.span_id, (
            "tool_a and tool_b should be parented to different agent nodes"
        )

    @pytest.mark.asyncio
    async def test_no_orphan_spans(self, instrumented_tool_graph):
        """No spans should be orphaned (parent outside their trace)."""
        graph, exporter, tp = instrumented_tool_graph

        await graph.ainvoke({"value": "start"})
        tp.force_flush()

        traces: dict[int, list] = {}
        for s in exporter.spans:
            traces.setdefault(s.context.trace_id, []).append(s)

        for trace_id, spans in traces.items():
            span_ids = {s.context.span_id for s in spans}
            for s in spans:
                if s.parent is not None:
                    assert s.parent.span_id in span_ids, (
                        f"Span '{s.name}' has parent not in same trace "
                        f"(orphan span detected)"
                    )

    def test_callback_handler_runs_inline(self):
        """LangchainCallbackHandler must have run_inline=True.

        Without run_inline, LangChain Core's callback manager dispatches
        sync handlers via copy_context().run() in a thread pool, which
        isolates ContextVar modifications from the caller's context.
        This breaks _current_genai_span propagation to LangGraph node
        bodies where user code calls handler.start_tool_call() manually.
        """
        from opentelemetry.instrumentation.langchain.callback_handler import (
            LangchainCallbackHandler,
        )

        handler = LangchainCallbackHandler()
        assert handler.run_inline is True, (
            "LangchainCallbackHandler.run_inline must be True to ensure "
            "ContextVar propagation through LangGraph node boundaries"
        )
