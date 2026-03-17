"""Integration tests for LangGraph interrupt/resume with real graph execution.

Tests the full flow: LangGraph graph → instrumentation callbacks → spans,
verifying that interrupt/resume produces correct telemetry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import pytest

_PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if _PACKAGE_SRC.exists():
    sys.path.insert(0, str(_PACKAGE_SRC))

try:
    from langgraph.graph import StateGraph
    from langgraph.types import Command, interrupt
    from langgraph.checkpoint.memory import MemorySaver
except ModuleNotFoundError:
    StateGraph = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
except ModuleNotFoundError:
    TracerProvider = None  # type: ignore[assignment]

from opentelemetry.instrumentation.langchain import LangchainInstrumentor  # noqa: E402

LANGGRAPH_AVAILABLE = StateGraph is not None
OTEL_SDK_AVAILABLE = TracerProvider is not None
DEPS_AVAILABLE = LANGGRAPH_AVAILABLE and OTEL_SDK_AVAILABLE


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


def _step_a(state: State) -> State:
    return {"value": state["value"] + " -> step_a"}


def _human_review(state: State) -> State:
    answer = interrupt({"question": "approve?"})
    return {"value": state["value"] + f" -> human_review({answer})"}


def _step_b(state: State) -> State:
    return {"value": state["value"] + " -> step_b"}


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


@pytest.fixture()
def instrumented_graph():
    """Per-test fixture: fresh TracerProvider, instrumentor, graph, and exporter."""
    import opentelemetry.util.genai.handler as _handler_mod

    # Reset the singleton TelemetryHandler so a fresh one is created
    # with our test TracerProvider.
    if hasattr(_handler_mod.get_telemetry_handler, "_default_handler"):
        setattr(_handler_mod.get_telemetry_handler, "_default_handler", None)

    exporter = _CollectingExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = LangchainInstrumentor()
    instrumentor.instrument(tracer_provider=tp)

    builder = StateGraph(State)
    builder.add_node("step_a", _step_a)
    builder.add_node("human_review", _human_review)
    builder.add_node("step_b", _step_b)
    builder.add_edge("__start__", "step_a")
    builder.add_edge("step_a", "human_review")
    builder.add_edge("human_review", "step_b")
    builder.add_edge("step_b", "__end__")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    yield graph, exporter, tp

    instrumentor.uninstrument()
    _fully_unwrap_callback_manager()
    tp.shutdown()
    setattr(_handler_mod.get_telemetry_handler, "_default_handler", None)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="langgraph or otel sdk not available")
class TestLangGraphInterruptResume:
    """End-to-end tests for interrupt/resume with real LangGraph execution."""

    def test_interrupt_resume_produces_two_traces(self, instrumented_graph):
        """Interrupt + resume should produce two separate traces,
        each with a workflow root span."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-two-traces"}}

        # Phase 1: Run until interrupt
        for _ in graph.stream({"value": "start"}, config):
            pass

        tp.force_flush()
        phase1_spans = list(exporter.spans)
        exporter.spans.clear()

        # Phase 2: Resume
        for _ in graph.stream(Command(resume="approved!"), config):
            pass

        tp.force_flush()
        phase2_spans = list(exporter.spans)

        # Each phase should have spans
        assert len(phase1_spans) > 0, "Phase 1 should produce spans"
        assert len(phase2_spans) > 0, "Phase 2 should produce spans"

        # Traces should be different
        trace1_ids = {s.context.trace_id for s in phase1_spans}
        trace2_ids = {s.context.trace_id for s in phase2_spans}
        assert len(trace1_ids) == 1, "Phase 1 spans should share one trace"
        assert len(trace2_ids) == 1, "Phase 2 spans should share one trace"
        assert trace1_ids != trace2_ids, "Phases should have different trace IDs"

    def test_resume_workflow_has_command_attribute(self, instrumented_graph):
        """The resume workflow span should have
        gen_ai.workflow.command='resume'."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-cmd-attr"}}

        # Phase 1: Interrupt
        for _ in graph.stream({"value": "start"}, config):
            pass
        tp.force_flush()
        exporter.spans.clear()

        # Phase 2: Resume
        for _ in graph.stream(Command(resume="approved!"), config):
            pass
        tp.force_flush()

        # Find the root workflow span
        root_spans = [s for s in exporter.spans if s.parent is None]
        assert len(root_spans) == 1, "Should have exactly one root span"

        root = root_spans[0]
        assert root.name.startswith("workflow"), (
            f"Root should be workflow, got {root.name}"
        )
        assert root.attributes.get("gen_ai.workflow.command") == "resume"

    def test_resume_has_conversation_id(self, instrumented_graph):
        """All spans in both phases should have gen_ai.conversation.id
        from thread_id."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-conv-id"}}

        for _ in graph.stream({"value": "start"}, config):
            pass
        for _ in graph.stream(Command(resume="approved!"), config):
            pass
        tp.force_flush()

        for span in exporter.spans:
            conv_id = span.attributes.get("gen_ai.conversation.id")
            assert conv_id == "t-conv-id", (
                f"Span {span.name} missing conversation.id, got {conv_id}"
            )

    def test_no_orphan_spans_on_resume(self, instrumented_graph):
        """All child spans should have a parent within the same trace."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-no-orphan"}}

        for _ in graph.stream({"value": "start"}, config):
            pass
        for _ in graph.stream(Command(resume="approved!"), config):
            pass
        tp.force_flush()

        # Group spans by trace
        traces: dict[int, list] = {}
        for s in exporter.spans:
            traces.setdefault(s.context.trace_id, []).append(s)

        for trace_id, spans in traces.items():
            span_ids = {s.context.span_id for s in spans}
            for s in spans:
                if s.parent is not None:
                    assert s.parent.span_id in span_ids, (
                        f"Span {s.name} has parent not in same trace "
                        f"(orphan span detected)"
                    )
