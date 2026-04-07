"""Integration tests for LangGraph interrupt/resume with real graph execution.

Tests the full flow: LangGraph graph → instrumentation callbacks → spans,
verifying that interrupt/resume produces correct telemetry.
"""

from __future__ import annotations

import os
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

    # Enable content capture so gen_ai.input/output.messages are populated.
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE"] = "SPAN_ONLY"

    # Reset the singleton TelemetryHandler so a fresh one is created
    # with our test TracerProvider.
    _handler_mod.TelemetryHandler._reset_for_testing()

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
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", None)
    _handler_mod.TelemetryHandler._reset_for_testing()


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="langgraph or otel sdk not available")
class TestLangGraphInterruptResume:
    """End-to-end tests for interrupt/resume with real LangGraph execution."""

    def test_interrupt_resume_produces_two_traces(self, instrumented_graph):
        """Interrupt + resume should produce two separate traces,
        each with an agent root span (default)."""
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

        # Root spans should be invoke_agent (default, not invoke_workflow)
        for label, spans in [("Phase 1", phase1_spans), ("Phase 2", phase2_spans)]:
            roots = [s for s in spans if s.parent is None]
            assert len(roots) == 1, f"{label} should have one root span"
            op = roots[0].attributes.get("gen_ai.operation.name")
            assert op == "invoke_agent", (
                f"{label} root should be invoke_agent, got {op}"
            )

        # Traces should be different
        trace1_ids = {s.context.trace_id for s in phase1_spans}
        trace2_ids = {s.context.trace_id for s in phase2_spans}
        assert len(trace1_ids) == 1, "Phase 1 spans should share one trace"
        assert len(trace2_ids) == 1, "Phase 2 spans should share one trace"
        assert trace1_ids != trace2_ids, "Phases should have different trace IDs"

    def test_resume_root_has_command_attribute(self, instrumented_graph):
        """The resume root span should have
        gen_ai.command='resume'."""
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

        # Find the root span (agent by default)
        root_spans = [s for s in exporter.spans if s.parent is None]
        assert len(root_spans) == 1, "Should have exactly one root span"

        root = root_spans[0]
        assert root.attributes.get("gen_ai.operation.name") == "invoke_agent", (
            "Resume root should be invoke_agent by default"
        )
        assert root.attributes.get("gen_ai.command") == "resume"

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

    def test_interrupt_bubbles_finish_reason_to_root(self, instrumented_graph):
        """When a child step is interrupted, the root invoke_agent span
        should receive gen_ai.finish_reason='interrupted'.  The interrupt
        value is captured via gen_ai.output.messages (not duplicated in
        finish_reason_description on the root)."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-bubble-finish"}}

        # Phase 1: Run until interrupt at human_review node
        for _ in graph.stream({"value": "start"}, config):
            pass
        tp.force_flush()

        # Find the root span
        root_spans = [s for s in exporter.spans if s.parent is None]
        assert len(root_spans) == 1
        root = root_spans[0]

        # Root span should have finish_reason=interrupted (bubbled up from child)
        assert root.attributes.get("gen_ai.finish_reason") == "interrupted", (
            "Root span should have gen_ai.finish_reason='interrupted' "
            "bubbled up from the interrupted child step"
        )

        # finish_reason_description should NOT be on the root span — the
        # interrupt payload is already in gen_ai.output.messages.
        assert root.attributes.get("gen_ai.finish_reason_description") is None, (
            "Root span should NOT have finish_reason_description "
            "(interrupt payload is in gen_ai.output.messages instead)"
        )

        # The interrupted child step should have finish_reason=interrupted
        # with a clean finish_reason_description from the interrupt value.
        step_spans = [s for s in exporter.spans if "human_review" in s.name]
        assert len(step_spans) == 1
        step = step_spans[0]
        assert step.attributes.get("gen_ai.finish_reason") == "interrupted"
        step_desc = step.attributes.get("gen_ai.finish_reason_description")
        assert step_desc is not None and "approve?" in step_desc, (
            f"Child step should have clean finish_reason_description, got: {step_desc}"
        )

    def test_interrupt_sets_output_from_interrupt_value(self, instrumented_graph):
        """Root span output should contain the interrupt value (the question
        to the user), not the full graph state dump."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-int-output"}}

        for _ in graph.stream({"value": "start"}, config):
            pass
        tp.force_flush()

        root_spans = [s for s in exporter.spans if s.parent is None]
        assert len(root_spans) == 1
        root = root_spans[0]

        output = root.attributes.get("gen_ai.output.messages")
        assert output is not None, "Root span should have gen_ai.output.messages"

        import json

        parsed = json.loads(output)
        assert isinstance(parsed, list) and len(parsed) == 1
        msg = parsed[0]
        assert msg.get("role") == "assistant"

        # The content is the serialised interrupt value dict inside a text part
        parts = msg.get("parts", [])
        assert len(parts) == 1
        assert parts[0].get("type") == "text"
        content = parts[0].get("content", "")
        content_data = json.loads(content)
        assert content_data.get("question") == "approve?", (
            f"Expected interrupt question in output, got: {content}"
        )

    def test_resume_root_captures_command_input(self, instrumented_graph):
        """The resume root span should capture the Command(resume=...)
        value as its gen_ai.input.messages."""
        graph, exporter, tp = instrumented_graph
        config = {"configurable": {"thread_id": "t-resume-input"}}

        # Phase 1: Interrupt
        for _ in graph.stream({"value": "start"}, config):
            pass
        tp.force_flush()
        exporter.spans.clear()

        # Phase 2: Resume with a meaningful answer
        for _ in graph.stream(Command(resume="approved!"), config):
            pass
        tp.force_flush()

        root_spans = [s for s in exporter.spans if s.parent is None]
        assert len(root_spans) == 1
        root = root_spans[0]

        input_attr = root.attributes.get("gen_ai.input.messages")
        assert input_attr is not None, (
            "Resume root span should have gen_ai.input.messages "
            "with the Command(resume=...) value"
        )

        import json

        parsed = json.loads(input_attr)
        assert isinstance(parsed, list) and len(parsed) == 1
        msg = parsed[0]
        assert msg.get("role") == "user"
        # Content is inside a text part
        parts = msg.get("parts", [])
        assert len(parts) >= 1
        content = parts[0].get("content", "")
        assert "approved!" in content, (
            f"Resume input should contain the resume value, got: {parts}"
        )
