"""Tests for LangChain error classification helpers."""

from opentelemetry.util.genai.types import ErrorClassification
from opentelemetry.instrumentation.langchain.callback_handler import (
    _classify_error,
)


# --- _classify_error tests ---


def test_classify_graph_interrupt():
    class GraphInterrupt(Exception):
        pass

    assert _classify_error(GraphInterrupt("paused")) == ErrorClassification.INTERRUPT


def test_classify_node_interrupt():
    class NodeInterrupt(Exception):
        pass

    assert _classify_error(NodeInterrupt("paused")) == ErrorClassification.INTERRUPT


def test_classify_cancelled_error():
    import asyncio

    assert _classify_error(asyncio.CancelledError()) == ErrorClassification.CANCELLATION


def test_classify_real_error():
    assert _classify_error(RuntimeError("boom")) == ErrorClassification.REAL_ERROR


def test_classify_subclass_of_interrupt():
    class GraphInterrupt(Exception):
        pass

    class CustomInterrupt(GraphInterrupt):
        pass

    assert _classify_error(CustomInterrupt("custom")) == ErrorClassification.INTERRUPT
