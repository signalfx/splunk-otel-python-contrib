# Copyright Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Memory instrumentation."""

import pytest

from opentelemetry.instrumentation.bedrock_agentcore.memory_wrappers import (
    wrap_memory_create_blob_event,
    wrap_memory_create_event,
    wrap_memory_list_events,
    wrap_memory_operation,
    wrap_memory_retrieve,
)


class MockMemoryClient:
    """Mock MemoryClient for testing."""

    def __init__(self):
        self.region_name = "us-west-2"

    def retrieve_memories(self, memory_id, namespace, query, actor_id=None, top_k=3):
        return [
            {"id": "rec-1", "content": "test"},
            {"id": "rec-2", "content": "test2"},
        ]

    def create_event(self, memory_id, actor_id, session_id, payload=None):
        return {"eventId": "event-123"}

    def create_blob_event(self, memory_id, actor_id, session_id, blob=None):
        return {"eventId": "blob-456"}

    def list_events(self, memory_id, actor_id=None):
        return [{"eventId": "event-1"}, {"eventId": "event-2"}]

    def create_memory(self, **kwargs):
        return {"memoryId": "mem-new"}


# ---------------------------------------------------------------------------
# wrap_memory_retrieve
# ---------------------------------------------------------------------------


def test_memory_retrieve_creates_retrieval_invocation(stub_handler):
    """wrap_memory_retrieve creates a RetrievalInvocation span with content."""
    client = MockMemoryClient()

    wrap_memory_retrieve(
        client.retrieve_memories,
        client,
        ("mem-123", "ns/", "test query"),
        {},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_retrievals) == 1
    assert len(stub_handler.stopped_retrievals) == 1

    invocation = stub_handler.started_retrievals[0]
    assert invocation.query == "test query"
    assert invocation.retriever_type == "bedrock-agentcore-memory"
    assert invocation.documents_retrieved == 2


def test_memory_retrieve_top_k_from_args(stub_handler):
    """wrap_memory_retrieve captures top_k from positional args when content enabled."""
    client = MockMemoryClient()

    wrap_memory_retrieve(
        client.retrieve_memories,
        client,
        ("mem-123", "ns/", "query", None, 5),
        {},
        stub_handler,
        capture_content=True,
    )

    invocation = stub_handler.started_retrievals[0]
    assert invocation.top_k == 5


def test_memory_retrieve_top_k_from_kwargs(stub_handler):
    """wrap_memory_retrieve prefers kwargs over positional args for top_k."""

    def mock_retrieve(*args, **kwargs):
        return []

    wrap_memory_retrieve(
        mock_retrieve,
        None,
        ("mem-123", "ns/", "query", None, 3),
        {"top_k": 7},
        stub_handler,
        capture_content=True,
    )

    invocation = stub_handler.started_retrievals[0]
    assert invocation.top_k == 7


def test_memory_retrieve_no_content_by_default(stub_handler):
    """wrap_memory_retrieve suppresses query and document count when capture_content=False."""
    client = MockMemoryClient()

    wrap_memory_retrieve(
        client.retrieve_memories,
        client,
        ("mem-123", "ns/", "test query"),
        {},
        stub_handler,
    )

    assert len(stub_handler.started_retrievals) == 1
    invocation = stub_handler.started_retrievals[0]
    assert invocation.query == ""
    assert invocation.top_k is None
    assert invocation.documents_retrieved is None


def test_memory_retrieve_exception_fails_invocation(stub_handler):
    """wrap_memory_retrieve fails the invocation on exception."""

    def failing_retrieve(*args, **kwargs):
        raise ValueError("Memory retrieval failed")

    client = MockMemoryClient()

    with pytest.raises(ValueError, match="Memory retrieval failed"):
        wrap_memory_retrieve(
            failing_retrieve,
            client,
            ("mem-123", "ns/", "test"),
            {},
            stub_handler,
        )

    assert len(stub_handler.started_retrievals) == 1
    assert len(stub_handler.failed_entities) == 1
    _invocation, error = stub_handler.failed_entities[0]
    assert error.type == "ValueError"
    assert "Memory retrieval failed" in error.message


# ---------------------------------------------------------------------------
# wrap_memory_create_event
# ---------------------------------------------------------------------------


def test_memory_create_event_creates_tool_call(stub_handler):
    """wrap_memory_create_event creates a ToolCall span with content."""
    client = MockMemoryClient()

    wrap_memory_create_event(
        client.create_event,
        client,
        (),
        {"memory_id": "mem-123", "actor_id": "actor-1", "session_id": "sess-1"},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.create_event"
    assert tool_call.system == "bedrock-agentcore"
    assert "mem-123" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_memory_create_event_no_content_by_default(stub_handler):
    """wrap_memory_create_event suppresses arguments and result when capture_content=False."""
    client = MockMemoryClient()

    wrap_memory_create_event(
        client.create_event,
        client,
        (),
        {"memory_id": "mem-123", "actor_id": "actor-1", "session_id": "sess-1"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_memory_create_event_positional_args(stub_handler):
    """wrap_memory_create_event extracts ids from positional args."""
    client = MockMemoryClient()

    wrap_memory_create_event(
        client.create_event,
        client,
        ("mem-pos", "actor-pos", "sess-pos"),
        {},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-pos" in tool_call.arguments
    assert "actor-pos" in tool_call.arguments


def test_memory_create_event_kwargs_preferred_over_args(stub_handler):
    """wrap_memory_create_event prefers kwargs when both are provided."""

    def mock_create(*args, **kwargs):
        return {"eventId": "x"}

    wrap_memory_create_event(
        mock_create,
        None,
        ("mem-pos", "actor-pos", "sess-pos"),
        {"memory_id": "mem-kw"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-kw" in tool_call.arguments


def test_memory_create_event_exception_fails_tool_call(stub_handler):
    """wrap_memory_create_event fails the tool call on exception."""

    def failing_create(*args, **kwargs):
        raise ConnectionError("Service unavailable")

    client = MockMemoryClient()

    with pytest.raises(ConnectionError, match="Service unavailable"):
        wrap_memory_create_event(
            failing_create,
            client,
            (),
            {"memory_id": "mem-123", "actor_id": "actor-1", "session_id": "sess-1"},
            stub_handler,
        )

    assert len(stub_handler.failed_entities) == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"


# ---------------------------------------------------------------------------
# wrap_memory_create_blob_event
# ---------------------------------------------------------------------------


def test_memory_create_blob_event_creates_tool_call(stub_handler):
    """wrap_memory_create_blob_event creates a ToolCall span with content."""
    client = MockMemoryClient()

    wrap_memory_create_blob_event(
        client.create_blob_event,
        client,
        ("mem-123", "actor-1", "sess-1"),
        {},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.create_blob_event"
    assert tool_call.system == "bedrock-agentcore"
    assert "mem-123" in tool_call.arguments


def test_memory_create_blob_event_no_content_by_default(stub_handler):
    """wrap_memory_create_blob_event suppresses arguments when capture_content=False."""
    client = MockMemoryClient()

    wrap_memory_create_blob_event(
        client.create_blob_event,
        client,
        ("mem-123", "actor-1", "sess-1"),
        {},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_memory_create_blob_event_kwargs_preferred_over_args(stub_handler):
    """wrap_memory_create_blob_event prefers kwargs when both provided."""

    def mock_create(*args, **kwargs):
        return {"eventId": "x"}

    wrap_memory_create_blob_event(
        mock_create,
        None,
        ("mem-pos", "actor-pos", "sess-pos"),
        {"memory_id": "mem-kw"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-kw" in tool_call.arguments


# ---------------------------------------------------------------------------
# wrap_memory_list_events
# ---------------------------------------------------------------------------


def test_memory_list_events_creates_tool_call(stub_handler):
    """wrap_memory_list_events creates a ToolCall span with content."""
    client = MockMemoryClient()

    wrap_memory_list_events(
        client.list_events,
        client,
        (),
        {"memory_id": "mem-123"},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.list_events"
    assert tool_call.system == "bedrock-agentcore"
    assert "mem-123" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_memory_list_events_no_content_by_default(stub_handler):
    """wrap_memory_list_events suppresses arguments and result when capture_content=False."""
    client = MockMemoryClient()

    wrap_memory_list_events(
        client.list_events,
        client,
        (),
        {"memory_id": "mem-123"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


# ---------------------------------------------------------------------------
# wrap_memory_operation (generic factory)
# ---------------------------------------------------------------------------


def test_memory_operation_creates_tool_call(stub_handler):
    """wrap_memory_operation factory creates a ToolCall span."""
    client = MockMemoryClient()
    wrapper = wrap_memory_operation("create_memory")

    wrapper(
        client.create_memory,
        client,
        (),
        {"memory_name": "my-memory"},
        stub_handler,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1
    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.create_memory"
    assert tool_call.system == "bedrock-agentcore"


def test_memory_operation_with_content(stub_handler):
    """wrap_memory_operation captures kwargs as arguments when content enabled."""
    client = MockMemoryClient()
    wrapper = wrap_memory_operation("create_memory")

    wrapper(
        client.create_memory,
        client,
        (),
        {"memory_name": "my-memory"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "my-memory" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_memory_operation_no_content_by_default(stub_handler):
    """wrap_memory_operation suppresses arguments and result when capture_content=False."""
    client = MockMemoryClient()
    wrapper = wrap_memory_operation("create_memory")

    wrapper(
        client.create_memory,
        client,
        (),
        {"memory_name": "my-memory"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_memory_operation_exception_fails_tool_call(stub_handler):
    """wrap_memory_operation fails the tool call on exception."""

    def failing_op(*args, **kwargs):
        raise RuntimeError("Memory operation failed")

    wrapper = wrap_memory_operation("delete_memory")

    with pytest.raises(RuntimeError, match="Memory operation failed"):
        wrapper(failing_op, None, (), {}, stub_handler)

    assert len(stub_handler.failed_entities) == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"
