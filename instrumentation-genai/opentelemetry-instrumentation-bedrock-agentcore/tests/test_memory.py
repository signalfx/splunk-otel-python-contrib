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


def test_memory_retrieve_creates_retrieval_invocation(stub_handler):
    """wrap_memory_retrieve should create a RetrievalInvocation via start_retrieval."""
    client = MockMemoryClient()

    wrap_memory_retrieve(
        client.retrieve_memories,
        client,
        ("mem-123", "ns/", "test query"),
        {},
        stub_handler,
    )

    assert len(stub_handler.started_retrievals) == 1
    assert len(stub_handler.stopped_retrievals) == 1

    invocation = stub_handler.started_retrievals[0]
    assert invocation.query == "test query"
    assert invocation.retriever_type == "bedrock-agentcore-memory"
    assert invocation.documents_retrieved == 2


def test_memory_retrieve_top_k_from_args(stub_handler):
    """wrap_memory_retrieve should capture top_k from positional args."""
    client = MockMemoryClient()

    wrap_memory_retrieve(
        client.retrieve_memories,
        client,
        ("mem-123", "ns/", "query", None, 5),
        {},
        stub_handler,
    )

    invocation = stub_handler.started_retrievals[0]
    assert invocation.top_k == 5


def test_memory_retrieve_exception_fails_invocation(stub_handler):
    """wrap_memory_retrieve should fail the invocation on exception."""

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


def test_memory_create_event_creates_tool_call(stub_handler):
    """wrap_memory_create_event should create a ToolCall span."""
    client = MockMemoryClient()

    wrap_memory_create_event(
        client.create_event,
        client,
        (),
        {"memory_id": "mem-123", "actor_id": "actor-1", "session_id": "sess-1"},
        stub_handler,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.create_event"
    assert tool_call.system == "bedrock-agentcore"
    assert "mem-123" in tool_call.arguments


def test_memory_create_event_exception_fails_tool_call(stub_handler):
    """wrap_memory_create_event should fail the tool call on exception."""

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


def test_memory_create_blob_event_creates_tool_call(stub_handler):
    """wrap_memory_create_blob_event should create a ToolCall span."""
    client = MockMemoryClient()

    wrap_memory_create_blob_event(
        client.create_blob_event,
        client,
        ("mem-123", "actor-1", "sess-1"),
        {},
        stub_handler,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.create_blob_event"
    assert tool_call.system == "bedrock-agentcore"
    assert "mem-123" in tool_call.arguments


def test_memory_list_events_creates_tool_call(stub_handler):
    """wrap_memory_list_events should create a ToolCall span."""
    client = MockMemoryClient()

    wrap_memory_list_events(
        client.list_events,
        client,
        (),
        {"memory_id": "mem-123"},
        stub_handler,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.list_events"
    assert tool_call.system == "bedrock-agentcore"
    assert "mem-123" in tool_call.arguments
