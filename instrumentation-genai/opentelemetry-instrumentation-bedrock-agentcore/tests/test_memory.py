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
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler, get_telemetry_handler

from opentelemetry.instrumentation.bedrock_agentcore.memory_wrappers import (
    wrap_memory_conversation_operation,
    wrap_memory_create_blob_event,
    wrap_memory_create_event,
    wrap_memory_list_events,
    wrap_memory_operation,
    wrap_memory_retrieve,
    wrap_memory_session_async_operation,
    wrap_memory_session_operation,
    wrap_memory_session_search_long_term_memories,
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
        return {"eventId": "event-123", "payload": payload}

    def create_blob_event(self, memory_id, actor_id, session_id, blob=None):
        return {"eventId": "blob-456", "blob": blob}

    def list_events(self, memory_id, actor_id=None, include_payload=True):
        return [
            {
                "eventId": "event-1",
                "payload": {"message": "sensitive"} if include_payload else None,
            },
            {
                "eventId": "event-2",
                "payload": {"message": "sensitive2"} if include_payload else None,
            },
        ]

    def create_memory(self, **kwargs):
        return {"memoryId": "mem-new"}

    def process_turn_with_llm(
        self,
        memory_id,
        actor_id,
        session_id,
        messages,
        llm_callback=None,
    ):
        return {"messages": messages, "summary": "sensitive summary"}

    def save_conversation(self, memory_id, actor_id, session_id, conversation):
        return {"conversation": conversation}

    def fork_conversation(self, memory_id, actor_id, session_id, branch_name, messages):
        return {"branchName": branch_name, "messages": messages}

    def get_last_k_turns(self, memory_id, actor_id, session_id, k=3):
        return {"turns": [{"role": "user", "content": "sensitive"}]}


class MockMemorySessionManager:
    """Mock MemorySessionManager for testing."""

    def __init__(self):
        self._memory_id = "mem-session"

    def create_memory_session(self, actor_id, session_id=None):
        return {
            "memoryId": self._memory_id,
            "actorId": actor_id,
            "sessionId": session_id or "generated-session",
        }

    def add_turns(self, actor_id, session_id, messages, metadata=None):
        return {"eventId": "event-1", "payload": messages, "metadata": metadata}

    def process_turn_with_llm(
        self,
        actor_id,
        session_id,
        user_input,
        llm_callback,
        retrieval_config=None,
        metadata=None,
    ):
        return (
            [{"content": {"text": "sensitive retrieved memory"}}],
            llm_callback(user_input, []),
            {"payload": [{"content": user_input}], "metadata": metadata},
        )

    async def process_turn_with_llm_async(
        self,
        actor_id,
        session_id,
        user_input,
        llm_callback,
        retrieval_config=None,
        metadata=None,
    ):
        return (
            [{"content": {"text": "sensitive retrieved memory"}}],
            await llm_callback(user_input, []),
            {"payload": [{"content": user_input}], "metadata": metadata},
        )

    def list_events(self, actor_id, session_id, include_payload=True, max_results=100):
        return [
            {
                "eventId": "event-1",
                "payload": {"message": "sensitive"} if include_payload else None,
            }
        ][:max_results]

    def search_long_term_memories(self, query, namespace_prefix, top_k=3):
        return [{"memoryRecordId": "record-1", "content": {"text": query}}]


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
    assert invocation.operation_name == "retrieval"
    assert invocation.provider == "bedrock-agentcore-memory"
    assert invocation.query == "test query"
    assert invocation.retriever_type == "bedrock-agentcore-memory"
    assert invocation.data_source_id == "memory.retrieve_memories"
    assert invocation.system == "bedrock-agentcore"
    assert invocation.top_k == 3
    assert invocation.documents_retrieved == 2


def test_memory_retrieve_span_name_includes_memory_operation(
    span_exporter,
    tracer_provider,
):
    """wrap_memory_retrieve exports a named retrieval span for AgentCore memory."""
    TelemetryHandler._reset_for_testing()
    handler = get_telemetry_handler(tracer_provider=tracer_provider)
    client = MockMemoryClient()

    try:
        wrap_memory_retrieve(
            client.retrieve_memories,
            client,
            ("mem-123", "ns/", "test query"),
            {},
            handler,
            capture_content=True,
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "retrieval bedrock-agentcore-memory"
        assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "retrieval"
        assert (
            span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
            == "bedrock-agentcore-memory"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_DATA_SOURCE_ID]
            == "memory.retrieve_memories"
        )
    finally:
        TelemetryHandler._reset_for_testing()


def test_memory_retrieve_top_k_from_args(stub_handler):
    """wrap_memory_retrieve captures top_k from positional args."""
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


def test_memory_retrieve_top_k_from_kwargs(stub_handler):
    """wrap_memory_retrieve prefers kwargs over positional args for top_k."""

    def mock_retrieve(memory_id, namespace, query, actor_id=None, top_k=3):
        return []

    wrap_memory_retrieve(
        mock_retrieve,
        None,
        ("mem-123", "ns/", "query", None),
        {"top_k": 7},
        stub_handler,
        capture_content=True,
    )

    invocation = stub_handler.started_retrievals[0]
    assert invocation.top_k == 7


def test_memory_retrieve_binds_kwargs_and_defaults(stub_handler):
    """wrap_memory_retrieve binds keyword args and applies SDK defaults."""
    client = MockMemoryClient()

    wrap_memory_retrieve(
        client.retrieve_memories,
        client,
        (),
        {
            "memory_id": "mem-123",
            "namespace": "ns/",
            "query": "keyword query",
        },
        stub_handler,
        capture_content=True,
    )

    invocation = stub_handler.started_retrievals[0]
    assert invocation.query == "keyword query"
    assert invocation.top_k == 3
    assert invocation.documents_retrieved == 2


def test_memory_retrieve_no_content_by_default_preserves_non_content_attributes(
    stub_handler,
):
    """wrap_memory_retrieve suppresses query only when capture_content=False."""
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
    assert invocation.top_k == 3
    assert invocation.documents_retrieved == 2


def test_memory_retrieve_exception_fails_invocation(stub_handler):
    """wrap_memory_retrieve fails the invocation on exception."""
    call_count = 0

    def failing_retrieve(*args, **kwargs):
        nonlocal call_count
        call_count += 1
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
    assert call_count == 1
    _invocation, error = stub_handler.failed_entities[0]
    assert error.type is ValueError
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
    assert tool_call.tool_result is None


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


def test_memory_create_event_suppresses_result_with_content(stub_handler):
    """wrap_memory_create_event never captures returned event payloads."""
    client = MockMemoryClient()

    wrap_memory_create_event(
        client.create_event,
        client,
        (),
        {
            "memory_id": "mem-123",
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "payload": {"message": "sensitive"},
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-123" in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
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
    call_count = 0

    def failing_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
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
    assert call_count == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type is ConnectionError


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


def test_memory_create_blob_event_suppresses_result_with_content(stub_handler):
    """wrap_memory_create_blob_event never captures returned blob content."""
    client = MockMemoryClient()

    wrap_memory_create_blob_event(
        client.create_blob_event,
        client,
        ("mem-123", "actor-1", "sess-1"),
        {"blob": b"sensitive blob"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-123" in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
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
    assert tool_call.tool_result is None


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


def test_memory_list_events_suppresses_result_with_content(stub_handler):
    """wrap_memory_list_events never captures event payloads returned by the SDK."""
    client = MockMemoryClient()

    wrap_memory_list_events(
        client.list_events,
        client,
        (),
        {"memory_id": "mem-123"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-123" in tool_call.arguments
    assert tool_call.tool_result is None


# ---------------------------------------------------------------------------
# wrap_memory_conversation_operation
# ---------------------------------------------------------------------------


def test_memory_conversation_operation_allowlists_metadata(stub_handler):
    """conversation wrappers avoid capturing messages and callbacks."""
    client = MockMemoryClient()
    wrapper = wrap_memory_conversation_operation("process_turn_with_llm")

    wrapper(
        client.process_turn_with_llm,
        client,
        (),
        {
            "memory_id": "mem-123",
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "messages": [{"role": "user", "content": "sensitive message"}],
            "llm_callback": lambda _messages: "sensitive response",
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.process_turn_with_llm"
    assert "mem-123" in tool_call.arguments
    assert "actor-1" in tool_call.arguments
    assert "sess-1" in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
    assert "llm_callback" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_memory_conversation_operation_suppresses_conversation_result(stub_handler):
    """conversation wrappers never capture returned conversation messages."""
    client = MockMemoryClient()
    wrapper = wrap_memory_conversation_operation("save_conversation")

    wrapper(
        client.save_conversation,
        client,
        (),
        {
            "memory_id": "mem-123",
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "conversation": [{"role": "user", "content": "sensitive message"}],
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.save_conversation"
    assert "mem-123" in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_memory_conversation_operation_keeps_safe_turn_count(stub_handler):
    """get_last_k_turns captures safe count metadata but not returned turns."""
    client = MockMemoryClient()
    wrapper = wrap_memory_conversation_operation("get_last_k_turns")

    wrapper(
        client.get_last_k_turns,
        client,
        (),
        {
            "memory_id": "mem-123",
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "k": 5,
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.get_last_k_turns"
    assert '"k": 5' in tool_call.arguments
    assert tool_call.tool_result is None


# ---------------------------------------------------------------------------
# MemorySessionManager wrappers
# ---------------------------------------------------------------------------


def test_memory_session_operation_allowlists_metadata(stub_handler):
    """session wrappers avoid messages, metadata, and returned payloads."""
    manager = MockMemorySessionManager()
    wrapper = wrap_memory_session_operation("add_turns")

    wrapper(
        manager.add_turns,
        manager,
        (),
        {
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "messages": [{"role": "user", "content": "sensitive message"}],
            "metadata": {"customer": "sensitive metadata"},
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.session.add_turns"
    assert "mem-session" in tool_call.arguments
    assert "actor-1" in tool_call.arguments
    assert "sess-1" in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_memory_session_list_events_suppresses_payload_result(stub_handler):
    """session list_events never captures returned event payloads."""
    manager = MockMemorySessionManager()
    wrapper = wrap_memory_session_operation("list_events")

    wrapper(
        manager.list_events,
        manager,
        (),
        {
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "include_payload": True,
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.session.list_events"
    assert '"include_payload": true' in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_memory_session_search_long_term_memories_retrieval(stub_handler):
    """session long-term memory search is represented as retrieval."""
    manager = MockMemorySessionManager()

    wrap_memory_session_search_long_term_memories(
        manager.search_long_term_memories,
        manager,
        (),
        {
            "query": "sensitive query",
            "namespace_prefix": "support/facts/actor-1/",
            "top_k": 5,
        },
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_retrievals) == 1
    assert len(stub_handler.started_tool_calls) == 0
    invocation = stub_handler.started_retrievals[0]
    assert invocation.data_source_id == "memory.session.search_long_term_memories"
    assert invocation.query == "sensitive query"
    assert invocation.top_k == 5
    assert invocation.documents_retrieved == 1


@pytest.mark.asyncio
async def test_memory_session_async_operation_suppresses_result(stub_handler):
    """async session wrappers keep content out of arguments and result."""
    manager = MockMemorySessionManager()
    wrapper = wrap_memory_session_async_operation("process_turn_with_llm_async")

    async def llm_callback(_user_input, _memories):
        return "sensitive assistant response"

    await wrapper(
        manager.process_turn_with_llm_async,
        manager,
        (),
        {
            "actor_id": "actor-1",
            "session_id": "sess-1",
            "user_input": "sensitive user input",
            "llm_callback": llm_callback,
            "retrieval_config": {"namespace": "sensitive namespace"},
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "memory.session.process_turn_with_llm_async"
    assert "actor-1" in tool_call.arguments
    assert "sess-1" in tool_call.arguments
    assert "sensitive" not in tool_call.arguments
    assert tool_call.tool_result is None
    assert len(stub_handler.stopped_tool_calls) == 1


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
    """wrap_memory_operation captures safe kwargs but suppresses result."""
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
    assert tool_call.tool_result is None


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


def test_memory_operation_captures_positional_args(stub_handler):
    """wrap_memory_operation includes positional arguments when content enabled."""

    def create_memory(memory_name, description=None):
        return {"memoryId": "mem-new"}

    wrapper = wrap_memory_operation("create_memory")
    wrapper(create_memory, None, ("my-memory",), {}, stub_handler, capture_content=True)

    tool_call = stub_handler.started_tool_calls[0]
    assert "my-memory" in tool_call.arguments


def test_memory_operation_suppresses_control_plane_config(stub_handler):
    """generic MemoryClient wrappers do not serialize IAM or strategy config."""

    def create_memory(
        memory_name=None,
        execution_role_arn=None,
        memory_execution_role_arn=None,
        event_expiry_duration=None,
    ):
        return {
            "memoryId": "mem-new",
            "executionRoleArn": execution_role_arn,
            "memoryExecutionRoleArn": memory_execution_role_arn,
            "eventExpiryDuration": event_expiry_duration,
        }

    wrapper = wrap_memory_operation("create_memory")
    wrapper(
        create_memory,
        None,
        (),
        {
            "memory_name": "my-memory",
            "execution_role_arn": "arn:aws:iam::123:role/secret",
            "memory_execution_role_arn": "arn:aws:iam::123:role/secret-memory",
            "event_expiry_duration": {"unit": "DAYS", "value": 30},
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "my-memory" in tool_call.arguments
    assert "secret" not in tool_call.arguments
    assert "arn:aws" not in tool_call.arguments
    assert "event_expiry_duration" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_memory_strategy_operation_suppresses_strategy_payloads(stub_handler):
    """strategy updates capture IDs only and never serialize full payloads."""

    def update_memory_strategies(memory_id=None, strategies=None, client_token=None):
        return {
            "memoryId": memory_id,
            "memoryStrategies": strategies,
            "clientToken": client_token,
        }

    wrapper = wrap_memory_operation("update_memory_strategies")
    wrapper(
        update_memory_strategies,
        None,
        (),
        {
            "memory_id": "mem-123",
            "strategies": [
                {
                    "strategyId": "strategy-secret",
                    "configuration": {"model": "secret-model"},
                    "streamDeliveryResources": {
                        "s3BucketArn": "arn:aws:s3:::secret-bucket"
                    },
                }
            ],
            "client_token": "secret-token",
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "mem-123" in tool_call.arguments
    assert "secret" not in tool_call.arguments
    assert "arn:aws" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_memory_operation_exception_fails_tool_call(stub_handler):
    """wrap_memory_operation fails the tool call on exception."""
    call_count = 0

    def failing_op(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Memory operation failed")

    wrapper = wrap_memory_operation("delete_memory")

    with pytest.raises(RuntimeError, match="Memory operation failed"):
        wrapper(failing_op, None, (), {}, stub_handler)

    assert len(stub_handler.failed_entities) == 1
    assert call_count == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type is RuntimeError
