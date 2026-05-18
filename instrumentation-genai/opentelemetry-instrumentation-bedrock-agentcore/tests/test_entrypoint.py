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

"""Tests for BedrockAgentCoreApp.entrypoint wrapper."""

import pytest

from opentelemetry.instrumentation.bedrock_agentcore.entrypoint_wrappers import (
    wrap_bedrock_agentcore_app_entrypoint,
)


def _capture_start_input_messages(stub_handler):
    started_input_messages = []
    original_start_workflow = stub_handler.start_workflow

    def start_workflow(workflow):
        started_input_messages.append(list(workflow.input_messages))
        return original_start_workflow(workflow)

    stub_handler.start_workflow = start_workflow
    return started_input_messages


def test_bedrock_agentcore_app_wrapper_sync(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should create a Workflow span for sync functions."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func  # simple passthrough decorator

    app = MockApp()
    started_input_messages = _capture_start_input_messages(stub_handler)

    def my_handler(payload):
        return {"status": "success"}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (my_handler,), {}, stub_handler, capture_content=True
    )
    result = wrapped({"input": "test"})

    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1
    workflow = stub_handler.started_workflows[0]
    assert workflow.name == "test_app"
    assert workflow.system == "bedrock-agentcore"
    assert len(started_input_messages) == 1
    assert started_input_messages[0][0].role == "user"
    assert '"input": "test"' in started_input_messages[0][0].parts[0].content
    assert result == {"status": "success"}


def test_bedrock_agentcore_app_wrapper_sync_keyword_event(stub_handler):
    """entrypoint input capture binds keyword arguments before span start."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()
    started_input_messages = _capture_start_input_messages(stub_handler)

    def my_handler(event):
        return {"status": "success", "event": event}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (my_handler,), {}, stub_handler, capture_content=True
    )
    result = wrapped(event={"input": "kw-test"})

    assert len(started_input_messages) == 1
    assert '"input": "kw-test"' in started_input_messages[0][0].parts[0].content
    assert result == {"status": "success", "event": {"input": "kw-test"}}


def test_bedrock_agentcore_app_wrapper_sync_exception(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should fail the Workflow on exception."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()
    call_count = 0

    def failing_handler(payload):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Service unavailable")

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (failing_handler,), {}, stub_handler
    )

    with pytest.raises(ConnectionError, match="Service unavailable"):
        wrapped({})

    assert len(stub_handler.failed_entities) == 1
    assert call_count == 1
    _workflow, error = stub_handler.failed_entities[0]
    assert error.type is ConnectionError


@pytest.mark.asyncio
async def test_bedrock_agentcore_app_wrapper_async(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should handle async entrypoint functions."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()
    started_input_messages = _capture_start_input_messages(stub_handler)

    async def async_handler(payload):
        return {"status": "async_success"}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (async_handler,), {}, stub_handler, capture_content=True
    )
    result = await wrapped({"input": "test"})

    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1
    assert len(started_input_messages) == 1
    assert started_input_messages[0][0].role == "user"
    assert '"input": "test"' in started_input_messages[0][0].parts[0].content
    assert result == {"status": "async_success"}


def test_bedrock_agentcore_app_wrapper_no_content_by_default(stub_handler):
    """entrypoint wrapper does not capture payloads unless content capture is enabled."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()
    started_input_messages = _capture_start_input_messages(stub_handler)

    def my_handler(payload):
        return {"status": "success", "echo": payload}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (my_handler,), {}, stub_handler
    )
    result = wrapped({"input": "secret"})

    workflow = stub_handler.started_workflows[0]
    assert result == {"status": "success", "echo": {"input": "secret"}}
    assert started_input_messages == [[]]
    assert workflow.input_messages == []
    assert workflow.output_messages == []
