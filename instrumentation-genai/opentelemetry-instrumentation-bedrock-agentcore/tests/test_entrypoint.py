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


def test_bedrock_agentcore_app_wrapper_sync(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should create a Workflow span for sync functions."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func  # simple passthrough decorator

    app = MockApp()

    def my_handler(payload):
        return {"status": "success"}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (my_handler,), {}, stub_handler
    )
    result = wrapped({"input": "test"})

    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1
    workflow = stub_handler.started_workflows[0]
    assert workflow.name == "test_app"
    assert workflow.system == "bedrock-agentcore"
    assert result == {"status": "success"}


def test_bedrock_agentcore_app_wrapper_sync_exception(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should fail the Workflow on exception."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()

    def failing_handler(payload):
        raise ConnectionError("Service unavailable")

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (failing_handler,), {}, stub_handler
    )

    with pytest.raises(ConnectionError, match="Service unavailable"):
        wrapped({})

    assert len(stub_handler.failed_entities) == 1
    _workflow, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"


@pytest.mark.asyncio
async def test_bedrock_agentcore_app_wrapper_async(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should handle async entrypoint functions."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()

    async def async_handler(payload):
        return {"status": "async_success"}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (async_handler,), {}, stub_handler
    )
    result = await wrapped({"input": "test"})

    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1
    assert result == {"status": "async_success"}
