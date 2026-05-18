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

"""Tests for BrowserClient instrumentation."""

import pytest

from opentelemetry.instrumentation.bedrock_agentcore.browser_wrappers import (
    wrap_browser_create_browser,
    wrap_browser_generate_live_view_url,
    wrap_browser_generate_ws_headers,
    wrap_browser_get_browser,
    wrap_browser_get_session,
    wrap_browser_list_browsers,
    wrap_browser_operation,
    wrap_browser_release_control,
    wrap_browser_start,
    wrap_browser_stop,
    wrap_browser_take_control,
    wrap_browser_update_stream,
)


class MockBrowserClient:
    """Mock BrowserClient for testing."""

    def __init__(self):
        self.session_id = None

    def start(self, browser_id=None):
        self.session_id = "browser-session-123"
        return {"sessionId": self.session_id}

    def stop(self):
        return True

    def take_control(self):
        return {"status": "control_taken"}

    def release_control(self):
        return {"status": "control_released"}

    def get_session(self, browser_id=None, session_id=None):
        return {
            "sessionId": session_id,
            "status": "ACTIVE",
            "signingMaterial": "secret-session-signing-material",
        }

    def generate_ws_headers(self):
        return "wss://example.com/session", {"Authorization": "secret-token"}

    def generate_live_view_url(self):
        return "https://example.com/live-view?X-Amz-Signature=secret"

    def create_browser(
        self,
        name=None,
        execution_role_arn=None,
        network_configuration=None,
        recording_config=None,
        client_token=None,
    ):
        return {
            "browserId": "browser-123",
            "executionRoleArn": execution_role_arn,
            "networkConfiguration": network_configuration,
            "recordingConfig": recording_config,
            "clientToken": client_token,
        }

    def get_browser(self, browser_id):
        return {
            "browserId": browser_id,
            "status": "ACTIVE",
            "browserSigningConfig": {"privateKey": "secret"},
            "certificateReference": "cert-secret",
        }

    def list_browsers(self, max_results=None, next_token=None):
        return {
            "browserSummaries": [
                {
                    "browserId": "browser-1",
                    "executionRoleArn": "arn:aws:iam::123:role/secret",
                },
                {
                    "browserId": "browser-2",
                    "executionRoleArn": "arn:aws:iam::123:role/secret2",
                },
            ],
            "nextToken": next_token or "opaque-secret-token",
        }

    def update_stream(
        self,
        browser_id=None,
        session_id=None,
        stream_delivery_resources=None,
    ):
        return {
            "browserId": browser_id,
            "sessionId": session_id,
            "streamStatus": "ACTIVE",
            "streamDeliveryResources": stream_delivery_resources,
        }

    def list_sessions(self, **kwargs):
        return [{"sessionId": "s1"}, {"sessionId": "s2"}]


# ---------------------------------------------------------------------------
# wrap_browser_start
# ---------------------------------------------------------------------------


def test_browser_start_creates_tool_call(stub_handler):
    """wrap_browser_start creates a ToolCall span with browser_id attribute."""
    browser = MockBrowserClient()

    wrap_browser_start(
        browser.start, browser, (), {"browser_id": "browser-123"}, stub_handler
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.start"
    assert tool_call.system == "bedrock-agentcore"
    assert tool_call.attributes["bedrock.agentcore.tool.type"] == "browser"
    assert tool_call.attributes["bedrock.agentcore.browser.id"] == "browser-123"


def test_browser_start_binds_positional_browser_id(stub_handler):
    """wrap_browser_start extracts browser_id via the method signature."""
    browser = MockBrowserClient()

    wrap_browser_start(browser.start, browser, ("browser-pos",), {}, stub_handler)

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.attributes["bedrock.agentcore.browser.id"] == "browser-pos"


# ---------------------------------------------------------------------------
# wrap_browser_stop
# ---------------------------------------------------------------------------


def test_browser_stop_tracks_session(stub_handler):
    """wrap_browser_stop records session_id attribute and creates span."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    wrap_browser_stop(browser.stop, browser, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.stop"
    assert (
        tool_call.attributes["bedrock.agentcore.browser.session_id"]
        == "browser-session-123"
    )


def test_browser_stop_with_content(stub_handler):
    """wrap_browser_stop captures result when content enabled."""
    browser = MockBrowserClient()

    wrap_browser_stop(browser.stop, browser, (), {}, stub_handler, capture_content=True)

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.tool_result is not None


def test_browser_stop_no_content_by_default(stub_handler):
    """wrap_browser_stop suppresses result when capture_content=False."""
    browser = MockBrowserClient()

    wrap_browser_stop(browser.stop, browser, (), {}, stub_handler)

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.tool_result is None


# ---------------------------------------------------------------------------
# wrap_browser_take_control / wrap_browser_release_control
# ---------------------------------------------------------------------------


def test_browser_take_control_creates_tool_call(stub_handler):
    """wrap_browser_take_control creates ToolCall span."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    wrap_browser_take_control(browser.take_control, browser, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.take_control"
    assert tool_call.attributes["bedrock.agentcore.browser.operation"] == "take_control"


def test_browser_release_control_creates_tool_call(stub_handler):
    """wrap_browser_release_control creates ToolCall span."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    wrap_browser_release_control(browser.release_control, browser, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.release_control"


# ---------------------------------------------------------------------------
# wrap_browser_get_session
# ---------------------------------------------------------------------------


def test_browser_get_session_tracks_status(stub_handler):
    """wrap_browser_get_session records session_status attribute."""
    browser = MockBrowserClient()

    wrap_browser_get_session(
        browser.get_session,
        browser,
        (),
        {"browser_id": "browser-123", "session_id": "session-456"},
        stub_handler,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.get_session"
    assert tool_call.attributes["bedrock.agentcore.browser.session_status"] == "ACTIVE"


def test_browser_get_session_with_content(stub_handler):
    """wrap_browser_get_session captures arguments but suppresses result."""
    browser = MockBrowserClient()

    wrap_browser_get_session(
        browser.get_session,
        browser,
        (),
        {"browser_id": "browser-123", "session_id": "session-456"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "browser-123" in tool_call.arguments
    assert tool_call.tool_result is None


def test_browser_get_session_no_content_by_default(stub_handler):
    """wrap_browser_get_session suppresses arguments by default."""
    browser = MockBrowserClient()

    wrap_browser_get_session(
        browser.get_session,
        browser,
        (),
        {"browser_id": "browser-123", "session_id": "session-456"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None


# ---------------------------------------------------------------------------
# protected result wrappers
# ---------------------------------------------------------------------------


def test_browser_generate_ws_headers_suppresses_result_with_content(stub_handler):
    """generate_ws_headers never captures auth headers as tool_result."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    wrap_browser_generate_ws_headers(
        browser.generate_ws_headers,
        browser,
        (),
        {},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.generate_ws_headers"
    assert (
        tool_call.attributes["bedrock.agentcore.browser.session_id"]
        == "browser-session-123"
    )
    assert tool_call.tool_result is None


def test_browser_generate_live_view_url_suppresses_result_with_content(stub_handler):
    """generate_live_view_url never captures presigned URLs as tool_result."""
    browser = MockBrowserClient()

    wrap_browser_generate_live_view_url(
        browser.generate_live_view_url,
        browser,
        (),
        {},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.generate_live_view_url"
    assert tool_call.tool_result is None


def test_browser_create_browser_suppresses_control_plane_config(stub_handler):
    """create_browser captures only allowlisted fields and never result config."""
    browser = MockBrowserClient()

    wrap_browser_create_browser(
        browser.create_browser,
        browser,
        (),
        {
            "name": "browser-name",
            "execution_role_arn": "arn:aws:iam::123:role/secret",
            "network_configuration": {"subnets": ["subnet-secret"]},
            "recording_config": {"bucket": "secret-bucket"},
            "client_token": "secret-token",
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.create_browser"
    assert "browser-name" in tool_call.arguments
    assert "secret" not in tool_call.arguments
    assert tool_call.attributes["bedrock.agentcore.browser.id"] == "browser-123"
    assert tool_call.tool_result is None


def test_browser_get_browser_suppresses_control_plane_config(stub_handler):
    """get_browser captures browser_id but never result config."""
    browser = MockBrowserClient()

    wrap_browser_get_browser(
        browser.get_browser,
        browser,
        (),
        {"browser_id": "browser-123"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.get_browser"
    assert "browser-123" in tool_call.arguments
    assert tool_call.attributes["bedrock.agentcore.browser.id"] == "browser-123"
    assert tool_call.attributes["bedrock.agentcore.browser.status"] == "ACTIVE"
    assert tool_call.tool_result is None


def test_browser_list_browsers_suppresses_control_plane_config(stub_handler):
    """list_browsers captures safe paging metadata but never browser configs."""
    browser = MockBrowserClient()

    wrap_browser_list_browsers(
        browser.list_browsers,
        browser,
        (),
        {"max_results": 10, "next_token": "opaque-secret-token"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.list_browsers"
    assert "10" in tool_call.arguments
    assert "opaque-secret-token" not in tool_call.arguments
    assert tool_call.attributes["bedrock.agentcore.browser.count"] == 2
    assert tool_call.tool_result is None


def test_browser_create_browser_ignores_empty_result_id(stub_handler):
    """empty result IDs are not recorded as browser.id attributes."""
    browser = MockBrowserClient()

    def create_empty_id(name=None):
        return {"browserId": "", "status": "CREATED"}

    wrap_browser_create_browser(
        create_empty_id,
        browser,
        (),
        {"name": "browser-name"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "bedrock.agentcore.browser.id" not in tool_call.attributes


def test_browser_update_stream_suppresses_delivery_resources(stub_handler):
    """update_stream captures IDs but never stream delivery resource config."""
    browser = MockBrowserClient()

    wrap_browser_update_stream(
        browser.update_stream,
        browser,
        (),
        {
            "browser_id": "browser-123",
            "session_id": "session-456",
            "stream_delivery_resources": {
                "s3BucketArn": "arn:aws:s3:::secret-bucket",
                "kmsKeyArn": "arn:aws:kms:us-west-2:123:key/secret",
                "recordingPrefix": "secret-prefix",
            },
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.update_stream"
    assert "browser-123" in tool_call.arguments
    assert "session-456" in tool_call.arguments
    assert "secret" not in tool_call.arguments
    assert tool_call.attributes["bedrock.agentcore.browser.id"] == "browser-123"
    assert tool_call.attributes["bedrock.agentcore.browser.session_id"] == "session-456"
    assert tool_call.attributes["bedrock.agentcore.browser.stream_status"] == "ACTIVE"
    assert tool_call.tool_result is None


# ---------------------------------------------------------------------------
# wrap_browser_operation (generic factory)
# ---------------------------------------------------------------------------


def test_browser_operation_creates_tool_call(stub_handler):
    """wrap_browser_operation factory creates a ToolCall span."""
    browser = MockBrowserClient()
    wrapper = wrap_browser_operation("list_sessions")

    wrapper(browser.list_sessions, browser, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1
    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.list_sessions"
    assert tool_call.system == "bedrock-agentcore"


def test_browser_operation_with_content(stub_handler):
    """wrap_browser_operation captures kwargs and result when content enabled."""
    browser = MockBrowserClient()
    wrapper = wrap_browser_operation("list_sessions")

    wrapper(
        browser.list_sessions,
        browser,
        (),
        {"filter": "active"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "active" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_browser_operation_no_content_by_default(stub_handler):
    """wrap_browser_operation suppresses arguments and result by default."""
    browser = MockBrowserClient()
    wrapper = wrap_browser_operation("list_sessions")

    wrapper(browser.list_sessions, browser, (), {"filter": "active"}, stub_handler)

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_browser_operation_captures_positional_args(stub_handler):
    """wrap_browser_operation includes positional arguments when content enabled."""

    def list_sessions(browser_id, status=None):
        return [{"sessionId": "s1"}]

    wrapper = wrap_browser_operation("list_sessions")
    wrapper(
        list_sessions, None, ("browser-123",), {}, stub_handler, capture_content=True
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "browser-123" in tool_call.arguments


def test_browser_start_no_content_suppresses_arguments_and_result(stub_handler):
    """wrap_browser_start suppresses arguments and tool_result when capture_content=False."""
    browser = MockBrowserClient()

    wrap_browser_start(
        browser.start, browser, (), {"browser_id": "browser-123"}, stub_handler
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_browser_operation_exception_fails_tool_call(stub_handler):
    """wrap_browser_operation fails the tool call on exception."""
    call_count = 0

    def failing_op(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Browser connection failed")

    wrapper = wrap_browser_operation("create_browser")

    with pytest.raises(ConnectionError, match="Browser connection failed"):
        wrapper(failing_op, None, (), {}, stub_handler)

    assert len(stub_handler.failed_entities) == 1
    assert call_count == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type is ConnectionError


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_browser_exception_fails_tool_call(stub_handler):
    """Exceptions in browser operations fail the tool call."""
    browser = MockBrowserClient()
    call_count = 0

    def failing_start(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Browser connection failed")

    with pytest.raises(ConnectionError, match="Browser connection failed"):
        wrap_browser_start(failing_start, browser, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.failed_entities) == 1
    assert call_count == 1

    tool_call, error = stub_handler.failed_entities[0]
    assert error.type is ConnectionError
    assert "Browser connection failed" in error.message
