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
    wrap_browser_get_session,
    wrap_browser_operation,
    wrap_browser_release_control,
    wrap_browser_start,
    wrap_browser_stop,
    wrap_browser_take_control,
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
        return {"sessionId": session_id, "sessionStatus": "ACTIVE"}

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
    """wrap_browser_get_session captures arguments when content enabled."""
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


def test_browser_operation_exception_fails_tool_call(stub_handler):
    """wrap_browser_operation fails the tool call on exception."""

    def failing_op(*args, **kwargs):
        raise ConnectionError("Browser connection failed")

    wrapper = wrap_browser_operation("create_browser")

    with pytest.raises(ConnectionError, match="Browser connection failed"):
        wrapper(failing_op, None, (), {}, stub_handler)

    assert len(stub_handler.failed_entities) == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_browser_exception_fails_tool_call(stub_handler):
    """Exceptions in browser operations fail the tool call."""
    browser = MockBrowserClient()

    def failing_start(*args, **kwargs):
        raise ConnectionError("Browser connection failed")

    with pytest.raises(ConnectionError, match="Browser connection failed"):
        wrap_browser_start(failing_start, browser, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.failed_entities) == 1

    tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"
    assert "Browser connection failed" in error.message
