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


def test_browser_start_creates_tool_call(stub_handler):
    """Test that wrap_browser_start creates ToolCall span."""
    browser = MockBrowserClient()

    # Call wrapped method
    wrap_browser_start(
        browser.start, browser, (), {"browser_id": "browser-123"}, stub_handler
    )

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.start"
    assert tool_call.system == "bedrock-agentcore"
    assert tool_call.attributes["bedrock.agentcore.tool.type"] == "browser"
    assert tool_call.attributes["bedrock.agentcore.browser.id"] == "browser-123"


def test_browser_stop_tracks_session(stub_handler):
    """Test that wrap_browser_stop tracks browser session termination."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    # Call wrapped method
    wrap_browser_stop(browser.stop, browser, (), {}, stub_handler)

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.stop"
    assert (
        tool_call.attributes["bedrock.agentcore.browser.session_id"]
        == "browser-session-123"
    )


def test_browser_take_control_creates_tool_call(stub_handler):
    """Test that wrap_browser_take_control creates ToolCall span."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    # Call wrapped method
    wrap_browser_take_control(browser.take_control, browser, (), {}, stub_handler)

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.take_control"
    assert tool_call.attributes["bedrock.agentcore.browser.operation"] == "take_control"


def test_browser_release_control_creates_tool_call(stub_handler):
    """Test that wrap_browser_release_control creates ToolCall span."""
    browser = MockBrowserClient()
    browser.session_id = "browser-session-123"

    # Call wrapped method
    wrap_browser_release_control(browser.release_control, browser, (), {}, stub_handler)

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.release_control"


def test_browser_get_session_tracks_status(stub_handler):
    """Test that wrap_browser_get_session tracks session status."""
    browser = MockBrowserClient()

    # Call wrapped method
    wrap_browser_get_session(
        browser.get_session,
        browser,
        (),
        {"browser_id": "browser-123", "session_id": "session-456"},
        stub_handler,
    )

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "browser.get_session"
    assert tool_call.attributes["bedrock.agentcore.browser.session_status"] == "ACTIVE"


def test_browser_exception_fails_tool_call(stub_handler):
    """Test that exceptions in browser operations fail the tool call."""
    browser = MockBrowserClient()

    # Create a wrapper that raises exception
    def failing_start(*args, **kwargs):
        raise ConnectionError("Browser connection failed")

    # Call wrapped method and expect exception
    with pytest.raises(ConnectionError, match="Browser connection failed"):
        wrap_browser_start(failing_start, browser, (), {}, stub_handler)

    # Verify tool call was started and failed
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.failed_entities) == 1

    tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"
    assert "Browser connection failed" in error.message
