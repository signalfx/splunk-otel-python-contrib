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

"""Tests for CodeInterpreter instrumentation."""

import pytest

from opentelemetry.instrumentation.strands.code_interpreter_wrappers import (
    wrap_code_interpreter_execute,
    wrap_code_interpreter_install_packages,
    wrap_code_interpreter_start,
    wrap_code_interpreter_upload_file,
)


class MockCodeInterpreter:
    """Mock CodeInterpreter for testing."""

    def __init__(self):
        self.session_id = None

    def start(self):
        self.session_id = "session-123"
        return {"sessionId": self.session_id}

    def stop(self):
        return True

    def execute_code(self, code):
        return {"output": "Hello, World!", "errors": []}

    def install_packages(self, packages):
        return {"installed": packages}

    def upload_file(self, filename, content, description=None):
        return {"fileId": "file-123", "filename": filename}


def test_code_interpreter_start_creates_tool_call(stub_handler):
    """Test that wrap_code_interpreter_start creates ToolCall span."""
    interpreter = MockCodeInterpreter()

    # Call wrapped method
    wrap_code_interpreter_start(interpreter.start, interpreter, (), {}, stub_handler)

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.start"
    assert tool_call.system == "bedrock-agentcore"
    assert tool_call.attributes["bedrock.agentcore.tool.type"] == "code_interpreter"

    # Verify session ID was captured
    assert (
        "session-123"
        in tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"]
    )


def test_code_interpreter_execute_tracks_code_execution(stub_handler):
    """Test that wrap_code_interpreter_execute tracks code execution."""
    interpreter = MockCodeInterpreter()
    interpreter.session_id = "session-123"

    # Call wrapped method
    wrap_code_interpreter_execute(
        interpreter.execute_code,
        interpreter,
        (),
        {"code": "print('Hello, World!')"},
        stub_handler,
    )

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.execute"
    assert "print('Hello, World!')" in tool_call.arguments


def test_code_interpreter_install_packages_tracks_packages(stub_handler):
    """Test that wrap_code_interpreter_install_packages tracks package list."""
    interpreter = MockCodeInterpreter()
    packages = ["pandas", "numpy", "matplotlib"]

    # Call wrapped method
    wrap_code_interpreter_install_packages(
        interpreter.install_packages,
        interpreter,
        (),
        {"packages": packages},
        stub_handler,
    )

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.install_packages"
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.package_count"] == 3


def test_code_interpreter_upload_file_tracks_filename(stub_handler):
    """Test that wrap_code_interpreter_upload_file tracks file uploads."""
    interpreter = MockCodeInterpreter()

    # Call wrapped method
    wrap_code_interpreter_upload_file(
        interpreter.upload_file,
        interpreter,
        (),
        {"filename": "data.csv", "content": b"test", "description": "Test data"},
        stub_handler,
    )

    # Verify tool call was started and stopped
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.upload_file"
    assert (
        tool_call.attributes["bedrock.agentcore.code_interpreter.filename"]
        == "data.csv"
    )


def test_code_interpreter_exception_fails_tool_call(stub_handler):
    """Test that exceptions in code interpreter operations fail the tool call."""
    interpreter = MockCodeInterpreter()

    # Create a wrapper that raises exception
    def failing_execute(*args, **kwargs):
        raise RuntimeError("Code execution failed")

    # Call wrapped method and expect exception
    with pytest.raises(RuntimeError, match="Code execution failed"):
        wrap_code_interpreter_execute(
            failing_execute,
            interpreter,
            (),
            {"code": "raise Exception()"},
            stub_handler,
        )

    # Verify tool call was started and failed
    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.failed_entities) == 1

    tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"
    assert "Code execution failed" in error.message
