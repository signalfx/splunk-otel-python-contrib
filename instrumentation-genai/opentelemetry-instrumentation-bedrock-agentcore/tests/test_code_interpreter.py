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

from opentelemetry.instrumentation.bedrock_agentcore.code_interpreter_wrappers import (
    wrap_code_interpreter_execute,
    wrap_code_interpreter_install_packages,
    wrap_code_interpreter_operation,
    wrap_code_interpreter_start,
    wrap_code_interpreter_stop,
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

    def get_session(self, session_id=None):
        return {"sessionId": session_id, "status": "ACTIVE"}


# ---------------------------------------------------------------------------
# wrap_code_interpreter_start
# ---------------------------------------------------------------------------

def test_code_interpreter_start_creates_tool_call(stub_handler):
    """wrap_code_interpreter_start creates a ToolCall span."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_start(interpreter.start, interpreter, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.start"
    assert tool_call.system == "bedrock-agentcore"
    assert tool_call.attributes["bedrock.agentcore.tool.type"] == "code_interpreter"
    assert (
        "session-123"
        in tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"]
    )


# ---------------------------------------------------------------------------
# wrap_code_interpreter_stop
# ---------------------------------------------------------------------------

def test_code_interpreter_stop_creates_tool_call(stub_handler):
    """wrap_code_interpreter_stop creates a ToolCall span."""
    interpreter = MockCodeInterpreter()
    interpreter.session_id = "session-123"

    wrap_code_interpreter_stop(interpreter.stop, interpreter, (), {}, stub_handler)

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1
    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.stop"
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"] == "session-123"


# ---------------------------------------------------------------------------
# wrap_code_interpreter_execute
# ---------------------------------------------------------------------------

def test_code_interpreter_execute_with_content(stub_handler):
    """wrap_code_interpreter_execute captures code and output when content enabled."""
    interpreter = MockCodeInterpreter()
    interpreter.session_id = "session-123"

    wrap_code_interpreter_execute(
        interpreter.execute_code,
        interpreter,
        (),
        {"code": "print('Hello, World!')"},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.execute"
    assert "print('Hello, World!')" in tool_call.arguments
    assert tool_call.tool_result is not None
    assert "Hello, World!" in tool_call.tool_result


def test_code_interpreter_execute_no_content_by_default(stub_handler):
    """wrap_code_interpreter_execute suppresses code and output when capture_content=False."""
    interpreter = MockCodeInterpreter()
    interpreter.session_id = "session-123"

    wrap_code_interpreter_execute(
        interpreter.execute_code,
        interpreter,
        (),
        {"code": "print('Hello, World!')"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.execute"
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_code_interpreter_execute_has_errors_attribute_set_regardless_of_content(stub_handler):
    """has_errors attribute is set even when content capture is disabled."""
    interpreter = MockCodeInterpreter()

    def execute_with_errors(code):
        return {"output": "", "errors": ["SyntaxError: invalid syntax"]}

    wrap_code_interpreter_execute(
        execute_with_errors,
        interpreter,
        (),
        {"code": "bad code"},
        stub_handler,
        capture_content=False,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.has_errors"] is True
    assert tool_call.tool_result is None


def test_code_interpreter_execute_code_from_positional_args(stub_handler):
    """wrap_code_interpreter_execute extracts code from positional args."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_execute(
        interpreter.execute_code,
        interpreter,
        ("print('from args')",),
        {},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "from args" in tool_call.arguments


def test_code_interpreter_execute_kwargs_preferred_over_args(stub_handler):
    """wrap_code_interpreter_execute prefers kwargs over positional for code."""

    def mock_execute(*args, **kwargs):
        return {"output": "ok", "errors": []}

    wrap_code_interpreter_execute(
        mock_execute,
        MockCodeInterpreter(),
        ("code from args",),
        {"code": "code from kwargs"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "code from kwargs" in tool_call.arguments


# ---------------------------------------------------------------------------
# wrap_code_interpreter_install_packages
# ---------------------------------------------------------------------------

def test_code_interpreter_install_packages_with_content(stub_handler):
    """wrap_code_interpreter_install_packages captures package list when content enabled."""
    interpreter = MockCodeInterpreter()
    packages = ["pandas", "numpy", "matplotlib"]

    wrap_code_interpreter_install_packages(
        interpreter.install_packages,
        interpreter,
        (),
        {"packages": packages},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.install_packages"
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.package_count"] == 3
    assert "pandas" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_code_interpreter_install_packages_no_content_by_default(stub_handler):
    """wrap_code_interpreter_install_packages suppresses list when capture_content=False."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_install_packages(
        interpreter.install_packages,
        interpreter,
        (),
        {"packages": ["pandas"]},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None
    # package_count attribute is safe metadata — always captured
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.package_count"] == 1


# ---------------------------------------------------------------------------
# wrap_code_interpreter_upload_file
# ---------------------------------------------------------------------------

def test_code_interpreter_upload_file_with_content(stub_handler):
    """wrap_code_interpreter_upload_file captures filename and result when content enabled."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_upload_file(
        interpreter.upload_file,
        interpreter,
        (),
        {"filename": "data.csv", "content": b"test", "description": "Test data"},
        stub_handler,
        capture_content=True,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.upload_file"
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.filename"] == "data.csv"
    assert "data.csv" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_code_interpreter_upload_file_no_content_by_default(stub_handler):
    """wrap_code_interpreter_upload_file suppresses arguments when capture_content=False."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_upload_file(
        interpreter.upload_file,
        interpreter,
        (),
        {"filename": "data.csv", "content": b"test"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    # filename attribute is safe metadata — always captured
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.filename"] == "data.csv"
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


# ---------------------------------------------------------------------------
# wrap_code_interpreter_operation (generic factory)
# ---------------------------------------------------------------------------

def test_code_interpreter_operation_creates_tool_call(stub_handler):
    """wrap_code_interpreter_operation factory creates a ToolCall span."""
    interpreter = MockCodeInterpreter()
    wrapper = wrap_code_interpreter_operation("get_session")

    wrapper(
        interpreter.get_session,
        interpreter,
        (),
        {"session_id": "sess-123"},
        stub_handler,
    )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.stopped_tool_calls) == 1
    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.get_session"
    assert tool_call.system == "bedrock-agentcore"


def test_code_interpreter_operation_with_content(stub_handler):
    """wrap_code_interpreter_operation captures kwargs and result when content enabled."""
    interpreter = MockCodeInterpreter()
    wrapper = wrap_code_interpreter_operation("get_session")

    wrapper(
        interpreter.get_session,
        interpreter,
        (),
        {"session_id": "sess-123"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert "sess-123" in tool_call.arguments
    assert tool_call.tool_result is not None


def test_code_interpreter_operation_no_content_by_default(stub_handler):
    """wrap_code_interpreter_operation suppresses arguments and result by default."""
    interpreter = MockCodeInterpreter()
    wrapper = wrap_code_interpreter_operation("get_session")

    wrapper(
        interpreter.get_session,
        interpreter,
        (),
        {"session_id": "sess-123"},
        stub_handler,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_code_interpreter_operation_exception_fails_tool_call(stub_handler):
    """wrap_code_interpreter_operation fails the tool call on exception."""

    def failing_op(*args, **kwargs):
        raise RuntimeError("Code execution failed")

    wrapper = wrap_code_interpreter_operation("invoke")

    with pytest.raises(RuntimeError, match="Code execution failed"):
        wrapper(failing_op, None, (), {}, stub_handler)

    assert len(stub_handler.failed_entities) == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"


# ---------------------------------------------------------------------------
# Exception propagation (original tests)
# ---------------------------------------------------------------------------

def test_code_interpreter_exception_fails_tool_call(stub_handler):
    """Exceptions in code interpreter operations fail the tool call."""
    interpreter = MockCodeInterpreter()

    def failing_execute(*args, **kwargs):
        raise RuntimeError("Code execution failed")

    with pytest.raises(RuntimeError, match="Code execution failed"):
        wrap_code_interpreter_execute(
            failing_execute,
            interpreter,
            (),
            {"code": "raise Exception()"},
            stub_handler,
        )

    assert len(stub_handler.started_tool_calls) == 1
    assert len(stub_handler.failed_entities) == 1

    tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"
    assert "Code execution failed" in error.message
