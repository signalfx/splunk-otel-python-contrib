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
    wrap_code_interpreter_clear_context,
    wrap_code_interpreter_create,
    wrap_code_interpreter_download_file,
    wrap_code_interpreter_execute,
    wrap_code_interpreter_execute_command,
    wrap_code_interpreter_get,
    wrap_code_interpreter_install_packages,
    wrap_code_interpreter_list,
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

    def download_file(self, path):
        return "sensitive file content"

    def download_files(self, paths):
        return {path: "sensitive file content" for path in paths}

    def execute_command(self, command):
        return {"stdout": "sensitive command output", "stderr": ""}

    def clear_context(self):
        return {"cleared": True, "state": "sensitive context summary"}

    def create_code_interpreter(
        self,
        name=None,
        description=None,
        execution_role_arn=None,
        network_configuration=None,
    ):
        return {
            "codeInterpreterId": "ci-123",
            "executionRoleArn": execution_role_arn,
            "networkConfiguration": network_configuration,
            "resourceArn": "arn:aws:bedrock-agentcore:us-west-2:123:code-interpreter/ci-123",
        }

    def get_code_interpreter(self, interpreter_id):
        return {
            "codeInterpreterId": interpreter_id,
            "status": "READY",
            "executionRoleArn": "arn:aws:iam::123:role/secret",
            "networkConfiguration": {"subnets": ["subnet-secret"]},
        }

    def list_code_interpreters(
        self, interpreter_type=None, max_results=10, next_token=None
    ):
        return {
            "codeInterpreterSummaries": [
                {
                    "codeInterpreterId": "ci-1",
                    "status": "READY",
                    "executionRoleArn": "arn:aws:iam::123:role/secret",
                },
                {
                    "codeInterpreterId": "ci-2",
                    "status": "READY",
                    "executionRoleArn": "arn:aws:iam::123:role/secret2",
                },
            ],
            "nextToken": next_token or "opaque-secret-token",
        }

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
    assert (
        tool_call.attributes["bedrock.agentcore.code_interpreter.session_id"]
        == "session-123"
    )


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


def test_code_interpreter_execute_has_errors_attribute_set_regardless_of_content(
    stub_handler,
):
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
    assert (
        tool_call.attributes["bedrock.agentcore.code_interpreter.filename"]
        == "data.csv"
    )
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
    assert (
        tool_call.attributes["bedrock.agentcore.code_interpreter.filename"]
        == "data.csv"
    )
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_code_interpreter_upload_file_binds_path_parameter(stub_handler):
    """wrap_code_interpreter_upload_file supports SDKs using path instead of filename."""
    interpreter = MockCodeInterpreter()

    def upload_path(path, content, description=None):
        return {"fileId": "file-123", "filename": path}

    wrap_code_interpreter_upload_file(
        upload_path,
        interpreter,
        (),
        {"path": "path-data.csv", "content": b"test", "description": "Test data"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert (
        tool_call.attributes["bedrock.agentcore.code_interpreter.filename"]
        == "path-data.csv"
    )
    assert "path-data.csv" in tool_call.arguments


# ---------------------------------------------------------------------------
# protected result wrappers
# ---------------------------------------------------------------------------


def test_code_interpreter_download_file_suppresses_result_with_content(stub_handler):
    """download_file never captures raw file content as tool_result."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_download_file(
        interpreter.download_file,
        interpreter,
        (),
        {"path": "secret.txt"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.download_file"
    assert "secret.txt" in tool_call.arguments
    assert tool_call.tool_result is None


def test_code_interpreter_download_files_suppresses_result_with_content(stub_handler):
    """download_files never captures returned file-content maps."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_download_file(
        interpreter.download_files,
        interpreter,
        (),
        {"paths": ["secret-1.txt", "secret-2.txt"]},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.download_files"
    assert "secret-1.txt" in tool_call.arguments
    assert tool_call.tool_result is None


def test_code_interpreter_execute_command_suppresses_result_with_content(
    stub_handler,
):
    """execute_command never captures stdout/stderr as tool_result."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_execute_command(
        interpreter.execute_command,
        interpreter,
        (),
        {"command": "cat /tmp/secret"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.execute_command"
    assert "cat /tmp/secret" in tool_call.arguments
    assert tool_call.tool_result is None


def test_code_interpreter_clear_context_suppresses_result_with_content(stub_handler):
    """clear_context uses a dedicated wrapper so state details are not captured."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_clear_context(
        interpreter.clear_context,
        interpreter,
        (),
        {},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.clear_context"
    assert tool_call.arguments is None
    assert tool_call.tool_result is None


def test_code_interpreter_create_suppresses_infrastructure_result_with_content(
    stub_handler,
):
    """create_code_interpreter never captures returned ARNs or network config."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_create(
        interpreter.create_code_interpreter,
        interpreter,
        (),
        {
            "name": "analysis-runtime",
            "description": "safe description",
            "execution_role_arn": "arn:aws:iam::123:role/secret",
            "network_configuration": {"subnets": ["subnet-secret"]},
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.create_code_interpreter"
    assert "analysis-runtime" in tool_call.arguments
    assert "safe description" in tool_call.arguments
    assert "secret" not in tool_call.arguments
    assert tool_call.tool_result is None


def test_code_interpreter_get_suppresses_infrastructure_result_with_content(
    stub_handler,
):
    """get_code_interpreter never captures returned ARNs or network config."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_get(
        interpreter.get_code_interpreter,
        interpreter,
        (),
        {"interpreter_id": "ci-123"},
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.get_code_interpreter"
    assert "ci-123" in tool_call.arguments
    assert "secret" not in tool_call.arguments
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.id"] == "ci-123"
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.status"] == "READY"
    assert tool_call.tool_result is None


def test_code_interpreter_list_suppresses_infrastructure_result_with_content(
    stub_handler,
):
    """list_code_interpreters captures safe filters but never result config."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_list(
        interpreter.list_code_interpreters,
        interpreter,
        (),
        {
            "interpreter_type": "CUSTOM",
            "max_results": 5,
            "next_token": "opaque-secret-token",
        },
        stub_handler,
        capture_content=True,
    )

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "code_interpreter.list_code_interpreters"
    assert "CUSTOM" in tool_call.arguments
    assert "5" in tool_call.arguments
    assert "opaque-secret-token" not in tool_call.arguments
    assert tool_call.attributes["bedrock.agentcore.code_interpreter.count"] == 2
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


def test_code_interpreter_operation_captures_positional_args(stub_handler):
    """wrap_code_interpreter_operation includes positional arguments when content enabled."""

    def get_session(session_id, include_logs=False):
        return {"sessionId": session_id, "status": "ACTIVE"}

    wrapper = wrap_code_interpreter_operation("get_session")
    wrapper(get_session, None, ("sess-pos",), {}, stub_handler, capture_content=True)

    tool_call = stub_handler.started_tool_calls[0]
    assert "sess-pos" in tool_call.arguments


def test_code_interpreter_start_no_content_suppresses_result(stub_handler):
    """wrap_code_interpreter_start suppresses tool_result when capture_content=False."""
    interpreter = MockCodeInterpreter()

    wrap_code_interpreter_start(interpreter.start, interpreter, (), {}, stub_handler)

    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.tool_result is None


def test_code_interpreter_operation_exception_fails_tool_call(stub_handler):
    """wrap_code_interpreter_operation fails the tool call on exception."""
    call_count = 0

    def failing_op(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Code execution failed")

    wrapper = wrap_code_interpreter_operation("invoke")

    with pytest.raises(RuntimeError, match="Code execution failed"):
        wrapper(failing_op, None, (), {}, stub_handler)

    assert len(stub_handler.failed_entities) == 1
    assert call_count == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type is RuntimeError


# ---------------------------------------------------------------------------
# Exception propagation (original tests)
# ---------------------------------------------------------------------------


def test_code_interpreter_exception_fails_tool_call(stub_handler):
    """Exceptions in code interpreter operations fail the tool call."""
    interpreter = MockCodeInterpreter()
    call_count = 0

    def failing_execute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
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
    assert call_count == 1

    tool_call, error = stub_handler.failed_entities[0]
    assert error.type is RuntimeError
    assert "Code execution failed" in error.message
