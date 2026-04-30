#!/usr/bin/env python3
"""
MCP Development Assistant Server

A sample MCP server with useful development tools to demonstrate
OpenTelemetry instrumentation with Splunk Distro.

Usage:
    # Set environment variables for observability
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
    export OTEL_SERVICE_NAME="mcp-dev-assistant-server"
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"

    # Run the server
    python dev_assistant_server.py

NOTE: MCP servers that use stdio transport CANNOT use ConsoleSpanExporter
      because it writes to stdout which interferes with MCP protocol.
      Use OTLP exporter for production or export to stderr for debugging.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel

# Import and apply instrumentation BEFORE creating FastMCP instance
from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

# Configure OpenTelemetry
# NOTE: For stdio-based MCP servers, we must NOT use ConsoleSpanExporter
# because it writes to stdout which breaks the MCP protocol.
# Instead, use OTLP exporter for production or skip console export.
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# Check if OTLP endpoint is configured
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp_endpoint:
    # Use OTLP exporter for production
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        print("Using OTLP exporter", file=sys.stderr)
    except ImportError:
        print("OTLP exporter not available, spans not exported", file=sys.stderr)
        trace.set_tracer_provider(TracerProvider())
else:
    # No exporter - spans are created but not exported
    # This is fine for demo - the client will show its spans
    trace.set_tracer_provider(TracerProvider())
    print("No OTLP endpoint configured, server spans not exported", file=sys.stderr)

# Apply FastMCP instrumentation
FastMCPInstrumentor().instrument()

# Load environment variables
load_dotenv()

# Initialize the MCP server
server = FastMCP("dev-assistant")


class FileInfo(BaseModel):
    """File information structure."""

    name: str
    size: int
    modified: str
    is_directory: bool


class GitStatus(BaseModel):
    """Git status structure."""

    branch: str
    staged: List[str]
    modified: List[str]
    untracked: List[str]
    ahead: int
    behind: int


class ProcessResult(BaseModel):
    """Process execution result."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


@server.tool()
def list_files(directory: str = ".") -> List[FileInfo]:
    """
    List files and directories in the specified path.

    Args:
        directory: The directory path to list (defaults to current directory)

    Returns:
        List of FileInfo objects with file details
    """
    try:
        path = Path(directory).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")

        files = []
        for item in path.iterdir():
            stat = item.stat()
            files.append(
                FileInfo(
                    name=item.name,
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    is_directory=item.is_dir(),
                )
            )

        return sorted(files, key=lambda f: (not f.is_directory, f.name.lower()))
    except Exception as e:
        raise RuntimeError(f"Failed to list directory: {str(e)}")


@server.tool()
def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (optional)

    Returns:
        File contents as a string
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        if not path.is_file():
            raise ValueError(f"{file_path} is not a file")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"... (truncated at {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines)
            else:
                return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {str(e)}")


@server.tool()
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write to the file

    Returns:
        Success message with file path and size
    """
    try:
        path = Path(file_path).expanduser().resolve()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        stat = path.stat()
        return f"Successfully wrote to {path} ({stat.st_size} bytes)"
    except Exception as e:
        raise RuntimeError(f"Failed to write file: {str(e)}")


@server.tool()
def run_command(
    command: str, working_directory: str = ".", timeout: int = 30
) -> ProcessResult:
    """
    Execute a shell command and return the result.

    Args:
        command: Shell command to execute
        working_directory: Directory to run the command in (default: current)
        timeout: Command timeout in seconds (default: 30)

    Returns:
        Process execution result with stdout, stderr, and exit code
    """
    try:
        start_time = datetime.now()
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return ProcessResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time=execution_time,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout} seconds")
    except Exception as e:
        raise RuntimeError(f"Failed to execute command: {str(e)}")


@server.tool()
def git_status(repo_path: str = ".") -> GitStatus:
    """
    Get the current Git status of a repository.

    Args:
        repo_path: Path to the Git repository (default: current directory)

    Returns:
        GitStatus with branch name, staged/modified/untracked files
    """
    try:
        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        branch = (
            branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        )

        # Get status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        staged = []
        modified = []
        untracked = []

        if status_result.returncode == 0:
            for line in status_result.stdout.strip().split("\n"):
                if not line:
                    continue
                status_code = line[:2]
                filename = line[3:]

                if status_code[0] in ["A", "M", "D", "R", "C"]:
                    staged.append(filename)
                if status_code[1] in ["M", "D"]:
                    modified.append(filename)
                if status_code == "??":
                    untracked.append(filename)

        # Get ahead/behind info
        ahead, behind = 0, 0
        try:
            ahead_behind_result = subprocess.run(
                [
                    "git",
                    "rev-list",
                    "--left-right",
                    "--count",
                    f"{branch}...origin/{branch}",
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            if ahead_behind_result.returncode == 0:
                parts = ahead_behind_result.stdout.strip().split("\t")
                ahead = int(parts[0]) if len(parts) > 0 else 0
                behind = int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            pass

        return GitStatus(
            branch=branch,
            staged=staged,
            modified=modified,
            untracked=untracked,
            ahead=ahead,
            behind=behind,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get Git status: {str(e)}")


@server.tool()
def search_code(
    pattern: str,
    directory: str = ".",
    file_extensions: Optional[List[str]] = None,
    max_results: int = 50,
) -> Dict[str, List[str]]:
    """
    Search for a pattern in code files.

    Args:
        pattern: Text pattern to search for (case-insensitive)
        directory: Directory to search in (default: current)
        file_extensions: List of file extensions to search (default: common code files)
        max_results: Maximum number of matches to return (default: 50)

    Returns:
        Dictionary mapping file paths to lists of matching lines
    """
    try:
        if file_extensions is None:
            file_extensions = [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".rb",
                ".go",
                ".rs",
                ".md",
                ".txt",
                ".yaml",
                ".yml",
            ]

        search_path = Path(directory).expanduser().resolve()
        if not search_path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")

        results: Dict[str, List[str]] = {}
        total_matches = 0

        for ext in file_extensions:
            if total_matches >= max_results:
                break

            for file_path in search_path.rglob(f"*{ext}"):
                if total_matches >= max_results:
                    break

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        matches = []
                        for line_num, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                matches.append(f"{line_num}: {line.strip()}")
                                total_matches += 1
                                if total_matches >= max_results:
                                    break

                        if matches:
                            relative_path = str(file_path.relative_to(search_path))
                            results[relative_path] = matches
                except Exception:
                    continue

        return results
    except Exception as e:
        raise RuntimeError(f"Failed to search code: {str(e)}")


@server.tool()
def get_system_info() -> Dict[str, Any]:
    """
    Get system information including Python version, OS, and environment.

    Returns:
        Dictionary containing system information
    """
    try:
        import platform

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "current_directory": str(Path.cwd()),
            "environment_variables": {
                k: v
                for k, v in os.environ.items()
                if not any(
                    secret in k.lower()
                    for secret in ["key", "token", "password", "secret"]
                )
            },
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get system info: {str(e)}")


def main():
    """Main entry point for the server."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    print(f"Starting Development Assistant MCP Server on {transport}", file=sys.stderr)
    print("Server has 7 tools available:", file=sys.stderr)
    print("  - list_files", file=sys.stderr)
    print("  - read_file", file=sys.stderr)
    print("  - write_file", file=sys.stderr)
    print("  - run_command", file=sys.stderr)
    print("  - git_status", file=sys.stderr)
    print("  - search_code", file=sys.stderr)
    print("  - get_system_info", file=sys.stderr)

    server.run(transport=transport)


if __name__ == "__main__":
    main()
