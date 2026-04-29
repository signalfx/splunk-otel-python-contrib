#!/usr/bin/env python3
"""
MCP Development Assistant Server

A sample MCP server with useful development tools (list_files, read_file,
write_file, run_command, git_status, search_code, get_system_info) that
demonstrates OpenTelemetry instrumentation with the Splunk Distro.

Supports both transport modes:
  • stdio  — server is spawned as a sub-process; client communicates via
             stdin/stdout pipes.  Use this with ``dev_assistant_client.py``
             (default, no extra flags needed).
  • HTTP   — server listens on a TCP port using Streamable-HTTP; client
             connects over the network.  Pass ``--http`` (and optionally
             ``--port``/``--host``).

Usage — stdio (default):
    # Terminal 1 — start client (it spawns the server automatically):
    source .env
    OTEL_SERVICE_NAME=dev-assistant-client \\
        python dev_assistant_client.py

Usage — HTTP:
    # Terminal 1 — start the HTTP server:
    source .env
    OTEL_SERVICE_NAME=dev-assistant-server \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
        python dev_assistant_server.py --http --port 8001

    # Terminal 2 — run the client:
    source .env
    OTEL_SERVICE_NAME=dev-assistant-client \\
        python dev_assistant_client.py --http --server-url http://localhost:8001/mcp

Usage — zero-code instrumentation (HTTP):
    source .env
    OTEL_SERVICE_NAME=dev-assistant-server \\
    OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \\
        opentelemetry-instrument python dev_assistant_server.py --http

Environment Variables:
    OTEL_SERVICE_NAME                       Service name reported in Splunk
    OTEL_EXPORTER_OTLP_ENDPOINT            OTLP gRPC endpoint (e.g. http://localhost:4317)
    OTEL_EXPORTER_OTLP_HEADERS             Auth headers (e.g. X-SF-Token=<token>)
    OTEL_INSTRUMENTATION_GENAI_EMITTERS    span | span_metric | span_metric_event (default: span)
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT  true/false (default: false)
    MCP_HOST                                Bind host for HTTP mode (default: 0.0.0.0)
    MCP_PORT                                Bind port for HTTP mode (default: 8001)

NOTE: stdio servers MUST NOT write to stdout — it is reserved for the MCP
      wire protocol.  All diagnostic output goes to stderr.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from _otel_helpers import load_dotenv, providers_already_configured

# Load .env before importing OTel or FastMCP so env vars are available.
load_dotenv()


def setup_telemetry() -> None:
    """Configure OpenTelemetry SDK and instrument FastMCP.

    Called automatically when OTEL_EXPORTER_OTLP_ENDPOINT is set and the
    process was *not* already configured by ``opentelemetry-instrument``.

    For stdio servers, ConsoleSpanExporter is intentionally skipped because it
    writes to stdout which would corrupt the MCP protocol.
    """
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.environ.get("OTEL_SERVICE_NAME", "dev-assistant-server")
    resource = Resource.create({"service.name": service_name})

    trace_provider = TracerProvider(resource=resource)
    meter_readers: list = []

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            meter_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            print(f"[server] OTLP exporter → {otlp_endpoint}", file=sys.stderr)
        except ImportError:
            print("[server] OTLP exporter unavailable — install opentelemetry-exporter-otlp-proto-grpc", file=sys.stderr)
    else:
        print("[server] No OTLP endpoint set — spans created but not exported", file=sys.stderr)

    trace.set_tracer_provider(trace_provider)
    if meter_readers:
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=meter_readers))

    # Instrument AFTER providers are registered.
    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()
    print("[server] FastMCP instrumentation applied", file=sys.stderr)


# Apply telemetry:
#   • If opentelemetry-instrument already configured the SDK, just instrument.
#   • If OTLP endpoint is set in .env / env, run full setup.
#   • Otherwise skip — no-op providers will be used.
if providers_already_configured():
    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor
    FastMCPInstrumentor().instrument()
elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    setup_telemetry()
else:
    print("[server] No OTel provider configured — skipping telemetry setup.", file=sys.stderr)


# ---------------------------------------------------------------------------
# FastMCP server definition
# ---------------------------------------------------------------------------
from fastmcp import FastMCP  # noqa: E402

server = FastMCP("dev-assistant")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402


class FileInfo(BaseModel):
    name: str
    size: int
    modified: str
    is_directory: bool


class GitStatus(BaseModel):
    branch: str
    staged: List[str]
    modified: List[str]
    untracked: List[str]
    ahead: int
    behind: int


class ProcessResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


@server.tool()
def list_files(directory: str = ".") -> List[FileInfo]:
    """List files and directories in the specified path."""
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
        raise RuntimeError(f"Failed to list directory: {e}")


@server.tool()
def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """Read the contents of a file."""
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        if not path.is_file():
            raise ValueError(f"{file_path} is not a file")
        with open(path, encoding="utf-8", errors="replace") as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"... (truncated at {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines)
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")


@server.tool()
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path} ({path.stat().st_size} bytes)"
    except Exception as e:
        raise RuntimeError(f"Failed to write file: {e}")


@server.tool()
def run_command(command: str, working_directory: str = ".", timeout: int = 30) -> ProcessResult:
    """Execute a shell command and return the result."""
    try:
        start = datetime.now()
        result = subprocess.run(
            command, shell=True, cwd=working_directory,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = (datetime.now() - start).total_seconds()
        return ProcessResult(
            command=command, exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr, execution_time=elapsed,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout} seconds")
    except Exception as e:
        raise RuntimeError(f"Failed to execute command: {e}")


@server.tool()
def git_status(repo_path: str = ".") -> GitStatus:
    """Get the current Git status of a repository."""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=repo_path,
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"

        status_out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_path,
            capture_output=True, text=True,
        ).stdout

        staged, modified, untracked = [], [], []
        for line in status_out.strip().splitlines():
            if not line:
                continue
            code, filename = line[:2], line[3:]
            if code[0] in "AMDRС":
                staged.append(filename)
            if code[1] in "MD":
                modified.append(filename)
            if code == "??":
                untracked.append(filename)

        ahead, behind = 0, 0
        try:
            ab = subprocess.run(
                ["git", "rev-list", "--left-right", "--count", f"{branch}...origin/{branch}"],
                cwd=repo_path, capture_output=True, text=True,
            ).stdout.strip().split("\t")
            ahead, behind = int(ab[0]), int(ab[1])
        except Exception:
            pass

        return GitStatus(branch=branch, staged=staged, modified=modified,
                         untracked=untracked, ahead=ahead, behind=behind)
    except Exception as e:
        raise RuntimeError(f"Failed to get Git status: {e}")


@server.tool()
def search_code(
    pattern: str,
    directory: str = ".",
    file_extensions: Optional[List[str]] = None,
    max_results: int = 50,
) -> Dict[str, List[str]]:
    """Search for a pattern in code files."""
    if file_extensions is None:
        file_extensions = [".py", ".js", ".ts", ".java", ".go", ".rs", ".md", ".yaml", ".yml"]
    search_path = Path(directory).expanduser().resolve()
    if not search_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    results: Dict[str, List[str]] = {}
    total = 0
    for ext in file_extensions:
        if total >= max_results:
            break
        for fp in search_path.rglob(f"*{ext}"):
            if total >= max_results:
                break
            try:
                matches = []
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    for n, line in enumerate(f, 1):
                        if pattern.lower() in line.lower():
                            matches.append(f"{n}: {line.strip()}")
                            total += 1
                            if total >= max_results:
                                break
                if matches:
                    results[str(fp.relative_to(search_path))] = matches
            except Exception:
                continue
    return results


@server.tool()
def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "current_directory": str(Path.cwd()),
        "environment_variables": {
            k: v for k, v in os.environ.items()
            if not any(s in k.lower() for s in ("key", "token", "password", "secret"))
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Development Assistant Server")
    parser.add_argument("--http", action="store_true",
                        help="Use Streamable-HTTP transport instead of stdio")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "0.0.0.0"),
                        help="Bind host (HTTP mode only, default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8001")),
                        help="Bind port (HTTP mode only, default: 8001)")
    args = parser.parse_args()

    if args.http:
        print(f"[server] Starting HTTP server on {args.host}:{args.port}", file=sys.stderr)
        print(f"[server] MCP endpoint: http://{args.host}:{args.port}/mcp", file=sys.stderr)
        print(f"[server] Service name: {os.environ.get('OTEL_SERVICE_NAME', 'dev-assistant-server')}", file=sys.stderr)
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        print("[server] Starting stdio server (spawned as sub-process)", file=sys.stderr)
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
