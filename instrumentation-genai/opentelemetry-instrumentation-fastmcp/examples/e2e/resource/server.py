#!/usr/bin/env python3
"""MCP System Dashboard Resource Server.

Exposes system information as MCP resources:
  - system://info           (static)   — OS / Python / hostname
  - system://uptime         (static)   — system uptime
  - system://env/{var_name} (template) — environment variable lookup

Usage:
    python server.py                       # stdio
    python server.py --sse --port 8002     # SSE
"""

import os
import platform
import time

from fastmcp import FastMCP

mcp = FastMCP("System Dashboard Server")

_START_TIME = time.time()


@mcp.resource("system://info")
def system_info() -> str:
    """Basic system information.

    Returns hostname, OS, architecture, and Python version.
    """
    return (
        f"hostname: {platform.node()}\n"
        f"os: {platform.system()} {platform.release()}\n"
        f"arch: {platform.machine()}\n"
        f"python: {platform.python_version()}\n"
        f"pid: {os.getpid()}"
    )


@mcp.resource("system://uptime")
def system_uptime() -> str:
    """Server process uptime since start.

    Returns uptime in seconds and a human-readable format.
    """
    elapsed = time.time() - _START_TIME
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"uptime_seconds: {elapsed:.1f}\nuptime_human: {hours}h {minutes}m {seconds}s"
    )


@mcp.resource("system://env/{var_name}")
def env_variable(var_name: str) -> str:
    """Read an environment variable by name.

    Args:
        var_name: Name of the environment variable to read.
    """
    value = os.environ.get(var_name)
    if value is None:
        return f"{var_name} is not set"
    return f"{var_name}={value}"


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MCP System Dashboard Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode")
    parser.add_argument("--port", type=int, default=8002, help="SSE port")
    args = parser.parse_args()

    if args.sse:
        print(
            f"Starting SSE server at http://localhost:{args.port}/sse",
            file=sys.stderr,
        )
        mcp.run(transport="sse", host="localhost", port=args.port)
    else:
        mcp.run()
