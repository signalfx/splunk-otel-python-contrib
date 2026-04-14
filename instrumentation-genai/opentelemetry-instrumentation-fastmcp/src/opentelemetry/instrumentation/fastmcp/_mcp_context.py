# Copyright The OpenTelemetry Authors
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

"""ContextVar bridge for passing MCP transport metadata to instrumentor layers.

The transport instrumentor populates this context on the server side so that
the server instrumentor can read transport-level attributes (jsonrpc.request.id,
network.transport, etc.) without coupling directly to transport internals.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass
class MCPRequestContext:
    """Transport-layer metadata extracted from an incoming MCP request."""

    jsonrpc_request_id: Optional[str] = None
    mcp_method_name: Optional[str] = None
    network_transport: Optional[str] = None


_mcp_request_context: ContextVar[Optional[MCPRequestContext]] = ContextVar(
    "mcp_request_context", default=None
)


def set_mcp_request_context(ctx: MCPRequestContext) -> None:
    _mcp_request_context.set(ctx)


def get_mcp_request_context() -> Optional[MCPRequestContext]:
    return _mcp_request_context.get()


def clear_mcp_request_context() -> None:
    _mcp_request_context.set(None)
