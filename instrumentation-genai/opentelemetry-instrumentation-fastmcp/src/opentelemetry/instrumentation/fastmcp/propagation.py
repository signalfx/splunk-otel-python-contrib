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

"""
Trace context propagation for MCP protocol.

This module handles injecting trace context into outgoing MCP requests (client-side)
and extracting trace context from incoming MCP requests (server-side).

The trace context is propagated via the `_meta` field in MCP request parameters,
following the W3C TraceContext format (traceparent, tracestate).
"""

import logging
from typing import Any, Optional

from opentelemetry import context, propagate
from opentelemetry.context import Context

_LOGGER = logging.getLogger(__name__)


def inject_trace_context(params: Any) -> Any:
    """Inject trace context into MCP request parameters.

    Injects the current trace context (traceparent, tracestate) into the
    `_meta` field of the request parameters.

    Args:
        params: MCP request parameters (can be dict or object with __dict__)

    Returns:
        The modified params with trace context injected into _meta
    """
    try:
        if params is None:
            return params

        # Handle dict-like params
        if isinstance(params, dict):
            meta = params.setdefault("_meta", {})
            propagate.inject(meta)
            return params

        # Handle object with __dict__ (e.g., dataclass, pydantic model)
        if hasattr(params, "__dict__"):
            params_dict = params.__dict__
            if "_meta" not in params_dict:
                params_dict["_meta"] = {}
            meta = params_dict["_meta"]
            if isinstance(meta, dict):
                propagate.inject(meta)

        return params
    except Exception as e:
        _LOGGER.debug(f"Failed to inject trace context: {e}")
        return params


def extract_trace_context(params: Any) -> Optional[Context]:
    """Extract trace context from MCP request parameters.

    Extracts trace context (traceparent, tracestate) from the `_meta` field
    of the request parameters.

    Args:
        params: MCP request parameters (can be dict or object with attributes)

    Returns:
        Extracted context, or None if no trace context found
    """
    try:
        if params is None:
            return None

        meta = None

        # Handle dict-like params
        if isinstance(params, dict):
            meta = params.get("_meta")
        # Handle object with _meta attribute
        elif hasattr(params, "_meta"):
            meta = params._meta
        # Handle object with __dict__
        elif hasattr(params, "__dict__") and "_meta" in params.__dict__:
            meta = params.__dict__["_meta"]
        # Handle object with get method (Mapping-like)
        elif hasattr(params, "get"):
            meta = params.get("_meta")

        if meta is None:
            return None

        # Extract trace context from meta
        # meta could be a dict or an object with attributes
        if isinstance(meta, dict):
            ctx = propagate.extract(meta)
            return ctx
        elif hasattr(meta, "__dict__"):
            # Some MCP frameworks return meta as an object
            ctx = propagate.extract(meta.__dict__)
            return ctx

        return None
    except Exception as e:
        _LOGGER.debug(f"Failed to extract trace context: {e}")
        return None


class ContextManager:
    """Manages context attachment and detachment for trace propagation."""

    def __init__(self):
        self._token: Optional[object] = None

    def attach(self, ctx: Optional[Context]) -> bool:
        """Attach extracted context to the current context.

        Args:
            ctx: Context to attach (from extract_trace_context)

        Returns:
            True if context was attached, False otherwise
        """
        if ctx is None:
            return False

        try:
            self._token = context.attach(ctx)
            return True
        except Exception as e:
            _LOGGER.debug(f"Failed to attach context: {e}")
            return False

    def detach(self) -> None:
        """Detach previously attached context."""
        if self._token is not None:
            try:
                context.detach(self._token)
            except Exception as e:
                _LOGGER.debug(f"Failed to detach context: {e}")
            finally:
                self._token = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()
        return False
