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
MCP Transport layer instrumentation for trace context propagation.

This module instruments the low-level MCP SDK transport to ensure trace context
(traceparent, tracestate) and baggage are propagated between client and server.

Approach:
- Client side: Wrap BaseSession.send_request to inject trace context into _meta
- Server side: Wrap Server._handle_request to extract from request_meta and
  populate MCPRequestContext for downstream instrumentors.
"""

import logging
from typing import Any

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry import context, propagate

from opentelemetry.instrumentation.fastmcp._mcp_context import (
    MCPRequestContext,
    clear_mcp_request_context,
    set_mcp_request_context,
)
from opentelemetry.instrumentation.fastmcp.utils import detect_transport

_LOGGER = logging.getLogger(__name__)


def _extract_carrier_from_meta(request_meta: Any) -> dict[str, str]:
    """Build a W3C carrier dict from a Pydantic Meta object.

    Checks both first-class attributes and ``model_extra`` so that
    traceparent, tracestate, and baggage are all captured.
    """
    carrier: dict[str, str] = {}
    if request_meta is None:
        return carrier

    for key in ("traceparent", "tracestate", "baggage"):
        val = getattr(request_meta, key, None)
        if val:
            carrier[key] = val

    if hasattr(request_meta, "model_extra"):
        extra = request_meta.model_extra or {}
        for key in ("traceparent", "tracestate", "baggage"):
            if key not in carrier and key in extra:
                carrier[key] = extra[key]

    return carrier


class TransportInstrumentor:
    """Instruments MCP transport layer for trace context propagation.

    Handles low-level MCP SDK to ensure traces are properly correlated
    across client/server process boundaries.
    """

    def __init__(self):
        self._instrumented = False

    def instrument(self):
        """Apply MCP transport instrumentation."""
        if self._instrumented:
            return

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_request",
                self._send_request_wrapper(),
            ),
            "mcp.shared.session",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.lowlevel",
                "Server._handle_request",
                self._server_handle_request_wrapper(),
            ),
            "mcp.server.lowlevel",
        )

        self._instrumented = True
        _LOGGER.debug("MCP transport instrumentation applied")

    def uninstrument(self):
        """Remove MCP transport instrumentation.

        Note: wrapt doesn't provide a clean way to unwrap post-import hooks.
        """
        self._instrumented = False

    def _send_request_wrapper(self):
        """Wrapper for BaseSession.send_request to inject trace context.

        Runs on the client side before sending any MCP request.  Injects
        traceparent, tracestate, and baggage into params.meta.
        """

        async def traced_send_request(wrapped, instance, args, kwargs) -> Any:
            try:
                request = args[0] if args else kwargs.get("request")

                actual_request = request
                if hasattr(request, "root"):
                    actual_request = request.root

                if (
                    actual_request
                    and hasattr(actual_request, "params")
                    and actual_request.params
                ):
                    params = actual_request.params

                    if hasattr(params, "meta"):
                        if params.meta is None:
                            try:
                                from mcp.types import RequestParams

                                params.meta = RequestParams.Meta()
                            except Exception:
                                pass

                        if params.meta is not None:
                            carrier: dict[str, str] = {}
                            propagate.inject(carrier)

                            if carrier:
                                for key, value in carrier.items():
                                    setattr(params.meta, key, value)

                                method = getattr(actual_request, "method", "unknown")
                                _LOGGER.debug(
                                    "Injected trace context into MCP request: "
                                    "%s, carrier=%s",
                                    method,
                                    carrier,
                                )
            except Exception as e:
                _LOGGER.debug("Error injecting trace context: %s", e, exc_info=True)

            return await wrapped(*args, **kwargs)

        return traced_send_request

    def _server_handle_request_wrapper(self):
        """Wrapper for Server._handle_request to extract trace context.

        Runs on the server side.  Extracts W3C context from ``request_meta``
        and populates :class:`MCPRequestContext` for the server instrumentor.

        Method signature:
            _handle_request(self, message, req, session, lifespan_context,
                            raise_exceptions)
        """

        async def traced_handle_request(wrapped, instance, args, kwargs) -> Any:
            token = None
            try:
                message = args[0] if args else kwargs.get("message")
                req = args[1] if len(args) > 1 else kwargs.get("req")

                carrier: dict[str, str] = {}
                jsonrpc_id: str | None = None
                method_name: str | None = None
                transport = detect_transport(instance)

                if message and hasattr(message, "request_meta"):
                    carrier = _extract_carrier_from_meta(message.request_meta)

                if message and hasattr(message, "request_id"):
                    raw_id = message.request_id
                    if raw_id is not None:
                        jsonrpc_id = str(raw_id)

                if req is not None:
                    method_name = getattr(req, "method", None)

                if carrier:
                    ctx = propagate.extract(carrier)
                    token = context.attach(ctx)
                    _LOGGER.debug(
                        "Attached trace context in _handle_request: carrier=%s",
                        carrier,
                    )

                mcp_ctx = MCPRequestContext(
                    jsonrpc_request_id=jsonrpc_id,
                    mcp_method_name=method_name,
                    network_transport=transport,
                )
                set_mcp_request_context(mcp_ctx)

            except Exception as e:
                _LOGGER.debug("Error extracting trace context: %s", e, exc_info=True)

            try:
                return await wrapped(*args, **kwargs)
            finally:
                clear_mcp_request_context()
                if token is not None:
                    try:
                        context.detach(token)
                    except Exception:
                        pass

        return traced_handle_request

    # kept for backward compatibility with existing tests
    def _server_received_request_wrapper(self):
        """Legacy wrapper - replaced by _server_handle_request_wrapper."""
        return self._server_handle_request_wrapper()
