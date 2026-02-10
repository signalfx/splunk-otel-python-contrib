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
MCP Transport layer instrumentation for trace context and session propagation.

This module instruments the low-level MCP SDK transport to ensure trace context
(traceparent, tracestate) and session baggage are propagated between client and
server processes using the standard OTel Propagation API.

Approach:
- Client side: Wrap BaseSession.send_request to inject context via propagate.inject()
- Server side: Wrap Server._handle_request to extract context via propagate.extract()
  and restore session context from baggage
"""

import logging
from typing import Any

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry import context, propagate

_LOGGER = logging.getLogger(__name__)


class TransportInstrumentor:
    """Instruments MCP transport layer for trace context propagation.

    This handles the low-level MCP SDK to ensure traces are properly correlated
    across client/server process boundaries.
    """

    def __init__(self):
        self._instrumented = False

    def instrument(self):
        """Apply MCP transport instrumentation."""
        if self._instrumented:
            return

        # Wrap client-side request sending to inject trace context
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_request",
                self._send_request_wrapper(),
            ),
            "mcp.shared.session",
        )

        # Wrap server-side request handling to extract trace context
        # Use Server._handle_request which is invoked for each request
        # and has access to message.request_meta with the traceparent
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
        This is a known limitation.
        """
        self._instrumented = False

    def _send_request_wrapper(self):
        """Wrapper for BaseSession.send_request to inject trace context and baggage.

        This runs on the client side before sending any MCP request.
        Injects traceparent/tracestate and baggage into the request's params.meta field.

        The baggage header carries session context (gen_ai.conversation.id, user.id, customer.id)
        when baggage propagation is enabled via OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION=baggage.

        The MCP SDK request structure:
        - ClientRequest is a discriminated union with a 'root' attribute
        - request.root contains the actual request (CallToolRequest, etc.)
        - request.root.params.meta (aliased as _meta in JSON) is a Meta object
        - Meta has extra='allow', so we can add traceparent/tracestate/baggage
        """

        async def traced_send_request(wrapped, instance, args, kwargs) -> Any:
            try:
                # args[0] is the request wrapper (ClientRequest)
                request = args[0] if args else kwargs.get("request")

                # Handle discriminated union pattern: ClientRequest has 'root'
                actual_request = request
                if hasattr(request, "root"):
                    actual_request = request.root

                if (
                    actual_request
                    and hasattr(actual_request, "params")
                    and actual_request.params
                ):
                    params = actual_request.params

                    # Create or get the meta object
                    # In pydantic models, 'meta' is the Python attribute,
                    # '_meta' is the JSON alias
                    if hasattr(params, "meta"):
                        if params.meta is None:
                            # Create a new Meta object
                            try:
                                from mcp.types import RequestParams

                                params.meta = RequestParams.Meta()
                            except Exception:
                                pass

                        if params.meta is not None:
                            # Inject trace context into meta
                            # Meta allows extra fields, so we can set
                            # traceparent and tracestate directly
                            carrier = {}
                            propagate.inject(carrier)

                            if carrier:
                                for key, value in carrier.items():
                                    setattr(params.meta, key, value)

                                method = getattr(actual_request, "method", "unknown")
                                _LOGGER.debug(
                                    f"Injected trace context into MCP request: "
                                    f"{method}, carrier={carrier}"
                                )
            except Exception as e:
                _LOGGER.debug(f"Error injecting trace context: {e}", exc_info=True)

            # Call original method
            return await wrapped(*args, **kwargs)

        return traced_send_request

    def _server_handle_request_wrapper(self):
        """Wrapper for Server._handle_request to extract trace context and session.

        This runs on the server side when handling an MCP request.
        Extracts trace context (traceparent/tracestate) and baggage from
        the request metadata, then restores session context from baggage
        for GenAI instrumentation.

        The method signature is:
            _handle_request(self, message, req, session, lifespan_context, raise_exceptions)

        message: RequestResponder with request_meta containing traceparent
        req: The actual request (CallToolRequest, etc.)
        """

        async def traced_handle_request(wrapped, instance, args, kwargs) -> Any:
            token = None
            try:
                # args[0] is the message (RequestResponder)
                message = args[0] if args else kwargs.get("message")

                if message and hasattr(message, "request_meta"):
                    request_meta = message.request_meta

                    if request_meta is not None:
                        # Extract trace context and baggage from request_meta
                        # The meta object may have traceparent/tracestate/baggage as attributes
                        carrier = {}

                        # Try to get traceparent, tracestate, and baggage from meta
                        # First check as attribute (getattr handles pydantic properly)
                        for key in ("traceparent", "tracestate", "baggage"):
                            value = getattr(request_meta, key, None)
                            if value:
                                carrier[key] = value

                        # Also try model_extra for pydantic v2 extra fields
                        if not carrier and hasattr(request_meta, "model_extra"):
                            extra = request_meta.model_extra
                            if extra:
                                for key in ("traceparent", "tracestate", "baggage"):
                                    if key in extra:
                                        carrier[key] = extra[key]

                        if carrier:
                            ctx = propagate.extract(carrier)
                            token = context.attach(ctx)
                            _LOGGER.debug(
                                f"Attached trace context in _handle_request: "
                                f"carrier={carrier}"
                            )

                            # Restore session context from baggage if enabled
                            try:
                                from .propagation import restore_session_from_context

                                restore_session_from_context(ctx)
                            except Exception as e:
                                _LOGGER.debug(
                                    f"Failed to restore session from baggage: {e}"
                                )

            except Exception as e:
                _LOGGER.debug(f"Error extracting trace context: {e}", exc_info=True)

            try:
                # Call original method with attached context
                return await wrapped(*args, **kwargs)
            finally:
                # Detach context after request handling
                if token is not None:
                    try:
                        context.detach(token)
                    except Exception:
                        pass

        return traced_handle_request
