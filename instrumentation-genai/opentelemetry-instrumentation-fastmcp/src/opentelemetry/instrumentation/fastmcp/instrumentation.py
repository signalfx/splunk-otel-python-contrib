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

"""FastMCP instrumentation for OpenTelemetry.

This module provides automatic instrumentation for FastMCP,
a Python library for building Model Context Protocol (MCP) servers and clients.
"""

import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.instrumentation.fastmcp.server_instrumentor import (
    ServerInstrumentor,
)
from opentelemetry.instrumentation.fastmcp.client_instrumentor import (
    ClientInstrumentor,
)
from opentelemetry.instrumentation.fastmcp.transport_instrumentor import (
    TransportInstrumentor,
)
from opentelemetry.instrumentation.fastmcp.utils import is_instrumentation_enabled

_LOGGER = logging.getLogger(__name__)

_instruments = ("fastmcp >= 2.0.0",)


class FastMCPInstrumentor(BaseInstrumentor):
    """An instrumentor for FastMCP library.

    This instrumentor provides telemetry for:
    - Server-side tool execution
    - Client-side session management
    - Tool calls and listings

    The instrumentation integrates with opentelemetry-util-genai to emit
    spans, metrics, and events following the Splunk Distro patterns.

    Usage:
        from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

        FastMCPInstrumentor().instrument()

    Environment Variables:
        OTEL_INSTRUMENTATION_GENAI_ENABLE: Enable/disable instrumentation (default: true)
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: Capture tool I/O (default: false)
        OTEL_INSTRUMENTATION_GENAI_EMITTERS: Select emitters (default: span)
    """

    def __init__(self, exception_logger=None):
        """Initialize the FastMCP instrumentor.

        Args:
            exception_logger: Optional custom exception logger
        """
        super().__init__()
        self._exception_logger = exception_logger
        self._telemetry_handler = None
        self._server_instrumentor = None
        self._client_instrumentor = None
        self._transport_instrumentor = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the dependencies required for this instrumentation."""
        return _instruments

    def _instrument(self, **kwargs):
        """Apply FastMCP instrumentation.

        Args:
            **kwargs: Instrumentation options including:
                - tracer_provider: Optional tracer provider
                - meter_provider: Optional meter provider
                - logger_provider: Optional logger provider
        """
        if not is_instrumentation_enabled():
            _LOGGER.debug("FastMCP instrumentation is disabled")
            return

        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        # Get the telemetry handler from util-genai
        self._telemetry_handler = get_telemetry_handler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Initialize and apply transport-level instrumentation for trace propagation
        # This MUST be applied first to ensure context is available for other instrumentors
        self._transport_instrumentor = TransportInstrumentor()
        self._transport_instrumentor.instrument()

        # Initialize and apply server-side instrumentation
        self._server_instrumentor = ServerInstrumentor(self._telemetry_handler)
        self._server_instrumentor.instrument()

        # Initialize and apply client-side instrumentation
        self._client_instrumentor = ClientInstrumentor(self._telemetry_handler)
        self._client_instrumentor.instrument()

        _LOGGER.debug("FastMCP instrumentation applied (with trace propagation)")

    def _uninstrument(self, **kwargs):
        """Remove FastMCP instrumentation.

        Note: Due to wrapt limitations with post-import hooks,
        complete uninstrumentation may not be possible.
        """
        if self._transport_instrumentor:
            self._transport_instrumentor.uninstrument()

        if self._server_instrumentor:
            self._server_instrumentor.uninstrument()

        if self._client_instrumentor:
            self._client_instrumentor.uninstrument()

        _LOGGER.debug("FastMCP instrumentation removed")
