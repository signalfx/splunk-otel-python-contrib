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

"""Tests verifying trace context + baggage propagation across MCP transports.

The MCP protocol supports multiple transports (stdio, SSE, streamable-http).
Trace context and baggage are injected/extracted at the MCP SDK session layer
(BaseSession.send_request / Server._handle_request), which is *above* the
transport layer. This means the same _meta-based propagation works identically
for all transports.

These tests verify the full inject→extract→restore round-trip using the actual
TransportInstrumentor wrappers with realistic pydantic-style meta objects that
mirror what each transport produces.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opentelemetry import baggage, context
from opentelemetry.instrumentation.fastmcp.transport_instrumentor import (
    TransportInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.genai.handler import (
    clear_session_context,
    get_session_context,
)


class _PydanticMeta:
    """Simulates a pydantic Meta object with extra='allow'.

    In real MCP, RequestParams.Meta is a pydantic model with extra='allow',
    meaning any attribute can be set on it (traceparent, tracestate, baggage).
    This class mimics that behavior for tests.
    """

    def __init__(self, **kwargs):
        self.model_extra = {}
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_client_request(meta=None):
    """Build a mock ClientRequest with root.params.meta."""
    mock_meta = meta if meta is not None else _PydanticMeta()
    mock_params = MagicMock()
    mock_params.meta = mock_meta
    mock_root = MagicMock()
    mock_root.method = "tools/call"
    mock_root.params = mock_params
    mock_request = MagicMock()
    mock_request.root = mock_root
    return mock_request, mock_meta


def _make_server_responder(meta_attrs: dict | None = None):
    """Build a mock server RequestResponder with request_meta."""
    if meta_attrs is None:
        meta = None
    else:
        meta = _PydanticMeta(**meta_attrs)
    mock_responder = MagicMock()
    mock_responder.request_meta = meta
    return mock_responder


class TestRoundTripPropagation:
    """Full inject→extract round-trip: client injects, server extracts."""

    def setup_method(self):
        clear_session_context()

    def teardown_method(self):
        clear_session_context()

    @pytest.mark.asyncio
    async def test_traceparent_round_trip(self):
        """traceparent injected by client is extracted by server."""
        # Need a tracer provider so propagate.inject() produces traceparent
        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        instrumentor = TransportInstrumentor()

        # --- Client side: inject within an active span ---
        with tracer.start_as_current_span("client-call"):
            request, meta = _make_client_request()
            client_wrapper = instrumentor._send_request_wrapper()
            await client_wrapper(
                AsyncMock(return_value="ok"), MagicMock(), (request,), {}
            )

        # Meta should have traceparent set
        assert hasattr(meta, "traceparent")
        traceparent = meta.traceparent
        assert traceparent is not None
        assert traceparent.startswith("00-")

        # --- Server side: extract ---
        responder = _make_server_responder({"traceparent": traceparent})
        server_wrapper = instrumentor._server_handle_request_wrapper()

        mock_wrapped = AsyncMock(return_value="ok")
        await server_wrapper(mock_wrapped, MagicMock(), (responder,), {})
        # If we got here without error, trace context was attached and detached

        provider.shutdown()

    @pytest.mark.asyncio
    async def test_baggage_round_trip(self):
        """Session baggage injected by client is extracted and restored on server."""
        instrumentor = TransportInstrumentor()

        # --- Client side: set baggage and inject ---
        ctx = baggage.set_baggage("gen_ai.conversation.id", "round-trip-sess")
        ctx = baggage.set_baggage("user.id", "round-trip-user", ctx)
        ctx = baggage.set_baggage("customer.id", "round-trip-cust", ctx)
        token = context.attach(ctx)

        try:
            request, meta = _make_client_request()
            client_wrapper = instrumentor._send_request_wrapper()
            await client_wrapper(
                AsyncMock(return_value="ok"), MagicMock(), (request,), {}
            )
        finally:
            context.detach(token)

        # Meta should have baggage header
        assert hasattr(meta, "baggage")
        assert "gen_ai.conversation.id=round-trip-sess" in meta.baggage
        assert "user.id=round-trip-user" in meta.baggage
        assert "customer.id=round-trip-cust" in meta.baggage

        # --- Server side: extract and restore ---
        carrier = {"baggage": meta.baggage}
        tp = getattr(meta, "traceparent", None)
        if tp:
            carrier["traceparent"] = tp
        responder = _make_server_responder(carrier)
        server_wrapper = instrumentor._server_handle_request_wrapper()

        # Enable baggage propagation for restore
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            mock_wrapped = AsyncMock(return_value="ok")
            await server_wrapper(mock_wrapped, MagicMock(), (responder,), {})

        # Session should be restored
        session = get_session_context()
        assert session.session_id == "round-trip-sess"
        assert session.user_id == "round-trip-user"
        assert session.customer_id == "round-trip-cust"


class TestTransportAgnostic:
    """Verify propagation is transport-agnostic: same meta format works for all.

    The MCP protocol carries _meta identically regardless of transport (stdio,
    SSE, streamable-http). The inject/extract hooks operate on pydantic Meta
    objects, not on the transport wire format. These tests confirm the meta
    format works with various transport-like scenarios.
    """

    def setup_method(self):
        clear_session_context()

    def teardown_method(self):
        clear_session_context()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_inject_works_for_all_transports(self, transport_label):
        """Client-side inject produces same meta format regardless of transport."""
        instrumentor = TransportInstrumentor()

        ctx = baggage.set_baggage("gen_ai.conversation.id", f"sess-{transport_label}")
        token = context.attach(ctx)

        try:
            request, meta = _make_client_request()
            wrapper = instrumentor._send_request_wrapper()
            await wrapper(AsyncMock(return_value="ok"), MagicMock(), (request,), {})
        finally:
            context.detach(token)

        # All transports produce the same meta attributes
        # Note: traceparent is only injected when there's an active span;
        # baggage is always injected when baggage entries exist
        assert hasattr(meta, "baggage")
        assert f"gen_ai.conversation.id=sess-{transport_label}" in meta.baggage

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_inject_with_span_for_all_transports(self, transport_label):
        """With active span, traceparent + baggage are injected for all transports."""
        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        instrumentor = TransportInstrumentor()

        ctx = baggage.set_baggage("gen_ai.conversation.id", f"sess-{transport_label}")
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("test-call"):
                request, meta = _make_client_request()
                wrapper = instrumentor._send_request_wrapper()
                await wrapper(AsyncMock(return_value="ok"), MagicMock(), (request,), {})
        finally:
            context.detach(token)

        assert hasattr(meta, "traceparent")
        assert meta.traceparent.startswith("00-")
        assert hasattr(meta, "baggage")
        assert f"gen_ai.conversation.id=sess-{transport_label}" in meta.baggage
        provider.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_extract_works_for_all_transports(self, transport_label):
        """Server-side extract restores session from any transport's meta."""
        instrumentor = TransportInstrumentor()

        responder = _make_server_responder(
            {
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
                "baggage": f"gen_ai.conversation.id=sess-{transport_label},user.id=user-{transport_label}",
            }
        )

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            wrapper = instrumentor._server_handle_request_wrapper()
            await wrapper(AsyncMock(return_value="ok"), MagicMock(), (responder,), {})

        session = get_session_context()
        assert session.session_id == f"sess-{transport_label}"
        assert session.user_id == f"user-{transport_label}"
        clear_session_context()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_model_extra_fallback_for_all_transports(self, transport_label):
        """Server-side extract falls back to model_extra for pydantic v2 extras."""
        instrumentor = TransportInstrumentor()

        # Some pydantic models store extra fields in model_extra instead of
        # as direct attributes
        meta = MagicMock(spec=[])
        meta.model_extra = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "baggage": f"gen_ai.conversation.id=extra-{transport_label}",
        }
        # Ensure no direct attributes (force model_extra path)
        del meta.traceparent

        mock_responder = MagicMock()
        mock_responder.request_meta = meta

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            wrapper = instrumentor._server_handle_request_wrapper()
            await wrapper(
                AsyncMock(return_value="ok"), MagicMock(), (mock_responder,), {}
            )

        session = get_session_context()
        assert session.session_id == f"extra-{transport_label}"
        clear_session_context()


class TestNoSessionLeakInMeta:
    """Verify session fields are ONLY propagated via baggage, never as direct meta attrs."""

    @pytest.mark.asyncio
    async def test_no_session_id_attribute_on_meta(self):
        """Meta should NOT have gen_ai.conversation.id/user.id/customer.id as direct attributes."""
        instrumentor = TransportInstrumentor()

        ctx = baggage.set_baggage("gen_ai.conversation.id", "test-sess")
        ctx = baggage.set_baggage("user.id", "test-user", ctx)
        ctx = baggage.set_baggage("customer.id", "test-cust", ctx)
        token = context.attach(ctx)

        try:
            request, meta = _make_client_request()
            wrapper = instrumentor._send_request_wrapper()
            await wrapper(AsyncMock(return_value="ok"), MagicMock(), (request,), {})
        finally:
            context.detach(token)

        # Session values should be in the baggage header, NOT as individual attributes
        assert not hasattr(meta, "gen_ai.conversation.id")
        assert not hasattr(meta, "session.id")
        assert not hasattr(meta, "user.id")
        assert not hasattr(meta, "customer.id")
        assert not hasattr(meta, "session_id")
        assert not hasattr(meta, "user_id")
        assert not hasattr(meta, "customer_id")

        # They should be in the baggage string
        assert hasattr(meta, "baggage")
        assert "gen_ai.conversation.id=test-sess" in meta.baggage
        assert "user.id=test-user" in meta.baggage
        assert "customer.id=test-cust" in meta.baggage

    @pytest.mark.asyncio
    async def test_meta_only_has_w3c_headers(self):
        """Meta should only contain W3C standard headers (traceparent, tracestate, baggage)."""
        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        instrumentor = TransportInstrumentor()

        ctx = baggage.set_baggage("gen_ai.conversation.id", "check-headers")
        token = context.attach(ctx)

        try:
            # Use a real dict to track exactly what was set
            set_attrs = {}

            class TrackingMeta:
                def __setattr__(self, name, value):
                    set_attrs[name] = value
                    super().__setattr__(name, value)

            meta = TrackingMeta()
            meta.model_extra = {}  # required for pydantic compatibility
            request, _ = _make_client_request(meta)

            with tracer.start_as_current_span("test-call"):
                wrapper = instrumentor._send_request_wrapper()
                await wrapper(AsyncMock(return_value="ok"), MagicMock(), (request,), {})
        finally:
            context.detach(token)

        # Only W3C standard headers should be set (plus model_extra from init)
        w3c_keys = {"traceparent", "tracestate", "baggage", "model_extra"}
        unexpected = set(set_attrs.keys()) - w3c_keys
        assert not unexpected, f"Unexpected meta attributes: {unexpected}"
        provider.shutdown()
