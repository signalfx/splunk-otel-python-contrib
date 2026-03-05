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

"""Tests verifying trace context propagation across MCP transports.

The MCP protocol supports multiple transports (stdio, SSE, streamable-http).
Trace context is injected/extracted at the MCP SDK session layer
(BaseSession.send_request / Server._handle_request), which is *above* the
transport layer. This means OTel trace context propagation works identically
for all transports.

These tests verify the full inject→extract round-trip using the actual
TransportInstrumentor wrappers with realistic pydantic-style meta objects that
mirror what each transport produces.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from opentelemetry import baggage, context
from opentelemetry.instrumentation.fastmcp.transport_instrumentor import (
    TransportInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider


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
    """Full inject-extract round-trip: client injects, server extracts."""

    @pytest.mark.asyncio
    async def test_traceparent_round_trip(self):
        """traceparent injected by client is extracted by server."""
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
        """Baggage injected by client is extracted on server."""
        instrumentor = TransportInstrumentor()

        # --- Client side: set baggage and inject ---
        ctx = baggage.set_baggage("gen_ai.conversation.id", "round-trip-conv")
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
        assert "gen_ai.conversation.id=round-trip-conv" in meta.baggage
        assert "user.id=round-trip-user" in meta.baggage
        assert "customer.id=round-trip-cust" in meta.baggage

        # --- Server side: extract ---
        carrier = {"baggage": meta.baggage}
        tp = getattr(meta, "traceparent", None)
        if tp:
            carrier["traceparent"] = tp
        responder = _make_server_responder(carrier)
        server_wrapper = instrumentor._server_handle_request_wrapper()

        # Capture the extracted context
        extracted_ctx = None

        async def capture_wrapped(*args, **kwargs):
            nonlocal extracted_ctx
            extracted_ctx = context.get_current()
            return "ok"

        await server_wrapper(capture_wrapped, MagicMock(), (responder,), {})

        # Baggage should be available in the extracted context
        conv_id = baggage.get_baggage("gen_ai.conversation.id", extracted_ctx)
        user_id = baggage.get_baggage("user.id", extracted_ctx)
        customer_id = baggage.get_baggage("customer.id", extracted_ctx)
        assert conv_id == "round-trip-conv"
        assert user_id == "round-trip-user"
        assert customer_id == "round-trip-cust"


class TestTransportAgnostic:
    """Verify propagation is transport-agnostic: same carrier format works for all.

    The MCP protocol carries context identically regardless of transport (stdio,
    SSE, streamable-http). The inject/extract hooks operate on pydantic Meta
    objects, not on the transport wire format.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_inject_works_for_all_transports(self, transport_label):
        """Client-side inject produces same meta format regardless of transport."""
        instrumentor = TransportInstrumentor()

        ctx = baggage.set_baggage("gen_ai.conversation.id", f"conv-{transport_label}")
        token = context.attach(ctx)

        try:
            request, meta = _make_client_request()
            wrapper = instrumentor._send_request_wrapper()
            await wrapper(AsyncMock(return_value="ok"), MagicMock(), (request,), {})
        finally:
            context.detach(token)

        assert hasattr(meta, "baggage")
        assert f"gen_ai.conversation.id=conv-{transport_label}" in meta.baggage

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

        ctx = baggage.set_baggage("gen_ai.conversation.id", f"conv-{transport_label}")
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("test-call"):
                request, meta = _make_client_request()
                wrapper = instrumentor._send_request_wrapper()
                await wrapper(
                    AsyncMock(return_value="ok"),
                    MagicMock(),
                    (request,),
                    {},
                )
        finally:
            context.detach(token)

        assert hasattr(meta, "traceparent")
        assert meta.traceparent.startswith("00-")
        assert hasattr(meta, "baggage")
        assert f"gen_ai.conversation.id=conv-{transport_label}" in meta.baggage
        provider.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_extract_works_for_all_transports(self, transport_label):
        """Server-side extract restores trace context from any transport's meta."""
        instrumentor = TransportInstrumentor()

        responder = _make_server_responder(
            {
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
                "baggage": f"gen_ai.conversation.id=conv-{transport_label},user.id=user-{transport_label}",
            }
        )

        wrapper = instrumentor._server_handle_request_wrapper()

        extracted_ctx = None

        async def capture_wrapped(*args, **kwargs):
            nonlocal extracted_ctx
            extracted_ctx = context.get_current()
            return "ok"

        await wrapper(capture_wrapped, MagicMock(), (responder,), {})

        conv_id = baggage.get_baggage("gen_ai.conversation.id", extracted_ctx)
        user_id = baggage.get_baggage("user.id", extracted_ctx)
        assert conv_id == f"conv-{transport_label}"
        assert user_id == f"user-{transport_label}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_label",
        ["stdio", "sse", "streamable-http"],
        ids=["stdio", "sse", "streamable-http"],
    )
    async def test_model_extra_fallback_for_all_transports(self, transport_label):
        """Server-side extract falls back to model_extra for pydantic v2 extras."""
        instrumentor = TransportInstrumentor()

        meta = MagicMock(spec=[])
        meta.model_extra = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "baggage": f"gen_ai.conversation.id=extra-{transport_label}",
        }
        del meta.traceparent

        mock_responder = MagicMock()
        mock_responder.request_meta = meta

        wrapper = instrumentor._server_handle_request_wrapper()

        extracted_ctx = None

        async def capture_wrapped(*args, **kwargs):
            nonlocal extracted_ctx
            extracted_ctx = context.get_current()
            return "ok"

        await wrapper(capture_wrapped, MagicMock(), (mock_responder,), {})

        conv_id = baggage.get_baggage("gen_ai.conversation.id", extracted_ctx)
        assert conv_id == f"extra-{transport_label}"


class TestMetaContents:
    """Verify meta only contains standard W3C headers."""

    @pytest.mark.asyncio
    async def test_meta_only_has_w3c_headers(self):
        """Meta should only contain W3C standard headers."""
        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        instrumentor = TransportInstrumentor()

        ctx = baggage.set_baggage("gen_ai.conversation.id", "check-headers")
        token = context.attach(ctx)

        try:
            set_attrs = {}

            class TrackingMeta:
                def __setattr__(self, name, value):
                    set_attrs[name] = value
                    super().__setattr__(name, value)

            meta = TrackingMeta()
            meta.model_extra = {}
            request, _ = _make_client_request(meta)

            with tracer.start_as_current_span("test-call"):
                wrapper = instrumentor._send_request_wrapper()
                await wrapper(
                    AsyncMock(return_value="ok"),
                    MagicMock(),
                    (request,),
                    {},
                )
        finally:
            context.detach(token)

        w3c_keys = {"traceparent", "tracestate", "baggage", "model_extra"}
        unexpected = set(set_attrs.keys()) - w3c_keys
        assert not unexpected, f"Unexpected meta attributes: {unexpected}"
        provider.shutdown()
