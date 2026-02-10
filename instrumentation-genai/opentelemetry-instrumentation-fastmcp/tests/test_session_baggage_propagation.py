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

"""Tests for session baggage propagation through MCP transport."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opentelemetry import baggage, context
from opentelemetry.instrumentation.fastmcp.propagation import (
    restore_session_from_context,
)
from opentelemetry.instrumentation.fastmcp.transport_instrumentor import (
    TransportInstrumentor,
)
from opentelemetry.util.genai.handler import (
    clear_session_context,
    get_session_context,
)


class TestBaggageInjection:
    """Test that baggage is injected via OTel propagation in transport instrumentor."""

    @pytest.mark.asyncio
    async def test_inject_includes_baggage_when_set(self):
        """When OTel baggage is set, send_request wrapper injects it into meta."""
        # Set session via OTel Baggage API
        ctx = baggage.set_baggage("gen_ai.conversation.id", "test-sess-inject")
        ctx = baggage.set_baggage("user.id", "test-user-inject", ctx)
        token = context.attach(ctx)

        try:
            instrumentor = TransportInstrumentor()
            wrapper_func = instrumentor._send_request_wrapper()

            # Build a mock request with meta
            class MockMeta:
                pass

            mock_meta = MockMeta()
            mock_params = MagicMock()
            mock_params.meta = mock_meta
            mock_root = MagicMock()
            mock_root.method = "tools/call"
            mock_root.params = mock_params
            mock_request = MagicMock()
            mock_request.root = mock_root

            mock_wrapped = AsyncMock(return_value="result")

            # Use real propagate.inject so baggage is actually injected
            result = await wrapper_func(mock_wrapped, MagicMock(), (mock_request,), {})

            assert result == "result"
            # The meta should have baggage set as an attribute
            # Note: traceparent only appears when there's an active span
            assert hasattr(mock_meta, "baggage")
            assert "gen_ai.conversation.id=test-sess-inject" in mock_meta.baggage
        finally:
            context.detach(token)

    @pytest.mark.asyncio
    async def test_inject_with_no_baggage(self):
        """When no baggage set but active span exists, traceparent is injected."""
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        instrumentor = TransportInstrumentor()

        class MockMeta:
            pass

        mock_meta = MockMeta()
        mock_params = MagicMock()
        mock_params.meta = mock_meta
        mock_root = MagicMock()
        mock_root.method = "tools/call"
        mock_root.params = mock_params
        mock_request = MagicMock()
        mock_request.root = mock_root

        mock_wrapped = AsyncMock(return_value="ok")

        with tracer.start_as_current_span("test-call"):
            wrapper = instrumentor._send_request_wrapper()
            result = await wrapper(mock_wrapped, MagicMock(), (mock_request,), {})
        assert result == "ok"
        # traceparent should be set when there's an active span
        assert hasattr(mock_meta, "traceparent")
        provider.shutdown()

    @pytest.mark.asyncio
    async def test_inject_none_params_does_not_fail(self):
        """Wrapper handles request with None params gracefully."""
        instrumentor = TransportInstrumentor()
        wrapper_func = instrumentor._send_request_wrapper()

        mock_root = MagicMock()
        mock_root.params = None
        mock_request = MagicMock()
        mock_request.root = mock_root

        mock_wrapped = AsyncMock(return_value="ok")
        result = await wrapper_func(mock_wrapped, MagicMock(), (mock_request,), {})
        assert result == "ok"


class TestBaggageExtraction:
    """Test that baggage is extracted via OTel propagation in transport instrumentor."""

    @pytest.mark.asyncio
    async def test_extract_with_baggage_header(self):
        """Server wrapper extracts baggage from request_meta carrier."""
        instrumentor = TransportInstrumentor()

        class MockRequestMeta:
            def __init__(self):
                self.traceparent = (
                    "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
                )
                self.baggage = (
                    "gen_ai.conversation.id=extracted-sess,user.id=extracted-user"
                )
                self.model_extra = {}

        mock_responder = MagicMock()
        mock_responder.request_meta = MockRequestMeta()

        wrapper_func = instrumentor._server_handle_request_wrapper()

        # Capture the context that gets attached so we can read baggage from it
        extracted_ctx = None

        async def capture_wrapped(*args, **kwargs):
            nonlocal extracted_ctx
            # Inside the wrapper, the context is attached — read baggage
            extracted_ctx = context.get_current()
            return "result"

        result = await wrapper_func(capture_wrapped, MagicMock(), (mock_responder,), {})
        assert result == "result"

        # Verify baggage was extracted
        session_id = baggage.get_baggage("gen_ai.conversation.id", extracted_ctx)
        user_id = baggage.get_baggage("user.id", extracted_ctx)
        assert session_id == "extracted-sess"
        assert user_id == "extracted-user"

    @pytest.mark.asyncio
    async def test_extract_no_request_meta(self):
        """Server wrapper handles None request_meta gracefully."""
        instrumentor = TransportInstrumentor()

        mock_responder = MagicMock()
        mock_responder.request_meta = None

        wrapper_func = instrumentor._server_handle_request_wrapper()
        mock_wrapped = AsyncMock(return_value="ok")

        result = await wrapper_func(mock_wrapped, MagicMock(), (mock_responder,), {})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_extract_without_baggage(self):
        """Server wrapper works when meta has traceparent but no baggage."""
        instrumentor = TransportInstrumentor()

        class MockRequestMeta:
            def __init__(self):
                self.traceparent = (
                    "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
                )
                self.model_extra = {}

        mock_responder = MagicMock()
        mock_responder.request_meta = MockRequestMeta()

        wrapper_func = instrumentor._server_handle_request_wrapper()
        mock_wrapped = AsyncMock(return_value="ok")

        result = await wrapper_func(mock_wrapped, MagicMock(), (mock_responder,), {})
        assert result == "ok"


class TestRestoreSessionFromContext:
    """Test restore_session_from_context."""

    def test_restores_session_from_baggage(self):
        """Should set session context from baggage when enabled."""
        clear_session_context()

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            # Create a context with baggage
            ctx = baggage.set_baggage("gen_ai.conversation.id", "restored-sess")
            ctx = baggage.set_baggage("user.id", "restored-user", ctx)

            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id == "restored-sess"
            assert session.user_id == "restored-user"

        clear_session_context()

    def test_no_restore_when_disabled(self):
        """Should not restore when propagation is contextvar mode."""
        clear_session_context()

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "contextvar"},
        ):
            ctx = baggage.set_baggage("gen_ai.conversation.id", "should-not-restore")
            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id is None

    def test_no_error_on_empty_context(self):
        """Should not error when context has no baggage."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            restore_session_from_context(None)
            # Should not raise


class TestMetricSessionAttributes:
    """Test OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS."""

    def test_no_session_in_metrics_by_default(self):
        """By default, no session attributes in metrics."""
        from opentelemetry.util.genai.emitters.utils import (
            _get_session_metric_include_set,
        )

        # Clear cached value
        _get_session_metric_include_set.cache_clear()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS", None
            )
            result = _get_session_metric_include_set()
            assert len(result) == 0

        _get_session_metric_include_set.cache_clear()

    def test_all_session_in_metrics(self):
        from opentelemetry.util.genai.emitters.utils import (
            _get_session_metric_include_set,
        )

        _get_session_metric_include_set.cache_clear()

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS": "all"},
        ):
            result = _get_session_metric_include_set()
            assert "gen_ai.conversation.id" in result
            assert "user.id" in result
            assert "customer.id" in result

        _get_session_metric_include_set.cache_clear()

    def test_selective_session_in_metrics(self):
        from opentelemetry.util.genai.emitters.utils import (
            _get_session_metric_include_set,
        )

        _get_session_metric_include_set.cache_clear()

        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS": "user.id,customer.id"
            },
        ):
            result = _get_session_metric_include_set()
            assert "gen_ai.conversation.id" not in result
            assert "user.id" in result
            assert "customer.id" in result

        _get_session_metric_include_set.cache_clear()

    def test_get_session_metric_attributes(self):
        from opentelemetry.util.genai.emitters.utils import (
            _get_session_metric_include_set,
            get_session_metric_attributes,
        )
        from opentelemetry.util.genai.types import LLMInvocation

        _get_session_metric_include_set.cache_clear()

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_INCLUDE_IN_METRICS": "user.id"},
        ):
            invocation = LLMInvocation(
                request_model="test",
                session_id="sess-1",
                user_id="user-1",
                customer_id="cust-1",
            )
            attrs = get_session_metric_attributes(invocation)
            # Only user.id should be included
            assert "user.id" in attrs
            assert attrs["user.id"] == "user-1"
            assert "gen_ai.conversation.id" not in attrs
            assert "customer.id" not in attrs

        _get_session_metric_include_set.cache_clear()
