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

"""Tests for transport_instrumentor module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry.instrumentation.fastmcp.transport_instrumentor import (
    TransportInstrumentor,
)


class TestTransportInstrumentor:
    """Tests for TransportInstrumentor class."""

    def test_init(self):
        """Test TransportInstrumentor initialization."""
        instrumentor = TransportInstrumentor()
        assert instrumentor._instrumented is False

    def test_instrument_sets_flag(self):
        """Test that instrument sets the instrumented flag."""
        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.register_post_import_hook"
        ):
            instrumentor = TransportInstrumentor()
            instrumentor.instrument()
            assert instrumentor._instrumented is True

    def test_instrument_idempotent(self):
        """Test that instrument is idempotent."""
        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.register_post_import_hook"
        ) as mock_hook:
            instrumentor = TransportInstrumentor()
            instrumentor.instrument()
            instrumentor.instrument()  # Second call should be no-op
            # Should only register hooks once
            assert mock_hook.call_count == 2  # Two hooks registered once

    def test_uninstrument_clears_flag(self):
        """Test that uninstrument clears the instrumented flag."""
        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.register_post_import_hook"
        ):
            instrumentor = TransportInstrumentor()
            instrumentor.instrument()
            instrumentor.uninstrument()
            assert instrumentor._instrumented is False


class TestSendRequestWrapper:
    """Tests for the send_request wrapper."""

    @pytest.mark.asyncio
    async def test_injects_trace_context_into_request_params_meta(self):
        """Test that trace context is injected into request.root.params.meta."""
        instrumentor = TransportInstrumentor()

        # Create a simple class to track attribute setting
        class MockMeta:
            def __init__(self):
                self.model_extra = {}
                self.traceparent = None
                self.tracestate = None

        mock_meta = MockMeta()

        mock_params = MagicMock()
        mock_params.meta = mock_meta

        # The actual request (inside the root)
        mock_root = MagicMock()
        mock_root.method = "tools/call"
        mock_root.params = mock_params

        # ClientRequest wrapper with root attribute
        mock_request = MagicMock()
        mock_request.root = mock_root

        wrapper_func = instrumentor._send_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.propagate"
        ) as mock_propagate:
            mock_propagate.inject.return_value = None

            # Simulate inject populating the carrier
            def inject_side_effect(carrier):
                carrier["traceparent"] = "00-abc123-def456-01"

            mock_propagate.inject.side_effect = inject_side_effect

            result = await wrapper_func(mock_wrapped, MagicMock(), (mock_request,), {})

            mock_propagate.inject.assert_called_once()
            mock_wrapped.assert_called_once()
            assert result == "result"
            # Verify traceparent was set on meta
            assert mock_meta.traceparent == "00-abc123-def456-01"

    @pytest.mark.asyncio
    async def test_creates_meta_if_none(self):
        """Test that meta is created if it doesn't exist."""
        instrumentor = TransportInstrumentor()

        mock_params = MagicMock()
        mock_params.meta = None

        mock_root = MagicMock()
        mock_root.method = "tools/call"
        mock_root.params = mock_params

        mock_request = MagicMock()
        mock_request.root = mock_root

        wrapper_func = instrumentor._send_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        # Mock the RequestParams.Meta class
        mock_meta_class = MagicMock()
        mock_meta_instance = MagicMock()
        mock_meta_class.return_value = mock_meta_instance

        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.propagate"
        ) as mock_propagate:
            mock_propagate.inject.return_value = None

            with patch.dict(
                "sys.modules",
                {"mcp.types": MagicMock(RequestParams=MagicMock(Meta=mock_meta_class))},
            ):
                result = await wrapper_func(
                    mock_wrapped, MagicMock(), (mock_request,), {}
                )

            assert result == "result"

    @pytest.mark.asyncio
    async def test_handles_request_without_params(self):
        """Test that requests without params are handled gracefully."""
        instrumentor = TransportInstrumentor()

        mock_root = MagicMock()
        mock_root.params = None

        mock_request = MagicMock()
        mock_request.root = mock_root

        wrapper_func = instrumentor._send_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        # Should not raise
        result = await wrapper_func(mock_wrapped, MagicMock(), (mock_request,), {})
        assert result == "result"

    @pytest.mark.asyncio
    async def test_handles_empty_args(self):
        """Test that empty args are handled gracefully."""
        instrumentor = TransportInstrumentor()
        wrapper_func = instrumentor._send_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        # Should not raise even with empty args
        result = await wrapper_func(mock_wrapped, MagicMock(), (), {})
        assert result == "result"


class TestServerReceivedRequestWrapper:
    """Tests for the server received request wrapper."""

    @pytest.mark.asyncio
    async def test_extracts_trace_context_from_responder_request_meta(self):
        """Test that trace context is extracted from responder.request_meta."""
        instrumentor = TransportInstrumentor()

        # Create mock responder with request_meta containing traceparent
        # Use a simple class instead of MagicMock to avoid __dict__ issues
        class MockRequestMeta:
            def __init__(self):
                self.traceparent = "00-abc123-def456-01"
                self.tracestate = "vendor=value"
                self.model_extra = {}

        mock_request_meta = MockRequestMeta()

        mock_responder = MagicMock()
        mock_responder.request_meta = mock_request_meta

        wrapper_func = instrumentor._server_received_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        mock_ctx = MagicMock()
        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.propagate"
        ) as mock_propagate:
            with patch(
                "opentelemetry.instrumentation.fastmcp.transport_instrumentor.context"
            ) as mock_context:
                mock_propagate.extract.return_value = mock_ctx
                mock_context.attach.return_value = "token"

                result = await wrapper_func(
                    mock_wrapped, MagicMock(), (mock_responder,), {}
                )

                # Verify extract was called with carrier containing traceparent
                mock_propagate.extract.assert_called_once()
                call_args = mock_propagate.extract.call_args[0][0]
                assert "traceparent" in call_args

                mock_context.attach.assert_called_once_with(mock_ctx)
                mock_context.detach.assert_called_once_with("token")
                assert result == "result"

    @pytest.mark.asyncio
    async def test_extracts_from_model_extra(self):
        """Test extraction from pydantic model_extra field."""
        instrumentor = TransportInstrumentor()

        # Create mock with traceparent in model_extra
        mock_request_meta = MagicMock(spec=[])
        mock_request_meta.model_extra = {
            "traceparent": "00-abc123-def456-01",
        }
        # Remove traceparent attribute so it falls through to model_extra
        del mock_request_meta.traceparent

        mock_responder = MagicMock()
        mock_responder.request_meta = mock_request_meta

        wrapper_func = instrumentor._server_received_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        mock_ctx = MagicMock()
        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.propagate"
        ) as mock_propagate:
            with patch(
                "opentelemetry.instrumentation.fastmcp.transport_instrumentor.context"
            ) as mock_context:
                mock_propagate.extract.return_value = mock_ctx
                mock_context.attach.return_value = "token"

                result = await wrapper_func(
                    mock_wrapped, MagicMock(), (mock_responder,), {}
                )

                mock_propagate.extract.assert_called_once()
                assert result == "result"

    @pytest.mark.asyncio
    async def test_handles_responder_without_request_meta(self):
        """Test that responders without request_meta are handled gracefully."""
        instrumentor = TransportInstrumentor()

        mock_responder = MagicMock()
        mock_responder.request_meta = None

        wrapper_func = instrumentor._server_received_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        result = await wrapper_func(mock_wrapped, MagicMock(), (mock_responder,), {})
        assert result == "result"

    @pytest.mark.asyncio
    async def test_handles_empty_carrier(self):
        """Test that empty carrier (no traceparent) skips context attachment."""
        instrumentor = TransportInstrumentor()

        # Create mock without traceparent
        mock_request_meta = MagicMock(spec=[])
        mock_request_meta.model_extra = {}

        mock_responder = MagicMock()
        mock_responder.request_meta = mock_request_meta

        wrapper_func = instrumentor._server_received_request_wrapper()
        mock_wrapped = AsyncMock(return_value="result")

        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.context"
        ) as mock_context:
            result = await wrapper_func(
                mock_wrapped, MagicMock(), (mock_responder,), {}
            )

            # Context should not be attached when no trace context found
            mock_context.attach.assert_not_called()
            assert result == "result"

    @pytest.mark.asyncio
    async def test_detaches_context_on_exception(self):
        """Test that context is detached even when wrapped function raises."""
        instrumentor = TransportInstrumentor()

        # Use a simple class instead of MagicMock to avoid __dict__ issues
        class MockRequestMeta:
            def __init__(self):
                self.traceparent = "00-abc123-def456-01"
                self.model_extra = {}

        mock_request_meta = MockRequestMeta()

        mock_responder = MagicMock()
        mock_responder.request_meta = mock_request_meta

        wrapper_func = instrumentor._server_received_request_wrapper()
        mock_wrapped = AsyncMock(side_effect=ValueError("test error"))

        mock_ctx = MagicMock()
        with patch(
            "opentelemetry.instrumentation.fastmcp.transport_instrumentor.propagate"
        ) as mock_propagate:
            with patch(
                "opentelemetry.instrumentation.fastmcp.transport_instrumentor.context"
            ) as mock_context:
                mock_propagate.extract.return_value = mock_ctx
                mock_context.attach.return_value = "token"

                with pytest.raises(ValueError):
                    await wrapper_func(mock_wrapped, MagicMock(), (mock_responder,), {})

                # Context should still be detached
                mock_context.detach.assert_called_once_with("token")
