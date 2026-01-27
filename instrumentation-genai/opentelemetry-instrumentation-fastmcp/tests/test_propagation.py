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

"""Tests for propagation module."""

from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from opentelemetry.instrumentation.fastmcp.propagation import (
    inject_trace_context,
    extract_trace_context,
    ContextManager,
)


class TestInjectTraceContext:
    """Tests for inject_trace_context function."""

    def test_none_params(self):
        """Test that None params are returned unchanged."""
        result = inject_trace_context(None)
        assert result is None

    def test_dict_params_creates_meta(self):
        """Test that _meta is created in dict params."""
        params = {"key": "value"}
        result = inject_trace_context(params)
        assert "_meta" in result
        assert result["key"] == "value"

    def test_dict_params_preserves_existing_meta(self):
        """Test that existing _meta content is preserved."""
        params = {"key": "value", "_meta": {"existing": "data"}}
        result = inject_trace_context(params)
        assert result["_meta"]["existing"] == "data"

    def test_object_with_dict(self):
        """Test injection into object with __dict__."""

        @dataclass
        class Params:
            key: str
            _meta: dict = None

            def __post_init__(self):
                if self._meta is None:
                    self._meta = {}

        params = Params(key="value")
        result = inject_trace_context(params)
        assert hasattr(result, "_meta")

    @patch("opentelemetry.instrumentation.fastmcp.propagation.propagate")
    def test_propagate_inject_called(self, mock_propagate):
        """Test that propagate.inject is called with meta dict."""
        params = {}
        inject_trace_context(params)
        mock_propagate.inject.assert_called_once()


class TestExtractTraceContext:
    """Tests for extract_trace_context function."""

    def test_none_params(self):
        """Test that None params returns None."""
        result = extract_trace_context(None)
        assert result is None

    def test_dict_without_meta(self):
        """Test dict without _meta returns None."""
        params = {"key": "value"}
        result = extract_trace_context(params)
        assert result is None

    def test_dict_with_meta(self):
        """Test dict with _meta calls extract."""
        params = {
            "_meta": {
                "traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
            }
        }
        with patch(
            "opentelemetry.instrumentation.fastmcp.propagation.propagate"
        ) as mock_propagate:
            mock_propagate.extract.return_value = MagicMock()
            _result = extract_trace_context(params)
            mock_propagate.extract.assert_called_once()

    def test_object_with_meta_attribute(self):
        """Test extraction from object with _meta attribute."""

        @dataclass
        class Params:
            _meta: dict

        params = Params(_meta={"traceparent": "00-abc-def-01"})
        with patch(
            "opentelemetry.instrumentation.fastmcp.propagation.propagate"
        ) as mock_propagate:
            mock_propagate.extract.return_value = MagicMock()
            _result = extract_trace_context(params)
            mock_propagate.extract.assert_called_once()

    def test_mapping_like_object(self):
        """Test extraction from mapping-like object with get method."""

        class MappingParams:
            def get(self, key, default=None):
                if key == "_meta":
                    return {"traceparent": "00-abc-def-01"}
                return default

        params = MappingParams()
        with patch(
            "opentelemetry.instrumentation.fastmcp.propagation.propagate"
        ) as mock_propagate:
            mock_propagate.extract.return_value = MagicMock()
            _result = extract_trace_context(params)
            mock_propagate.extract.assert_called_once()


class TestContextManager:
    """Tests for ContextManager class."""

    def test_attach_none_context(self):
        """Test attach with None context returns False."""
        cm = ContextManager()
        result = cm.attach(None)
        assert result is False

    def test_attach_valid_context(self):
        """Test attach with valid context returns True."""
        mock_ctx = MagicMock()
        with patch(
            "opentelemetry.instrumentation.fastmcp.propagation.context"
        ) as mock_context:
            mock_context.attach.return_value = "token"
            cm = ContextManager()
            result = cm.attach(mock_ctx)
            assert result is True
            mock_context.attach.assert_called_once_with(mock_ctx)

    def test_detach_without_attach(self):
        """Test detach without prior attach does nothing."""
        cm = ContextManager()
        cm.detach()  # Should not raise

    def test_detach_after_attach(self):
        """Test detach after attach calls context.detach."""
        mock_ctx = MagicMock()
        with patch(
            "opentelemetry.instrumentation.fastmcp.propagation.context"
        ) as mock_context:
            mock_context.attach.return_value = "token"
            cm = ContextManager()
            cm.attach(mock_ctx)
            cm.detach()
            mock_context.detach.assert_called_once_with("token")

    def test_context_manager_protocol(self):
        """Test ContextManager works as context manager."""
        mock_ctx = MagicMock()
        with patch(
            "opentelemetry.instrumentation.fastmcp.propagation.context"
        ) as mock_context:
            mock_context.attach.return_value = "token"
            cm = ContextManager()
            cm.attach(mock_ctx)
            with cm:
                pass
            mock_context.detach.assert_called_once_with("token")
