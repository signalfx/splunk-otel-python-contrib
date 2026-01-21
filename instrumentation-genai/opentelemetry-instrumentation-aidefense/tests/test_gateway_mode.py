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

"""Tests for AI Defense Gateway Mode functionality."""

import pytest
from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.aidefense.instrumentation import (
    AI_DEFENSE_EVENT_ID_HEADER,
    _extract_event_id_from_headers,
    _is_aidefense_gateway_url,
    _get_gateway_patterns,
)


# =============================================================================
# Tests for _extract_event_id_from_headers
# =============================================================================


class TestExtractEventIdFromHeaders:
    """Tests for _extract_event_id_from_headers function."""

    def test_exact_case_match(self):
        """Returns event_id when header matches exact case."""
        headers = {"X-Cisco-AI-Defense-Event-Id": "event-123"}
        assert _extract_event_id_from_headers(headers) == "event-123"

    def test_lowercase_match(self):
        """Returns event_id when header is lowercase."""
        headers = {"x-cisco-ai-defense-event-id": "event-456"}
        assert _extract_event_id_from_headers(headers) == "event-456"

    def test_mixed_case_match(self):
        """Returns event_id with case-insensitive iteration fallback."""
        headers = {"X-CISCO-AI-DEFENSE-EVENT-ID": "event-789"}
        assert _extract_event_id_from_headers(headers) == "event-789"

    def test_none_headers(self):
        """Returns None when headers is None."""
        assert _extract_event_id_from_headers(None) is None

    def test_empty_headers(self):
        """Returns None when headers is empty dict."""
        assert _extract_event_id_from_headers({}) is None

    def test_empty_list_headers(self):
        """Returns None when headers is empty list."""
        assert _extract_event_id_from_headers([]) is None

    def test_header_not_present(self):
        """Returns None when event_id header is not present."""
        headers = {"Content-Type": "application/json", "X-Request-Id": "req-123"}
        assert _extract_event_id_from_headers(headers) is None

    def test_httpx_headers_mock(self):
        """Works with httpx.Headers-like object (has .get() and .items())."""
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda k: (
            "httpx-event-id" if k == AI_DEFENSE_EVENT_ID_HEADER else None
        )
        assert _extract_event_id_from_headers(mock_headers) == "httpx-event-id"

    def test_httpx_headers_lowercase_fallback(self):
        """Falls back to lowercase .get() when exact case fails."""
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda k: (
            "lower-event-id" if k == AI_DEFENSE_EVENT_ID_HEADER.lower() else None
        )
        assert _extract_event_id_from_headers(mock_headers) == "lower-event-id"

    def test_items_iteration_fallback(self):
        """Uses .items() iteration when .get() returns None for both cases."""
        mock_headers = MagicMock()
        mock_headers.get.return_value = None
        mock_headers.items.return_value = [
            ("Content-Type", "application/json"),
            ("X-Cisco-Ai-Defense-Event-Id", "iterated-event-id"),  # Different case
        ]
        assert _extract_event_id_from_headers(mock_headers) == "iterated-event-id"

    def test_exception_returns_none(self):
        """Returns None when an exception occurs."""
        mock_headers = MagicMock()
        mock_headers.get.side_effect = RuntimeError("Unexpected error")
        assert _extract_event_id_from_headers(mock_headers) is None

    def test_empty_string_event_id(self):
        """Returns empty string if header value is empty string."""
        headers = {AI_DEFENSE_EVENT_ID_HEADER: ""}
        # Empty string is truthy in headers.get() check, so it returns ""
        assert _extract_event_id_from_headers(headers) == ""

    def test_object_without_get_but_with_items(self):
        """Works with objects that have .items() but not .get()."""

        class ItemsOnlyHeaders:
            def items(self):
                return [("x-cisco-ai-defense-event-id", "items-only-event")]

        headers = ItemsOnlyHeaders()
        assert _extract_event_id_from_headers(headers) == "items-only-event"


# =============================================================================
# Tests for _is_aidefense_gateway_url
# =============================================================================


class TestIsAIDefenseGatewayUrl:
    """Tests for _is_aidefense_gateway_url function."""

    @pytest.fixture(autouse=True)
    def reset_patterns(self):
        """Reset compiled patterns before each test."""
        global _gateway_patterns_compiled
        # Clear the module-level cache
        import opentelemetry.instrumentation.aidefense.instrumentation as mod

        mod._gateway_patterns_compiled = []
        yield
        mod._gateway_patterns_compiled = []

    def test_matches_default_gateway_pattern(self):
        """Matches the default AI Defense Gateway URL pattern."""
        url = "https://gateway.aidefense.security.cisco.com/v1/chat/completions"
        assert _is_aidefense_gateway_url(url) is True

    def test_matches_gateway_in_subdomain(self):
        """Matches when gateway pattern is in subdomain."""
        url = "https://api.gateway.aidefense.security.cisco.com/openai"
        assert _is_aidefense_gateway_url(url) is True

    def test_no_match_openai_direct(self):
        """Does not match direct OpenAI API calls."""
        url = "https://api.openai.com/v1/chat/completions"
        assert _is_aidefense_gateway_url(url) is False

    def test_no_match_azure_openai(self):
        """Does not match Azure OpenAI without gateway pattern."""
        url = "https://myresource.openai.azure.com/openai/deployments/gpt-4"
        assert _is_aidefense_gateway_url(url) is False

    def test_empty_url(self):
        """Returns False for empty URL."""
        assert _is_aidefense_gateway_url("") is False

    def test_none_url(self):
        """Returns False for None URL."""
        assert _is_aidefense_gateway_url(None) is False

    def test_case_insensitive_matching(self):
        """URL matching is case-insensitive."""
        url = "https://GATEWAY.AIDEFENSE.SECURITY.CISCO.COM/v1/chat"
        assert _is_aidefense_gateway_url(url) is True

    def test_custom_pattern_from_env(self, monkeypatch):
        """Adds custom patterns from OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS",
            "my-custom-gateway.example.com,another-gateway.corp.net",
        )

        # Custom pattern should match
        url = "https://my-custom-gateway.example.com/api/v1"
        assert _is_aidefense_gateway_url(url) is True

        # Second custom pattern should match
        url2 = "https://another-gateway.corp.net/llm"
        assert _is_aidefense_gateway_url(url2) is True

    def test_custom_pattern_with_regex(self, monkeypatch):
        """Custom patterns can include regex."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS",
            r"gateway\.preview\.aidefense\..*\.cisco\.com",
        )

        url = "https://gateway.preview.aidefense.aiteam.cisco.com/api"
        assert _is_aidefense_gateway_url(url) is True

    def test_custom_pattern_with_dots_treated_as_regex(self, monkeypatch):
        """Patterns with dots are treated as regex (dot matches any char)."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS", "simple.gateway.com"
        )

        # Should match literally
        url = "https://simple.gateway.com/api"
        assert _is_aidefense_gateway_url(url) is True

        # Dot in pattern matches any char (regex behavior)
        url_variant = "https://simplexgatewayxcom/api"
        assert _is_aidefense_gateway_url(url_variant) is True

    def test_custom_pattern_literal_no_special_chars(self, monkeypatch):
        """Patterns without any regex chars are auto-escaped."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS", "mygateway")

        # Should match literally
        url = "https://mygateway.example.com/api"
        assert _is_aidefense_gateway_url(url) is True

        # Substring match works
        url2 = "https://api.mygateway.net/llm"
        assert _is_aidefense_gateway_url(url2) is True


# =============================================================================
# Tests for _get_gateway_patterns
# =============================================================================


class TestGetGatewayPatterns:
    """Tests for _get_gateway_patterns function."""

    @pytest.fixture(autouse=True)
    def reset_patterns(self):
        """Reset compiled patterns before each test."""
        import opentelemetry.instrumentation.aidefense.instrumentation as mod

        mod._gateway_patterns_compiled = []
        yield
        mod._gateway_patterns_compiled = []

    def test_returns_compiled_regex_patterns(self):
        """Returns list of compiled regex patterns."""
        import re

        patterns = _get_gateway_patterns()
        assert len(patterns) >= 1
        for pattern in patterns:
            assert isinstance(pattern, re.Pattern)

    def test_caches_patterns(self):
        """Caches compiled patterns for efficiency."""
        patterns1 = _get_gateway_patterns()
        patterns2 = _get_gateway_patterns()
        assert patterns1 is patterns2  # Same object reference

    def test_includes_default_pattern(self):
        """Includes the default AI Defense Gateway pattern."""
        patterns = _get_gateway_patterns()
        # At least one pattern should match the default gateway
        test_url = "gateway.aidefense.security.cisco.com"
        matches = any(p.search(test_url) for p in patterns)
        assert matches is True

    def test_includes_custom_patterns_from_env(self, monkeypatch):
        """Includes custom patterns from environment variable."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS", "custom.gateway.io"
        )

        patterns = _get_gateway_patterns()
        test_url = "custom.gateway.io"
        matches = any(p.search(test_url) for p in patterns)
        assert matches is True

    def test_handles_empty_custom_patterns(self, monkeypatch):
        """Handles empty OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS."""
        monkeypatch.setenv("OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS", "")

        patterns = _get_gateway_patterns()
        # Should still have default patterns
        assert len(patterns) >= 1

    def test_handles_whitespace_in_custom_patterns(self, monkeypatch):
        """Strips whitespace from custom patterns."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS",
            "  gateway1.com  ,  gateway2.com  ",
        )

        patterns = _get_gateway_patterns()
        # Both should be included
        assert any(p.search("gateway1.com") for p in patterns)
        assert any(p.search("gateway2.com") for p in patterns)


# =============================================================================
# Tests for AI_DEFENSE_EVENT_ID_HEADER constant
# =============================================================================


class TestHeaderConstant:
    """Tests for the header constant."""

    def test_header_constant_value(self):
        """Verifies the header constant has the expected value."""
        assert AI_DEFENSE_EVENT_ID_HEADER == "X-Cisco-AI-Defense-Event-Id"

    def test_header_lowercase_derivation(self):
        """Verifies lowercase can be derived from the constant."""
        assert AI_DEFENSE_EVENT_ID_HEADER.lower() == "x-cisco-ai-defense-event-id"


# =============================================================================
# Tests for Gateway Mode Wrappers
# =============================================================================


class TestGatewayModeWrappers:
    """Tests for Gateway Mode httpx wrapper functions."""

    @pytest.fixture(autouse=True)
    def reset_patterns(self):
        """Reset compiled patterns before each test."""
        import opentelemetry.instrumentation.aidefense.instrumentation as mod

        mod._gateway_patterns_compiled = []
        yield
        mod._gateway_patterns_compiled = []

    def test_wrap_httpx_send_calls_wrapped_and_returns_response(self):
        """Wrapper calls the original function and returns response."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _wrap_httpx_send_for_gateway,
        )

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.request = MagicMock()
        mock_response.request.url = "https://api.openai.com/v1/chat"

        wrapped = MagicMock(return_value=mock_response)
        instance = MagicMock()

        result = _wrap_httpx_send_for_gateway(wrapped, instance, (), {})

        wrapped.assert_called_once_with()
        assert result is mock_response

    def test_wrap_httpx_send_extracts_event_id_from_gateway_response(self):
        """Extracts event_id from AI Defense Gateway response."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _wrap_httpx_send_for_gateway,
        )

        # Create response from AI Defense Gateway
        mock_response = MagicMock()
        mock_response.headers = {AI_DEFENSE_EVENT_ID_HEADER: "gateway-event-123"}
        mock_response.request = MagicMock()
        mock_response.request.url = (
            "https://gateway.aidefense.security.cisco.com/v1/chat"
        )

        wrapped = MagicMock(return_value=mock_response)

        # Mock span via patch on the trace module
        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_get_span.return_value = mock_span

            result = _wrap_httpx_send_for_gateway(wrapped, MagicMock(), (), {})

            assert result is mock_response
            mock_span.set_attribute.assert_called_with(
                "gen_ai.security.event_id", "gateway-event-123"
            )

    def test_wrap_httpx_send_ignores_non_gateway_urls(self):
        """Does not extract event_id from non-gateway URLs."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _wrap_httpx_send_for_gateway,
        )

        # Response from direct OpenAI (not gateway)
        mock_response = MagicMock()
        mock_response.headers = {AI_DEFENSE_EVENT_ID_HEADER: "should-not-capture"}
        mock_response.request = MagicMock()
        mock_response.request.url = "https://api.openai.com/v1/chat"

        wrapped = MagicMock(return_value=mock_response)

        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_get_span.return_value = mock_span

            result = _wrap_httpx_send_for_gateway(wrapped, MagicMock(), (), {})

            assert result is mock_response
            mock_span.set_attribute.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrap_async_httpx_send_calls_wrapped_and_returns_response(self):
        """Async wrapper calls the original function and returns response."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _wrap_async_httpx_send_for_gateway,
        )

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.request = MagicMock()
        mock_response.request.url = "https://api.openai.com/v1/chat"

        async def async_wrapped(*args, **kwargs):
            return mock_response

        result = await _wrap_async_httpx_send_for_gateway(
            async_wrapped, MagicMock(), (), {}
        )

        assert result is mock_response

    @pytest.mark.asyncio
    async def test_wrap_async_httpx_send_extracts_event_id(self):
        """Async wrapper extracts event_id from AI Defense Gateway response."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _wrap_async_httpx_send_for_gateway,
        )

        mock_response = MagicMock()
        mock_response.headers = {AI_DEFENSE_EVENT_ID_HEADER: "async-event-456"}
        mock_response.request = MagicMock()
        mock_response.request.url = (
            "https://gateway.aidefense.security.cisco.com/v1/chat"
        )

        async def async_wrapped(*args, **kwargs):
            return mock_response

        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_get_span.return_value = mock_span

            result = await _wrap_async_httpx_send_for_gateway(
                async_wrapped, MagicMock(), (), {}
            )

            assert result is mock_response
            mock_span.set_attribute.assert_called_with(
                "gen_ai.security.event_id", "async-event-456"
            )


class TestTryAddGatewayEventId:
    """Tests for _try_add_gateway_event_id_from_httpx_response helper."""

    @pytest.fixture(autouse=True)
    def reset_patterns(self):
        """Reset compiled patterns before each test."""
        import opentelemetry.instrumentation.aidefense.instrumentation as mod

        mod._gateway_patterns_compiled = []
        yield
        mod._gateway_patterns_compiled = []

    def test_adds_event_id_to_current_span(self):
        """Adds event_id to the current span when present."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _try_add_gateway_event_id_from_httpx_response,
        )

        response = MagicMock()
        response.headers = {AI_DEFENSE_EVENT_ID_HEADER: "test-event-id"}
        response.request = MagicMock()
        response.request.url = "https://gateway.aidefense.security.cisco.com/v1/chat"

        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_get_span.return_value = mock_span

            _try_add_gateway_event_id_from_httpx_response(response, "test")

            mock_span.set_attribute.assert_called_once_with(
                "gen_ai.security.event_id", "test-event-id"
            )

    def test_does_nothing_when_no_event_id_header(self):
        """Does nothing when event_id header is not present."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _try_add_gateway_event_id_from_httpx_response,
        )

        response = MagicMock()
        response.headers = {"Content-Type": "application/json"}
        response.request = MagicMock()
        response.request.url = "https://gateway.aidefense.security.cisco.com/v1/chat"

        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_get_span.return_value = mock_span

            _try_add_gateway_event_id_from_httpx_response(response, "test")

            mock_span.set_attribute.assert_not_called()

    def test_does_nothing_when_not_gateway_url(self):
        """Does nothing when URL is not an AI Defense Gateway."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _try_add_gateway_event_id_from_httpx_response,
        )

        response = MagicMock()
        response.headers = {AI_DEFENSE_EVENT_ID_HEADER: "should-not-capture"}
        response.request = MagicMock()
        response.request.url = "https://api.openai.com/v1/chat"

        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_get_span.return_value = mock_span

            _try_add_gateway_event_id_from_httpx_response(response, "test")

            mock_span.set_attribute.assert_not_called()

    def test_handles_missing_request_gracefully(self):
        """Handles responses without request attribute gracefully."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _try_add_gateway_event_id_from_httpx_response,
        )

        response = MagicMock(spec=[])  # No attributes

        # Should not raise
        _try_add_gateway_event_id_from_httpx_response(response, "test")

    def test_handles_exception_gracefully(self):
        """Handles exceptions gracefully without crashing."""
        from opentelemetry.instrumentation.aidefense.instrumentation import (
            _try_add_gateway_event_id_from_httpx_response,
        )

        response = MagicMock()
        response.request = MagicMock()
        response.request.url = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        # Should not raise
        _try_add_gateway_event_id_from_httpx_response(response, "test")


# =============================================================================
# Tests for AIDefenseInstrumentor Gateway Mode
# =============================================================================


class TestAIDefenseInstrumentorGatewayMode:
    """Tests for AIDefenseInstrumentor Gateway Mode functionality."""

    def test_gateway_mode_flag_initially_false(self):
        """Gateway mode flag is initially False on a new instance."""
        from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

        instrumentor = AIDefenseInstrumentor()
        # Reset flag to test initial state
        instrumentor._gateway_mode_applied = False
        assert instrumentor._gateway_mode_applied is False

    def test_gateway_mode_flag_set_after_instrumentation(self):
        """Gateway mode flag is set to True after successful instrumentation."""
        from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

        with patch(
            "opentelemetry.instrumentation.aidefense.instrumentation.wrap_function_wrapper"
        ) as mock_wrap:
            instrumentor = AIDefenseInstrumentor()
            instrumentor._gateway_mode_applied = False  # Reset

            instrumentor._instrument_gateway_mode()

            assert instrumentor._gateway_mode_applied is True
            assert mock_wrap.call_count == 2  # Client.send and AsyncClient.send

    def test_gateway_mode_flag_false_when_httpx_not_installed(self):
        """Gateway mode flag stays False when httpx is not installed."""
        from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'httpx'")

        with patch(
            "opentelemetry.instrumentation.aidefense.instrumentation.wrap_function_wrapper",
            side_effect=raise_import_error,
        ):
            instrumentor = AIDefenseInstrumentor()
            instrumentor._gateway_mode_applied = False  # Reset

            instrumentor._instrument_gateway_mode()

            assert instrumentor._gateway_mode_applied is False

    def test_uninstrument_clears_gateway_mode_flag(self):
        """Uninstrumentation clears the gateway mode flag."""
        from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

        with patch(
            "opentelemetry.instrumentation.aidefense.instrumentation.unwrap"
        ) as mock_unwrap:
            instrumentor = AIDefenseInstrumentor()
            instrumentor._gateway_mode_applied = True
            instrumentor._sdk_mode_applied = False

            instrumentor._uninstrument()

            assert instrumentor._gateway_mode_applied is False
            # Should have called unwrap for httpx Client and AsyncClient
            assert mock_unwrap.call_count >= 2

    def test_uninstrument_does_nothing_when_not_instrumented(self):
        """Uninstrumentation does nothing when gateway mode was not applied."""
        from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

        with patch(
            "opentelemetry.instrumentation.aidefense.instrumentation.unwrap"
        ) as mock_unwrap:
            instrumentor = AIDefenseInstrumentor()
            instrumentor._gateway_mode_applied = False
            instrumentor._sdk_mode_applied = False

            instrumentor._uninstrument()

            # unwrap should not be called since neither mode was applied
            mock_unwrap.assert_not_called()

    def test_wrapper_count_for_gateway_mode(self):
        """Gateway mode wraps exactly 2 httpx methods (sync and async)."""
        from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

        wrapped_methods = []

        def capture_wrap(module, method, wrapper):
            wrapped_methods.append((module, method))

        with patch(
            "opentelemetry.instrumentation.aidefense.instrumentation.wrap_function_wrapper",
            side_effect=capture_wrap,
        ):
            instrumentor = AIDefenseInstrumentor()
            instrumentor._gateway_mode_applied = False

            instrumentor._instrument_gateway_mode()

            assert len(wrapped_methods) == 2
            assert ("httpx", "Client.send") in wrapped_methods
            assert ("httpx", "AsyncClient.send") in wrapped_methods
