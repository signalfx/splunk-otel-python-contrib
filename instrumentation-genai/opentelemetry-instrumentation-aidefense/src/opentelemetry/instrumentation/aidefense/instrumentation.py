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
OpenTelemetry Cisco AI Defense Instrumentation

Wrapper-based instrumentation for Cisco AI Defense using splunk-otel-util-genai.

This instrumentation supports two modes:

1. **SDK Mode**: Wraps cisco-aidefense-sdk methods to capture security inspection events
   - ChatInspectionClient: inspect_prompt, inspect_response, inspect_conversation
   - HttpInspectionClient: inspect_request, inspect_response

2. **Gateway Mode**: Wraps HTTP client (httpx) to capture the X-Cisco-AI-Defense-Event-Id
   header from responses when LLM calls are proxied through AI Defense Gateway.
   - Automatically detects AI Defense Gateway URLs
   - Adds gen_ai.security.event_id to the current span (e.g., LangChain spans)
   - Supported providers: OpenAI, Azure OpenAI, Anthropic, Cohere, Mistral (any httpx-based SDK)

The critical `gen_ai.security.event_id` span attribute enables security event correlation
in Splunk APM and other observability platforms.
"""

import logging
import os
import re
from typing import Collection, List, Optional

from wrapt import wrap_function_wrapper
from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    LLMInvocation,
    OutputMessage,
    Text,
)

from opentelemetry.instrumentation.aidefense.util.helper import (
    add_event_id_to_current_span,
    create_ai_defense_invocation,
    create_input_message,
    execute_with_telemetry,
    get_server_address,
    MAX_SHORT_CONTENT_LENGTH,
    MAX_MESSAGES_IN_CONVERSATION,
)

_logger = logging.getLogger(__name__)

_instruments = ("cisco-aidefense-sdk >= 2.0.0",)

# AI Defense Gateway URL patterns
# These patterns identify requests going through AI Defense Gateway
AI_DEFENSE_GATEWAY_PATTERNS = [
    r"gateway\.aidefense\.security\.cisco\.com",
]

# Compiled regex patterns for efficient matching
_gateway_patterns_compiled: List[re.Pattern] = []

# Header containing the security event ID from AI Defense Gateway
AI_DEFENSE_EVENT_ID_HEADER = "X-Cisco-AI-Defense-Event-Id"

# Span attribute for security event ID
GEN_AI_SECURITY_EVENT_ID = "gen_ai.security.event_id"

# Global handler instance (singleton)
_handler: Optional[TelemetryHandler] = None


def _get_gateway_patterns() -> List[re.Pattern]:
    """
    Get compiled regex patterns for AI Defense Gateway URL matching.

    Includes both built-in patterns and custom patterns from environment variable.
    """
    global _gateway_patterns_compiled

    if _gateway_patterns_compiled:
        return _gateway_patterns_compiled

    patterns = list(AI_DEFENSE_GATEWAY_PATTERNS)

    # Add custom patterns from environment variable
    custom_patterns = os.environ.get("OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS", "")
    if custom_patterns:
        for pattern in custom_patterns.split(","):
            pattern = pattern.strip()
            if pattern:
                # Escape special regex chars if it looks like a literal string
                if not any(c in pattern for c in r".*+?^${}[]|\()"):
                    pattern = re.escape(pattern)
                patterns.append(pattern)

    _gateway_patterns_compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return _gateway_patterns_compiled


def _is_aidefense_gateway_url(url: str) -> bool:
    """
    Check if URL matches AI Defense Gateway patterns.

    Args:
        url: The URL to check (can be full URL or just host)

    Returns:
        True if the URL appears to be an AI Defense Gateway endpoint
    """
    if not url:
        return False

    patterns = _get_gateway_patterns()
    for pattern in patterns:
        if pattern.search(url):
            return True

    return False


def _extract_event_id_from_headers(headers) -> Optional[str]:
    """
    Extract X-Cisco-AI-Defense-Event-Id from HTTP response headers.

    Handles various header container types (dict, httpx.Headers, etc.)
    with case-insensitive matching for HTTP header compliance.
    """
    if not headers:
        return None

    header_lower = AI_DEFENSE_EVENT_ID_HEADER.lower()

    try:
        # Try .get() method first (works with dict, httpx.Headers, etc.)
        if hasattr(headers, "get"):
            # Try exact case first
            event_id = headers.get(AI_DEFENSE_EVENT_ID_HEADER)
            if event_id:
                return event_id
            # Try lowercase (HTTP headers are case-insensitive per RFC 7230)
            event_id = headers.get(header_lower)
            if event_id:
                return event_id

        # Fallback: iterate and do case-insensitive comparison
        if hasattr(headers, "items"):
            for key, value in headers.items():
                if key.lower() == header_lower:
                    return value

        return None
    except Exception:
        return None


class AIDefenseInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentation for Cisco AI Defense.

    This instrumentor provides standardized telemetry for AI Defense security
    operations, supporting two modes:

    1. **SDK Mode**: Wraps cisco-aidefense-sdk methods to capture security
       inspection events as dedicated spans.

    2. **Gateway Mode**: Wraps httpx to capture the X-Cisco-AI-Defense-Event-Id
       header when LLM calls are proxied through AI Defense Gateway.
       The event_id is added to the current span.

    Gateway Mode supports any LLM SDK that uses httpx internally:
    - OpenAI SDK
    - Azure OpenAI (via OpenAI SDK)
    - Anthropic SDK
    - Cohere SDK
    - Mistral SDK

    The primary attribute captured is `gen_ai.security.event_id`, which is
    essential for correlating security events in Splunk APM and GDI pipelines.
    """

    # Track which wrappers were applied for uninstrumentation
    _sdk_mode_applied: bool = False
    _gateway_mode_applied: bool = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Apply instrumentation to AI Defense SDK and Gateway Mode."""
        global _handler

        # Initialize TelemetryHandler with tracer provider
        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            tracer_provider = trace.get_tracer_provider()

        meter_provider = kwargs.get("meter_provider")
        if not meter_provider:
            from opentelemetry import metrics

            meter_provider = metrics.get_meter_provider()

        _handler = TelemetryHandler(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )

        # SDK Mode: Wrap AI Defense SDK methods (if SDK is installed)
        self._instrument_sdk_mode()

        # Gateway Mode: Wrap HTTP clients to capture gateway headers
        self._instrument_gateway_mode()

    def _instrument_sdk_mode(self):
        """Instrument AI Defense SDK methods (SDK Mode)."""
        if self._sdk_mode_applied:
            _logger.debug("SDK Mode already instrumented, skipping")
            return

        try:
            # ChatInspectionClient methods
            wrap_function_wrapper(
                "aidefense.runtime.chat_inspect",
                "ChatInspectionClient.inspect_prompt",
                _wrap_chat_inspect_prompt,
            )
            wrap_function_wrapper(
                "aidefense.runtime.chat_inspect",
                "ChatInspectionClient.inspect_response",
                _wrap_chat_inspect_response,
            )
            wrap_function_wrapper(
                "aidefense.runtime.chat_inspect",
                "ChatInspectionClient.inspect_conversation",
                _wrap_chat_inspect_conversation,
            )

            # HttpInspectionClient methods
            wrap_function_wrapper(
                "aidefense.runtime.http_inspect",
                "HttpInspectionClient.inspect_request",
                _wrap_http_inspect_request,
            )
            wrap_function_wrapper(
                "aidefense.runtime.http_inspect",
                "HttpInspectionClient.inspect_response",
                _wrap_http_inspect_response,
            )
            wrap_function_wrapper(
                "aidefense.runtime.http_inspect",
                "HttpInspectionClient.inspect_request_from_http_library",
                _wrap_http_inspect_request_from_library,
            )
            wrap_function_wrapper(
                "aidefense.runtime.http_inspect",
                "HttpInspectionClient.inspect_response_from_http_library",
                _wrap_http_inspect_response_from_library,
            )

            self._sdk_mode_applied = True
            _logger.debug("AI Defense SDK Mode instrumentation applied")
        except ImportError:
            _logger.debug(
                "cisco-aidefense-sdk not installed, skipping SDK Mode instrumentation"
            )
        except Exception as e:
            _logger.debug("Failed to apply SDK Mode instrumentation: %s", e)

    def _instrument_gateway_mode(self):
        """
        Instrument httpx for Gateway Mode.

        Wraps httpx (used by OpenAI, Anthropic, Cohere, Mistral SDKs) to capture
        X-Cisco-AI-Defense-Event-Id header from responses when LLM calls are
        proxied through AI Defense Gateway.
        """
        # Wrap httpx (covers OpenAI, Anthropic, Cohere, Mistral, etc.)
        try:
            wrap_function_wrapper(
                "httpx",
                "Client.send",
                _wrap_httpx_send_for_gateway,
            )
            wrap_function_wrapper(
                "httpx",
                "AsyncClient.send",
                _wrap_async_httpx_send_for_gateway,
            )
            self._gateway_mode_applied = True
            _logger.debug("AI Defense Gateway Mode instrumentation applied for httpx")
        except ImportError:
            _logger.debug("httpx not installed, skipping Gateway Mode instrumentation")
        except Exception as e:
            _logger.debug("Failed to apply Gateway Mode instrumentation: %s", e)

    def _uninstrument(self, **kwargs):
        """Remove instrumentation from AI Defense SDK and Gateway Mode."""
        # SDK Mode
        if self._sdk_mode_applied:
            try:
                unwrap(
                    "aidefense.runtime.chat_inspect.ChatInspectionClient",
                    "inspect_prompt",
                )
                unwrap(
                    "aidefense.runtime.chat_inspect.ChatInspectionClient",
                    "inspect_response",
                )
                unwrap(
                    "aidefense.runtime.chat_inspect.ChatInspectionClient",
                    "inspect_conversation",
                )
                unwrap(
                    "aidefense.runtime.http_inspect.HttpInspectionClient",
                    "inspect_request",
                )
                unwrap(
                    "aidefense.runtime.http_inspect.HttpInspectionClient",
                    "inspect_response",
                )
                unwrap(
                    "aidefense.runtime.http_inspect.HttpInspectionClient",
                    "inspect_request_from_http_library",
                )
                unwrap(
                    "aidefense.runtime.http_inspect.HttpInspectionClient",
                    "inspect_response_from_http_library",
                )
            except Exception:
                pass
            self._sdk_mode_applied = False

        # Gateway Mode
        if self._gateway_mode_applied:
            try:
                unwrap("httpx.Client", "send")
                unwrap("httpx.AsyncClient", "send")
            except Exception:
                pass
            self._gateway_mode_applied = False


# ============================================================================
# Gateway Mode Wrappers - httpx (OpenAI, Cohere, Mistral, etc.)
# ============================================================================


def _wrap_httpx_send_for_gateway(wrapped, instance, args, kwargs):
    """
    Wrap httpx.Client.send to capture AI Defense Gateway event_id.

    This wrapper intercepts HTTP responses and checks if they came from
    AI Defense Gateway. If so, it extracts the X-Cisco-AI-Defense-Event-Id
    header and adds it to the current span.

    This works because:
    1. LLM instrumentations (LangChain, OpenAI, etc.) create spans at a higher level
    2. Those spans are still active when httpx makes the actual HTTP call
    3. We add the event_id to that active span
    """
    response = wrapped(*args, **kwargs)
    _try_add_gateway_event_id_from_httpx_response(response, "httpx")
    return response


async def _wrap_async_httpx_send_for_gateway(wrapped, instance, args, kwargs):
    """
    Wrap httpx.AsyncClient.send to capture AI Defense Gateway event_id.

    Async version of _wrap_httpx_send_for_gateway.
    """
    response = await wrapped(*args, **kwargs)
    _try_add_gateway_event_id_from_httpx_response(response, "async httpx")
    return response


def _try_add_gateway_event_id_from_httpx_response(response, source: str) -> None:
    """
    Try to extract and add AI Defense event_id from httpx response to current span.

    Args:
        response: httpx response object
        source: Description for logging (e.g., "httpx", "async httpx")
    """
    try:
        request = response.request if hasattr(response, "request") else None
        url = str(request.url) if request and hasattr(request, "url") else ""

        if url and _is_aidefense_gateway_url(url):
            event_id = _extract_event_id_from_headers(response.headers)
            if event_id:
                if add_event_id_to_current_span(event_id):
                    _logger.debug(
                        "SUCCESS: Added gen_ai.security.event_id=%s to span from %s",
                        event_id,
                        source,
                    )
            else:
                _logger.debug(
                    "No event_id in %s response (request may not have triggered security)",
                    source,
                )
    except Exception as e:
        _logger.debug("Failed to extract AI Defense event_id from %s: %s", source, e)


# ============================================================================
# SDK Mode Wrappers - ChatInspectionClient
# ============================================================================


def _wrap_chat_inspect_prompt(wrapped, instance, args, kwargs):
    """
    Wrap ChatInspectionClient.inspect_prompt to create an LLMInvocation span.

    Captures the user prompt being inspected and the resulting security event_id.
    """
    prompt = kwargs.get("prompt") or (args[0] if args else "")
    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=[create_input_message("user", prompt)],
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


def _wrap_chat_inspect_response(wrapped, instance, args, kwargs):
    """
    Wrap ChatInspectionClient.inspect_response to create an LLMInvocation span.

    Captures the AI response being inspected and the resulting security event_id.
    """
    response = kwargs.get("response") or (args[0] if args else "")
    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=[create_input_message("assistant", response)],
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


def _wrap_chat_inspect_conversation(wrapped, instance, args, kwargs):
    """
    Wrap ChatInspectionClient.inspect_conversation to create an LLMInvocation span.

    Captures the full conversation being inspected and the resulting security event_id.
    """
    messages = kwargs.get("messages") or (args[0] if args else [])

    # Convert AI Defense messages to InputMessage format
    input_msgs = []
    for msg in messages[:MAX_MESSAGES_IN_CONVERSATION]:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        content = getattr(msg, "content", "")[:MAX_SHORT_CONTENT_LENGTH]
        input_msgs.append(create_input_message(role, content, MAX_SHORT_CONTENT_LENGTH))

    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=input_msgs,
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


# ============================================================================
# SDK Mode Wrappers - HttpInspectionClient
# ============================================================================


def _wrap_http_inspect_request(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_request to create an LLMInvocation span.

    Captures HTTP request inspection with method and URL context.
    """
    method = kwargs.get("method") or (args[0] if args else "")
    url = kwargs.get("url") or (args[1] if len(args) > 1 else "")
    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=[create_input_message("user", f"{method} {url}")],
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


def _wrap_http_inspect_response(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_response to create an LLMInvocation span.

    Captures HTTP response inspection with status code and URL context.
    """
    status_code = kwargs.get("status_code") or (args[0] if args else 0)
    url = kwargs.get("url") or (args[1] if len(args) > 1 else "")
    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=[
            create_input_message("assistant", f"HTTP {status_code} from {url}")
        ],
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


def _wrap_http_inspect_request_from_library(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_request_from_http_library.

    Handles requests from HTTP libraries like `requests`.
    """
    http_request = kwargs.get("http_request") or (args[0] if args else None)
    method = getattr(http_request, "method", "UNKNOWN") if http_request else "UNKNOWN"
    url = getattr(http_request, "url", "") if http_request else ""
    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=[create_input_message("user", f"{method} {url}")],
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


def _wrap_http_inspect_response_from_library(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_response_from_http_library.

    Handles responses from HTTP libraries like `requests`.
    """
    http_response = kwargs.get("http_response") or (args[0] if args else None)
    status_code = getattr(http_response, "status_code", 0) if http_response else 0
    url = getattr(http_response, "url", "") if http_response else ""
    invocation = create_ai_defense_invocation(
        server_address=get_server_address(instance),
        input_messages=[
            create_input_message("assistant", f"HTTP {status_code} from {url}")
        ],
    )
    return execute_with_telemetry(
        handler=_handler,
        invocation=invocation,
        wrapped=wrapped,
        args=args,
        kwargs=kwargs,
        result_processor=_populate_invocation_from_result,
    )


# ============================================================================
# SDK Mode Helper Functions
# ============================================================================


def _populate_invocation_from_result(invocation: LLMInvocation, result) -> None:
    """
    Populate LLMInvocation with InspectResponse data.

    The primary attribute captured is gen_ai.security.event_id, which is
    essential for security event correlation in Splunk APM.

    Args:
        invocation: The LLMInvocation to populate
        result: The InspectResponse from AI Defense API
    """
    # PRIMARY ATTRIBUTE: event_id for security event correlation
    if result.event_id:
        invocation.security_event_id = result.event_id

    # Build output message summarizing inspection result
    output_parts = []
    if result.action:
        action_value = (
            result.action.value
            if hasattr(result.action, "value")
            else str(result.action)
        )
        output_parts.append(f"action={action_value}")
    output_parts.append(f"is_safe={result.is_safe}")

    invocation.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[Text(content=", ".join(output_parts))],
            finish_reason="stop",
        )
    ]
