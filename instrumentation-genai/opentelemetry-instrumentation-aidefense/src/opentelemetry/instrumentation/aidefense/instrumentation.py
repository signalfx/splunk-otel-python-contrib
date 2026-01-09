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

Wrapper-based instrumentation for Cisco AI Defense Python SDK using splunk-otel-util-genai.

This instrumentation captures security inspection events from AI Defense API mode,
adding the critical `gen_ai.security.event_id` span attribute for security event correlation.

Supported methods:
- ChatInspectionClient: inspect_prompt, inspect_response, inspect_conversation
- HttpInspectionClient: inspect_request, inspect_response

Note: Proxy mode (x-cisco-ai-defense-tenant-api-key header) is not yet supported.
"""

from typing import Collection, Optional

from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    LLMInvocation,
    InputMessage,
    OutputMessage,
    Text,
    Error,
)

_instruments = ("cisco-aidefense-sdk >= 2.0.0",)

# Global handler instance (singleton)
_handler: Optional[TelemetryHandler] = None


class AIDefenseInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentation for Cisco AI Defense Python SDK.

    This instrumentor provides standardized telemetry for AI Defense security
    inspection operations, capturing the event_id and inspection results as
    span attributes under the gen_ai.security.* namespace.

    The primary attribute captured is `gen_ai.security.event_id`, which is
    essential for correlating security events in Splunk APM and GDI pipelines.

    Note: This instrumentation covers API mode only. Proxy mode (header-based
    event_id via x-cisco-ai-defense-tenant-api-key) is not yet supported.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Apply instrumentation to AI Defense SDK components."""
        global _handler

        # Initialize TelemetryHandler with tracer provider
        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            from opentelemetry import trace

            tracer_provider = trace.get_tracer_provider()

        meter_provider = kwargs.get("meter_provider")
        if not meter_provider:
            from opentelemetry import metrics

            meter_provider = metrics.get_meter_provider()

        _handler = TelemetryHandler(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )

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

    def _uninstrument(self, **kwargs):
        """Remove instrumentation from AI Defense SDK components."""
        unwrap("aidefense.runtime.chat_inspect.ChatInspectionClient", "inspect_prompt")
        unwrap(
            "aidefense.runtime.chat_inspect.ChatInspectionClient", "inspect_response"
        )
        unwrap(
            "aidefense.runtime.chat_inspect.ChatInspectionClient",
            "inspect_conversation",
        )
        unwrap("aidefense.runtime.http_inspect.HttpInspectionClient", "inspect_request")
        unwrap(
            "aidefense.runtime.http_inspect.HttpInspectionClient", "inspect_response"
        )
        unwrap(
            "aidefense.runtime.http_inspect.HttpInspectionClient",
            "inspect_request_from_http_library",
        )
        unwrap(
            "aidefense.runtime.http_inspect.HttpInspectionClient",
            "inspect_response_from_http_library",
        )


def _wrap_chat_inspect_prompt(wrapped, instance, args, kwargs):
    """
    Wrap ChatInspectionClient.inspect_prompt to create an LLMInvocation span.

    Captures the user prompt being inspected and the resulting security event_id.
    """
    try:
        handler = _handler
        prompt = kwargs.get("prompt") or (args[0] if args else "")

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=[
                InputMessage(role="user", parts=[Text(content=str(prompt)[:1000])])
            ],
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_chat_inspect_response(wrapped, instance, args, kwargs):
    """
    Wrap ChatInspectionClient.inspect_response to create an LLMInvocation span.

    Captures the AI response being inspected and the resulting security event_id.
    """
    try:
        handler = _handler
        response = kwargs.get("response") or (args[0] if args else "")

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=[
                InputMessage(
                    role="assistant", parts=[Text(content=str(response)[:1000])]
                )
            ],
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_chat_inspect_conversation(wrapped, instance, args, kwargs):
    """
    Wrap ChatInspectionClient.inspect_conversation to create an LLMInvocation span.

    Captures the full conversation being inspected and the resulting security event_id.
    """
    try:
        handler = _handler
        messages = kwargs.get("messages") or (args[0] if args else [])

        # Convert AI Defense messages to InputMessage format
        input_msgs = []
        for msg in messages[:10]:  # Limit to 10 messages for span size
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = getattr(msg, "content", "")[:500]
            input_msgs.append(InputMessage(role=role, parts=[Text(content=content)]))

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=input_msgs,
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_http_inspect_request(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_request to create an LLMInvocation span.

    Captures HTTP request inspection with method and URL context.
    """
    try:
        handler = _handler
        method = kwargs.get("method") or (args[0] if args else "")
        url = kwargs.get("url") or (args[1] if len(args) > 1 else "")

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=[
                InputMessage(role="user", parts=[Text(content=f"{method} {url}")])
            ],
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_http_inspect_response(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_response to create an LLMInvocation span.

    Captures HTTP response inspection with status code and URL context.
    """
    try:
        handler = _handler
        status_code = kwargs.get("status_code") or (args[0] if args else 0)
        url = kwargs.get("url") or (args[1] if len(args) > 1 else "")

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=[
                InputMessage(
                    role="assistant",
                    parts=[Text(content=f"HTTP {status_code} from {url}")],
                )
            ],
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_http_inspect_request_from_library(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_request_from_http_library.

    Handles requests from HTTP libraries like `requests`.
    """
    try:
        handler = _handler
        http_request = kwargs.get("http_request") or (args[0] if args else None)

        # Extract method and URL from request object
        method = (
            getattr(http_request, "method", "UNKNOWN") if http_request else "UNKNOWN"
        )
        url = getattr(http_request, "url", "") if http_request else ""

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=[
                InputMessage(
                    role="user", parts=[Text(content=f"{method} {url}"[:1000])]
                )
            ],
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_http_inspect_response_from_library(wrapped, instance, args, kwargs):
    """
    Wrap HttpInspectionClient.inspect_response_from_http_library.

    Handles responses from HTTP libraries like `requests`.
    """
    try:
        handler = _handler
        http_response = kwargs.get("http_response") or (args[0] if args else None)

        # Extract status code and URL from response object
        status_code = getattr(http_response, "status_code", 0) if http_response else 0
        url = getattr(http_response, "url", "") if http_response else ""

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            server_address=_get_server_address(instance),
            operation="chat",
            system="aidefense",
            framework="aidefense",
            input_messages=[
                InputMessage(
                    role="assistant",
                    parts=[Text(content=f"HTTP {status_code} from {url}"[:1000])],
                )
            ],
        )

        handler.start_llm(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        try:
            _populate_invocation_from_result(invocation, result)
            handler.stop_llm(invocation)
        except Exception:
            pass

        return result
    except Exception as exc:
        try:
            handler.fail(invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _get_server_address(instance) -> Optional[str]:
    """Extract the server address from the client instance."""
    try:
        return getattr(instance.config, "runtime_base_url", None)
    except Exception:
        return None


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
