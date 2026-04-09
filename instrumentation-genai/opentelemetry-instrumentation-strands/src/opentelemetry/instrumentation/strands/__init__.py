# Copyright Splunk Inc.
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
OpenTelemetry Strands Agents Instrumentation

Wrapper-based instrumentation for Strands Agents SDK using splunk-otel-util-genai.
"""

import logging
import os
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler, get_telemetry_handler
from wrapt import wrap_function_wrapper

from .hooks import StrandsHookProvider
from .package import _instruments
from .version import __version__
from .wrappers import (
    restore_builtin_tracer,
    suppress_builtin_tracer,
    wrap_agent_call,
    wrap_agent_init,
    wrap_agent_invoke_async,
    wrap_bedrock_agentcore_app_entrypoint,
)

__all__ = ["StrandsInstrumentor", "__version__"]

_LOGGER = logging.getLogger(__name__)

# Environment variable to control built-in tracer suppression
_ENV_SUPPRESS_BUILTIN_TRACER = "OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER"

# Global handler instance (singleton)
_handler: Optional[TelemetryHandler] = None
# Global hook provider instance
_hook_provider: Optional[StrandsHookProvider] = None


class StrandsInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentation for Strands Agents SDK using splunk-otel-util-genai.

    This instrumentor provides standardized telemetry for:
    - Agent invocations (Agent.__call__, Agent.invoke_async)
    - LLM calls (via Strands hooks)
    - Tool calls (via Strands hooks)
    - BedrockAgentCoreApp entrypoint (optional)

    Configuration:
        OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER:
            If "true" (default), suppresses Strands' built-in OTel tracer to avoid
            double-tracing. Set to "false" to keep both Strands and instrumentation spans.

    Note: This instrumentation uses a hybrid approach:
    - Wrapt wrappers for Agent lifecycle spans
    - Strands hooks for LLM and tool call spans
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Apply instrumentation to Strands components."""
        global _handler, _hook_provider

        # Initialize TelemetryHandler with tracer provider
        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            from opentelemetry import trace

            tracer_provider = trace.get_tracer_provider()

        meter_provider = kwargs.get("meter_provider")
        if not meter_provider:
            from opentelemetry import metrics

            meter_provider = metrics.get_meter_provider()

        _handler = get_telemetry_handler(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )

        # Create hook provider
        _hook_provider = StrandsHookProvider(_handler)

        # Conditionally suppress built-in tracer
        suppress_tracer = os.getenv(_ENV_SUPPRESS_BUILTIN_TRACER, "true").lower()
        if suppress_tracer in ("true", "1", "yes", "on"):
            suppress_builtin_tracer()

        # Wrapper helper function
        def _safe_wrap(module: str, name: str, wrapper: Any) -> None:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except (ImportError, ModuleNotFoundError):
                _LOGGER.debug(
                    "Strands not importable while instrumenting (%s.%s); proceeding without wrapping.",
                    module,
                    name,
                    exc_info=True,
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to instrument Strands (%s.%s); proceeding without wrapping.",
                    module,
                    name,
                    exc_info=True,
                )

        # Wrap Agent.__init__ to inject hook provider
        _safe_wrap(
            "strands.agent.agent",
            "Agent.__init__",
            lambda wrapped, instance, args, kwargs: wrap_agent_init(
                wrapped, instance, args, kwargs, _hook_provider
            ),
        )

        # Wrap Agent.__call__ for sync invocation
        _safe_wrap(
            "strands.agent.agent",
            "Agent.__call__",
            lambda wrapped, instance, args, kwargs: wrap_agent_call(
                wrapped, instance, args, kwargs, _handler
            ),
        )

        # Wrap Agent.invoke_async for async invocation
        _safe_wrap(
            "strands.agent.agent",
            "Agent.invoke_async",
            lambda wrapped, instance, args, kwargs: wrap_agent_invoke_async(
                wrapped, instance, args, kwargs, _handler
            ),
        )

        # Optionally wrap BedrockAgentCoreApp.entrypoint
        _safe_wrap(
            "bedrock_agentcore",
            "BedrockAgentCoreApp.entrypoint",
            lambda wrapped, instance, args, kwargs: wrap_bedrock_agentcore_app_entrypoint(
                wrapped, instance, args, kwargs, _handler
            ),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation from Strands components."""

        # Restore built-in tracer if it was suppressed
        restore_builtin_tracer()

        # Unwrap helper function
        def _safe_unwrap(module: str, name: str) -> None:
            try:
                unwrap(module, name)
            except (ImportError, ModuleNotFoundError):
                _LOGGER.debug(
                    "Strands not importable while uninstrumenting (%s.%s); continuing cleanup.",
                    module,
                    name,
                    exc_info=True,
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to uninstrument Strands (%s.%s); continuing cleanup.",
                    module,
                    name,
                    exc_info=True,
                )

        # Unwrap all wrapped methods
        _safe_unwrap("strands.agent.agent", "Agent.__init__")
        _safe_unwrap("strands.agent.agent", "Agent.__call__")
        _safe_unwrap("strands.agent.agent", "Agent.invoke_async")
        _safe_unwrap("bedrock_agentcore", "BedrockAgentCoreApp.entrypoint")
