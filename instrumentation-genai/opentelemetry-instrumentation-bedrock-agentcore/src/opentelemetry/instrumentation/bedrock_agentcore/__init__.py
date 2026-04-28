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
OpenTelemetry Bedrock AgentCore Instrumentation

Wrapper-based instrumentation for AWS Bedrock AgentCore using splunk-otel-util-genai.
"""

import logging
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler, get_telemetry_handler
from wrapt import wrap_function_wrapper

from .utils import is_content_enabled
from .browser_wrappers import (
    wrap_browser_get_session,
    wrap_browser_operation,
    wrap_browser_release_control,
    wrap_browser_start,
    wrap_browser_stop,
    wrap_browser_take_control,
)
from .code_interpreter_wrappers import (
    wrap_code_interpreter_execute,
    wrap_code_interpreter_install_packages,
    wrap_code_interpreter_operation,
    wrap_code_interpreter_start,
    wrap_code_interpreter_stop,
    wrap_code_interpreter_upload_file,
)
from .entrypoint_wrappers import wrap_bedrock_agentcore_app_entrypoint
from .memory_wrappers import (
    wrap_memory_create_blob_event,
    wrap_memory_create_event,
    wrap_memory_list_events,
    wrap_memory_operation,
    wrap_memory_retrieve,
)
from .package import _instruments
from .version import __version__

__all__ = ["BedrockAgentCoreInstrumentor", "__version__"]

_LOGGER = logging.getLogger(__name__)

# Global handler instance (singleton)
_handler: Optional[TelemetryHandler] = None


class BedrockAgentCoreInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentation for AWS Bedrock AgentCore components.

    This instrumentor provides standardized telemetry for:
    - BedrockAgentCoreApp entrypoint → Workflow spans
    - Memory operations (MemoryClient) → RetrievalInvocation/ToolCall spans
    - Code Interpreter (CodeInterpreter) → ToolCall spans
    - Browser automation (BrowserClient) → ToolCall spans

    All spans are properly nested with correct parent-child relationships and include
    rich attributes about the operation.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Apply instrumentation to Bedrock AgentCore components."""
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

        _handler = get_telemetry_handler(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )

        capture_content = is_content_enabled()

        # Wrapper helper function
        def _safe_wrap(module: str, name: str, wrapper: Any) -> None:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except (ImportError, ModuleNotFoundError):
                _LOGGER.debug(
                    "Bedrock AgentCore not importable while instrumenting (%s.%s); proceeding without wrapping.",
                    module,
                    name,
                    exc_info=True,
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to instrument Bedrock AgentCore (%s.%s); proceeding without wrapping.",
                    module,
                    name,
                    exc_info=True,
                )

        # Wrap BedrockAgentCoreApp.entrypoint
        _safe_wrap(
            "bedrock_agentcore",
            "BedrockAgentCoreApp.entrypoint",
            lambda wrapped,
            instance,
            args,
            kwargs: wrap_bedrock_agentcore_app_entrypoint(
                wrapped, instance, args, kwargs, _handler
            ),
        )

        # Wrap MemoryClient operations
        _safe_wrap(
            "bedrock_agentcore.memory.client",
            "MemoryClient.retrieve_memories",
            lambda wrapped, instance, args, kwargs: wrap_memory_retrieve(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.memory.client",
            "MemoryClient.create_event",
            lambda wrapped, instance, args, kwargs: wrap_memory_create_event(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.memory.client",
            "MemoryClient.create_blob_event",
            lambda wrapped, instance, args, kwargs: wrap_memory_create_blob_event(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.memory.client",
            "MemoryClient.list_events",
            lambda wrapped, instance, args, kwargs: wrap_memory_list_events(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )

        # Wrap CodeInterpreter operations
        _safe_wrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.start",
            lambda wrapped, instance, args, kwargs: wrap_code_interpreter_start(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.stop",
            lambda wrapped, instance, args, kwargs: wrap_code_interpreter_stop(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.execute_code",
            lambda wrapped, instance, args, kwargs: wrap_code_interpreter_execute(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.install_packages",
            lambda wrapped,
            instance,
            args,
            kwargs: wrap_code_interpreter_install_packages(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.upload_file",
            lambda wrapped, instance, args, kwargs: wrap_code_interpreter_upload_file(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )

        # Wrap BrowserClient operations
        _safe_wrap(
            "bedrock_agentcore.tools.browser_client",
            "BrowserClient.start",
            lambda wrapped, instance, args, kwargs: wrap_browser_start(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.browser_client",
            "BrowserClient.stop",
            lambda wrapped, instance, args, kwargs: wrap_browser_stop(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.browser_client",
            "BrowserClient.take_control",
            lambda wrapped, instance, args, kwargs: wrap_browser_take_control(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.browser_client",
            "BrowserClient.release_control",
            lambda wrapped, instance, args, kwargs: wrap_browser_release_control(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )
        _safe_wrap(
            "bedrock_agentcore.tools.browser_client",
            "BrowserClient.get_session",
            lambda wrapped, instance, args, kwargs: wrap_browser_get_session(
                wrapped, instance, args, kwargs, _handler, capture_content
            ),
        )

        # Additional MemoryClient operations
        for method in [
            "create_memory",
            "create_memory_and_wait",
            "create_or_get_memory",
            "delete_memory",
            "delete_memory_and_wait",
            "get_memory_status",
            "list_memories",
            "wait_for_memories",
            "save_conversation",
            "fork_conversation",
            "get_conversation_tree",
            "get_last_k_turns",
            "list_branch_events",
            "list_branches",
            "merge_branch_context",
            "process_turn_with_llm",
            "add_strategy",
            "add_episodic_strategy",
            "add_episodic_strategy_and_wait",
            "add_semantic_strategy",
            "add_semantic_strategy_and_wait",
            "add_summary_strategy",
            "add_summary_strategy_and_wait",
            "add_user_preference_strategy",
            "add_user_preference_strategy_and_wait",
            "add_custom_episodic_strategy",
            "add_custom_episodic_strategy_and_wait",
            "add_custom_semantic_strategy",
            "add_custom_semantic_strategy_and_wait",
            "delete_strategy",
            "modify_strategy",
            "get_memory_strategies",
            "update_memory_strategies",
            "update_memory_strategies_and_wait",
        ]:
            _safe_wrap(
                "bedrock_agentcore.memory.client",
                f"MemoryClient.{method}",
                lambda wrapped, instance, args, kwargs, m=method: wrap_memory_operation(
                    m
                )(wrapped, instance, args, kwargs, _handler, capture_content),
            )

        # Additional CodeInterpreter operations
        for method in [
            "download_file",
            "download_files",
            "upload_files",
            "get_session",
            "list_sessions",
            "execute_command",
            "clear_context",
            "invoke",
            "create_code_interpreter",
            "delete_code_interpreter",
            "get_code_interpreter",
            "list_code_interpreters",
        ]:
            _safe_wrap(
                "bedrock_agentcore.tools.code_interpreter_client",
                f"CodeInterpreter.{method}",
                lambda wrapped,
                instance,
                args,
                kwargs,
                m=method: wrap_code_interpreter_operation(m)(
                    wrapped, instance, args, kwargs, _handler, capture_content
                ),
            )

        # Additional BrowserClient operations
        for method in [
            "list_sessions",
            "create_browser",
            "delete_browser",
            "get_browser",
            "list_browsers",
            "generate_live_view_url",
            "generate_ws_headers",
            "update_stream",
        ]:
            _safe_wrap(
                "bedrock_agentcore.tools.browser_client",
                f"BrowserClient.{method}",
                lambda wrapped,
                instance,
                args,
                kwargs,
                m=method: wrap_browser_operation(m)(
                    wrapped, instance, args, kwargs, _handler, capture_content
                ),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation from Bedrock AgentCore components."""

        # Unwrap helper function
        def _safe_unwrap(module: str, name: str) -> None:
            try:
                unwrap(module, name)
            except (ImportError, ModuleNotFoundError):
                _LOGGER.debug(
                    "Bedrock AgentCore not importable while uninstrumenting (%s.%s); continuing cleanup.",
                    module,
                    name,
                    exc_info=True,
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to uninstrument Bedrock AgentCore (%s.%s); continuing cleanup.",
                    module,
                    name,
                    exc_info=True,
                )

        # Unwrap all wrapped methods
        _safe_unwrap("bedrock_agentcore", "BedrockAgentCoreApp.entrypoint")

        # Unwrap MemoryClient methods
        _safe_unwrap(
            "bedrock_agentcore.memory.client", "MemoryClient.retrieve_memories"
        )
        _safe_unwrap("bedrock_agentcore.memory.client", "MemoryClient.create_event")
        _safe_unwrap(
            "bedrock_agentcore.memory.client",
            "MemoryClient.create_blob_event",
        )
        _safe_unwrap("bedrock_agentcore.memory.client", "MemoryClient.list_events")

        # Unwrap CodeInterpreter methods
        _safe_unwrap(
            "bedrock_agentcore.tools.code_interpreter_client", "CodeInterpreter.start"
        )
        _safe_unwrap(
            "bedrock_agentcore.tools.code_interpreter_client", "CodeInterpreter.stop"
        )
        _safe_unwrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.execute_code",
        )
        _safe_unwrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.install_packages",
        )
        _safe_unwrap(
            "bedrock_agentcore.tools.code_interpreter_client",
            "CodeInterpreter.upload_file",
        )

        # Unwrap BrowserClient methods
        _safe_unwrap("bedrock_agentcore.tools.browser_client", "BrowserClient.start")
        _safe_unwrap("bedrock_agentcore.tools.browser_client", "BrowserClient.stop")
        _safe_unwrap(
            "bedrock_agentcore.tools.browser_client", "BrowserClient.take_control"
        )
        _safe_unwrap(
            "bedrock_agentcore.tools.browser_client", "BrowserClient.release_control"
        )
        _safe_unwrap(
            "bedrock_agentcore.tools.browser_client", "BrowserClient.get_session"
        )

        # Unwrap additional MemoryClient operations
        for method in [
            "create_memory",
            "create_memory_and_wait",
            "create_or_get_memory",
            "delete_memory",
            "delete_memory_and_wait",
            "get_memory_status",
            "list_memories",
            "wait_for_memories",
            "save_conversation",
            "fork_conversation",
            "get_conversation_tree",
            "get_last_k_turns",
            "list_branch_events",
            "list_branches",
            "merge_branch_context",
            "process_turn_with_llm",
            "add_strategy",
            "add_episodic_strategy",
            "add_episodic_strategy_and_wait",
            "add_semantic_strategy",
            "add_semantic_strategy_and_wait",
            "add_summary_strategy",
            "add_summary_strategy_and_wait",
            "add_user_preference_strategy",
            "add_user_preference_strategy_and_wait",
            "add_custom_episodic_strategy",
            "add_custom_episodic_strategy_and_wait",
            "add_custom_semantic_strategy",
            "add_custom_semantic_strategy_and_wait",
            "delete_strategy",
            "modify_strategy",
            "get_memory_strategies",
            "update_memory_strategies",
            "update_memory_strategies_and_wait",
        ]:
            _safe_unwrap("bedrock_agentcore.memory.client", f"MemoryClient.{method}")

        # Unwrap additional CodeInterpreter operations
        for method in [
            "download_file",
            "download_files",
            "upload_files",
            "get_session",
            "list_sessions",
            "execute_command",
            "clear_context",
            "invoke",
            "create_code_interpreter",
            "delete_code_interpreter",
            "get_code_interpreter",
            "list_code_interpreters",
        ]:
            _safe_unwrap(
                "bedrock_agentcore.tools.code_interpreter_client",
                f"CodeInterpreter.{method}",
            )

        # Unwrap additional BrowserClient operations
        for method in [
            "list_sessions",
            "create_browser",
            "delete_browser",
            "get_browser",
            "list_browsers",
            "generate_live_view_url",
            "generate_ws_headers",
            "update_stream",
        ]:
            _safe_unwrap(
                "bedrock_agentcore.tools.browser_client", f"BrowserClient.{method}"
            )
