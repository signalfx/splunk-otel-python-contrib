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
from typing import Any, Callable, Collection, Iterator, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler, get_telemetry_handler
from wrapt import wrap_function_wrapper

from .utils import is_content_enabled, is_instrumentation_enabled
from .browser_wrappers import (
    wrap_browser_create_browser,
    wrap_browser_generate_live_view_url,
    wrap_browser_generate_ws_headers,
    wrap_browser_get_browser,
    wrap_browser_get_session,
    wrap_browser_list_browsers,
    wrap_browser_operation,
    wrap_browser_release_control,
    wrap_browser_start,
    wrap_browser_stop,
    wrap_browser_take_control,
    wrap_browser_update_stream,
)
from .code_interpreter_wrappers import (
    wrap_code_interpreter_clear_context,
    wrap_code_interpreter_create,
    wrap_code_interpreter_download_file,
    wrap_code_interpreter_execute,
    wrap_code_interpreter_execute_command,
    wrap_code_interpreter_get,
    wrap_code_interpreter_install_packages,
    wrap_code_interpreter_list,
    wrap_code_interpreter_operation,
    wrap_code_interpreter_start,
    wrap_code_interpreter_stop,
    wrap_code_interpreter_upload_file,
)
from .entrypoint_wrappers import wrap_bedrock_agentcore_app_entrypoint
from .memory_wrappers import (
    wrap_memory_conversation_operation,
    wrap_memory_create_blob_event,
    wrap_memory_create_event,
    wrap_memory_list_events,
    wrap_memory_operation,
    wrap_memory_retrieve,
    wrap_memory_session_async_operation,
    wrap_memory_session_operation,
    wrap_memory_session_search_long_term_memories,
)
from .package import _instruments
from .version import __version__

__all__ = ["BedrockAgentCoreInstrumentor", "__version__"]

_LOGGER = logging.getLogger(__name__)

_AGENTCORE_MODULE = "bedrock_agentcore"
_MEMORY_MODULE = "bedrock_agentcore.memory.client"
_MEMORY_SESSION_MODULE = "bedrock_agentcore.memory.session"
_CODE_INTERPRETER_MODULE = "bedrock_agentcore.tools.code_interpreter_client"
_BROWSER_MODULE = "bedrock_agentcore.tools.browser_client"

_ENTRYPOINT_WRAP_TARGETS: tuple[tuple[str, str, Callable[..., Any]], ...] = (
    (
        _AGENTCORE_MODULE,
        "BedrockAgentCoreApp.entrypoint",
        wrap_bedrock_agentcore_app_entrypoint,
    ),
)

_CONTENT_WRAP_TARGETS: tuple[tuple[str, str, Callable[..., Any]], ...] = (
    (_MEMORY_MODULE, "MemoryClient.retrieve_memories", wrap_memory_retrieve),
    (_MEMORY_MODULE, "MemoryClient.create_event", wrap_memory_create_event),
    (
        _MEMORY_MODULE,
        "MemoryClient.create_blob_event",
        wrap_memory_create_blob_event,
    ),
    (_MEMORY_MODULE, "MemoryClient.list_events", wrap_memory_list_events),
    (
        _MEMORY_SESSION_MODULE,
        "MemorySessionManager.search_long_term_memories",
        wrap_memory_session_search_long_term_memories,
    ),
    (
        _MEMORY_SESSION_MODULE,
        "MemorySessionManager.process_turn_with_llm_async",
        wrap_memory_session_async_operation("process_turn_with_llm_async"),
    ),
    (
        _MEMORY_MODULE,
        "MemoryClient.process_turn_with_llm",
        wrap_memory_conversation_operation("process_turn_with_llm"),
    ),
    (
        _MEMORY_MODULE,
        "MemoryClient.save_conversation",
        wrap_memory_conversation_operation("save_conversation"),
    ),
    (
        _MEMORY_MODULE,
        "MemoryClient.fork_conversation",
        wrap_memory_conversation_operation("fork_conversation"),
    ),
    (
        _MEMORY_MODULE,
        "MemoryClient.get_last_k_turns",
        wrap_memory_conversation_operation("get_last_k_turns"),
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.start",
        wrap_code_interpreter_start,
    ),
    (_CODE_INTERPRETER_MODULE, "CodeInterpreter.stop", wrap_code_interpreter_stop),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.execute_code",
        wrap_code_interpreter_execute,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.install_packages",
        wrap_code_interpreter_install_packages,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.upload_file",
        wrap_code_interpreter_upload_file,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.upload_files",
        wrap_code_interpreter_upload_file,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.download_file",
        wrap_code_interpreter_download_file,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.download_files",
        wrap_code_interpreter_download_file,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.execute_command",
        wrap_code_interpreter_execute_command,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.clear_context",
        wrap_code_interpreter_clear_context,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.create_code_interpreter",
        wrap_code_interpreter_create,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.get_code_interpreter",
        wrap_code_interpreter_get,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter.list_code_interpreters",
        wrap_code_interpreter_list,
    ),
    (_BROWSER_MODULE, "BrowserClient.start", wrap_browser_start),
    (_BROWSER_MODULE, "BrowserClient.stop", wrap_browser_stop),
    (_BROWSER_MODULE, "BrowserClient.take_control", wrap_browser_take_control),
    (_BROWSER_MODULE, "BrowserClient.release_control", wrap_browser_release_control),
    (_BROWSER_MODULE, "BrowserClient.get_session", wrap_browser_get_session),
    (
        _BROWSER_MODULE,
        "BrowserClient.generate_ws_headers",
        wrap_browser_generate_ws_headers,
    ),
    (
        _BROWSER_MODULE,
        "BrowserClient.generate_live_view_url",
        wrap_browser_generate_live_view_url,
    ),
    (_BROWSER_MODULE, "BrowserClient.create_browser", wrap_browser_create_browser),
    (_BROWSER_MODULE, "BrowserClient.get_browser", wrap_browser_get_browser),
    (_BROWSER_MODULE, "BrowserClient.list_browsers", wrap_browser_list_browsers),
    (_BROWSER_MODULE, "BrowserClient.update_stream", wrap_browser_update_stream),
)

_MEMORY_OPERATION_METHODS = (
    "create_memory",
    "create_memory_and_wait",
    "create_or_get_memory",
    "delete_memory",
    "delete_memory_and_wait",
    "get_memory_status",
    "list_memories",
    "wait_for_memories",
    "get_conversation_tree",
    "list_branch_events",
    "list_branches",
    "merge_branch_context",
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
)

_MEMORY_SESSION_OPERATION_METHODS = (
    "create_memory_session",
    "process_turn_with_llm",
    "add_turns",
    "fork_conversation",
    "list_events",
    "list_branches",
    "get_last_k_turns",
    "get_event",
    "delete_event",
    "list_long_term_memory_records",
    "list_actors",
    "list_actor_sessions",
    "get_memory_record",
    "delete_memory_record",
    "delete_all_long_term_memories_in_namespace",
)

_CODE_INTERPRETER_OPERATION_METHODS = (
    "get_session",
    "list_sessions",
    "delete_code_interpreter",
)

_BROWSER_OPERATION_METHODS = (
    "list_sessions",
    "delete_browser",
)

_GENERIC_WRAP_TARGETS: tuple[
    tuple[str, str, tuple[str, ...], Callable[[str], Callable[..., Any]]], ...
] = (
    (
        _MEMORY_MODULE,
        "MemoryClient",
        _MEMORY_OPERATION_METHODS,
        wrap_memory_operation,
    ),
    (
        _MEMORY_SESSION_MODULE,
        "MemorySessionManager",
        _MEMORY_SESSION_OPERATION_METHODS,
        wrap_memory_session_operation,
    ),
    (
        _CODE_INTERPRETER_MODULE,
        "CodeInterpreter",
        _CODE_INTERPRETER_OPERATION_METHODS,
        wrap_code_interpreter_operation,
    ),
    (
        _BROWSER_MODULE,
        "BrowserClient",
        _BROWSER_OPERATION_METHODS,
        wrap_browser_operation,
    ),
)


def _with_handler(
    wrapper: Callable[..., Any],
    handler: TelemetryHandler,
    capture_content: bool,
) -> Callable[[Any, Any, tuple, dict], Any]:
    def _wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        return wrapper(wrapped, instance, args, kwargs, handler, capture_content)

    return _wrapper


def _with_entrypoint_handler(
    wrapper: Callable[..., Any],
    handler: TelemetryHandler,
    capture_content: bool,
) -> Callable[[Any, Any, tuple, dict], Any]:
    def _wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        return wrapper(wrapped, instance, args, kwargs, handler, capture_content)

    return _wrapper


def _iter_wrap_specs(
    handler: TelemetryHandler, capture_content: bool
) -> Iterator[tuple[str, str, Callable[[Any, Any, tuple, dict], Any]]]:
    for module, name, wrapper in _ENTRYPOINT_WRAP_TARGETS:
        yield module, name, _with_entrypoint_handler(wrapper, handler, capture_content)

    for module, name, wrapper in _CONTENT_WRAP_TARGETS:
        yield module, name, _with_handler(wrapper, handler, capture_content)

    for module, class_name, methods, operation_wrapper in _GENERIC_WRAP_TARGETS:
        for method in methods:
            yield (
                module,
                f"{class_name}.{method}",
                _with_handler(operation_wrapper(method), handler, capture_content),
            )


def _iter_wrap_targets() -> Iterator[tuple[str, str]]:
    for module, name, _wrapper in _ENTRYPOINT_WRAP_TARGETS:
        yield module, name

    for module, name, _wrapper in _CONTENT_WRAP_TARGETS:
        yield module, name

    for module, class_name, methods, _operation_wrapper in _GENERIC_WRAP_TARGETS:
        for method in methods:
            yield module, f"{class_name}.{method}"


class BedrockAgentCoreInstrumentor(BaseInstrumentor):
    def __init__(self) -> None:
        super().__init__()
        self._handler: Optional[TelemetryHandler] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not is_instrumentation_enabled():
            _LOGGER.debug("Bedrock AgentCore instrumentation is disabled")
            return

        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            from opentelemetry import trace

            tracer_provider = trace.get_tracer_provider()

        meter_provider = kwargs.get("meter_provider")
        if not meter_provider:
            from opentelemetry import metrics

            meter_provider = metrics.get_meter_provider()

        logger_provider = kwargs.get("logger_provider")

        self._handler = get_telemetry_handler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        capture_content = is_content_enabled()

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

        for module, name, wrapper in _iter_wrap_specs(self._handler, capture_content):
            _safe_wrap(module, name, wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        self._handler = None

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

        for module, name in _iter_wrap_targets():
            _safe_unwrap(module, name)
