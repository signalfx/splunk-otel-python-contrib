from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.instrumentation.llamaindex.config import Config
from opentelemetry.instrumentation.llamaindex.callback_handler import (
    LlamaindexCallbackHandler,
)
from opentelemetry.instrumentation.llamaindex.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.instrumentation.llamaindex.event_handler import (
    LlamaindexEventHandler,
    TTFCTracker,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.llamaindex.workflow_instrumentation import (
    wrap_agent_run,
)
from wrapt import wrap_function_wrapper

_instruments = ("llama-index-core >= 0.14.0",)


class LlamaindexInstrumentor(BaseInstrumentor):
    def __init__(
        self,
        exception_logger=None,
        disable_trace_context_propagation=False,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config._exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes
        self._disable_trace_context_propagation = disable_trace_context_propagation
        self._telemetry_handler = None

    def instrumentation_dependencies(self):
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._telemetry_handler = get_telemetry_handler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Create shared TTFC tracker and invocation manager
        ttfc_tracker = TTFCTracker()
        invocation_manager = _InvocationManager()
        invocation_manager.set_ttfc_tracker(ttfc_tracker)

        llamaindexCallBackHandler = LlamaindexCallbackHandler(
            telemetry_handler=self._telemetry_handler,
            invocation_manager=invocation_manager,
        )

        # Create and register event handler for TTFC tracking
        event_handler = LlamaindexEventHandler(ttfc_tracker=ttfc_tracker)
        self._event_handler = event_handler
        try:
            from llama_index.core.instrumentation import get_dispatcher

            dispatcher = get_dispatcher()
            dispatcher.add_event_handler(event_handler)
            self._dispatcher = dispatcher
        except Exception:
            # Event system might not be available in older versions
            self._dispatcher = None

        wrap_function_wrapper(
            module="llama_index.core.callbacks.base",
            name="CallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(llamaindexCallBackHandler),
        )

        # Instrument workflow-based agents by wrapping BaseWorkflowAgent (and async variant).
        # This covers ReActAgent, FunctionAgent, CodeActAgent, and subclasses.
        for method_name in ["run", "arun"]:
            try:
                wrap_function_wrapper(
                    module="llama_index.core.agent.workflow.base_agent",
                    name=f"BaseWorkflowAgent.{method_name}",
                    wrapper=wrap_agent_run,
                )
            except Exception:
                # Module/class/method might not be available in some versions.
                pass

        # Instrument MultiAgentWorkflow explicitly (sync/async), which does not inherit
        # from BaseWorkflowAgent in all versions.
        for method_name in ["run", "arun"]:
            try:
                wrap_function_wrapper(
                    module="llama_index.core.agent.workflow.multi_agent_workflow",
                    name=f"MultiAgentWorkflow.{method_name}",
                    wrapper=wrap_agent_run,
                )
            except Exception:
                # MultiAgentWorkflow might not be available or importable.
                pass

        # Instrument AgentWorkflow (sync/async) for workflow-level orchestration spans.
        for method_name in ["run", "arun"]:
            try:
                wrap_function_wrapper(
                    module="llama_index.core.agent.workflow.multi_agent_workflow",
                    name=f"AgentWorkflow.{method_name}",
                    wrapper=wrap_agent_run,
                )
            except Exception:
                # AgentWorkflow might not be available or importable.
                pass

    def _uninstrument(self, **kwargs):
        unwrap("llama_index.core.callbacks.base", "CallbackManager.__init__")
        # Remove event handler from dispatcher to avoid duplicate TTFC measurements on re-instrumentation
        if (
            hasattr(self, "_dispatcher")
            and self._dispatcher
            and hasattr(self, "_event_handler")
        ):
            try:
                self._dispatcher.event_handlers = [
                    h
                    for h in self._dispatcher.event_handlers
                    if h is not self._event_handler
                ]
            except Exception:
                pass


class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_handler: "LlamaindexCallbackHandler"):
        self._callback_handler = callback_handler

    def __call__(self, wrapped, instance, args, kwargs) -> None:
        wrapped(*args, **kwargs)
        # LlamaIndex uses 'handlers' instead of 'inheritable_handlers'
        for handler in instance.handlers:
            if isinstance(handler, type(self._callback_handler)):
                break
        else:
            self._callback_handler._callback_manager = instance
            instance.add_handler(self._callback_handler)
