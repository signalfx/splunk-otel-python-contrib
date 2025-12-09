from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.instrumentation.llamaindex.config import Config
from opentelemetry.instrumentation.llamaindex.callback_handler import (
    LlamaindexCallbackHandler,
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

        llamaindexCallBackHandler = LlamaindexCallbackHandler(
            telemetry_handler=self._telemetry_handler
        )

        wrap_function_wrapper(
            module="llama_index.core.callbacks.base",
            name="CallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(llamaindexCallBackHandler),
        )

    def _uninstrument(self, **kwargs):
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
