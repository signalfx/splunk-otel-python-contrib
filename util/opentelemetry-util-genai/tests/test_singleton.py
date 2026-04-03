from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)


def test_telemetry_handler_constructor_returns_singleton():
    TelemetryHandler._reset_for_testing()
    handler_ctor = TelemetryHandler()
    handler_factory = get_telemetry_handler()

    assert handler_ctor is handler_factory
    assert TelemetryHandler() is handler_factory

    TelemetryHandler._reset_for_testing()
