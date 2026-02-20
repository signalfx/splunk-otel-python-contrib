"""Tests for TelemetryHandler singleton behavior in CrewAI instrumentation."""

from unittest import mock

import opentelemetry.instrumentation.crewai.instrumentation as crewai_module
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor


class TestTelemetryHandlerSingleton:
    """Test suite for TelemetryHandler singleton behavior."""

    def test_handler_is_none_before_instrumentation(self):
        """Handler should be None before instrument() is called."""
        assert crewai_module._handler is None

    def test_handler_is_set_after_instrumentation(
        self, tracer_provider, meter_provider
    ):
        """Handler should be set after instrument() is called."""
        instrumentor = CrewAIInstrumentor()

        # Mock the wrap_function_wrapper to avoid importing crewai
        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

        assert crewai_module._handler is not None

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_handler_is_telemetry_handler_instance(
        self, tracer_provider, meter_provider
    ):
        """Handler should be an instance of TelemetryHandler."""
        from opentelemetry.util.genai.handler import TelemetryHandler

        instrumentor = CrewAIInstrumentor()

        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

        assert isinstance(crewai_module._handler, TelemetryHandler)

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_handler_receives_tracer_provider(self, tracer_provider, meter_provider):
        """Handler should be initialized with the provided tracer_provider."""
        instrumentor = CrewAIInstrumentor()

        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            with mock.patch(
                "opentelemetry.instrumentation.crewai.instrumentation.get_telemetry_handler"
            ) as mock_get_handler:
                instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                )

                mock_get_handler.assert_called_once_with(
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                )

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_handler_uses_default_tracer_provider_when_not_provided(
        self, meter_provider
    ):
        """Handler should use default tracer provider when not explicitly provided."""
        instrumentor = CrewAIInstrumentor()

        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            with mock.patch(
                "opentelemetry.trace.get_tracer_provider"
            ) as mock_get_tracer:
                mock_tracer_provider = mock.MagicMock()
                mock_get_tracer.return_value = mock_tracer_provider

                with mock.patch(
                    "opentelemetry.instrumentation.crewai.instrumentation.get_telemetry_handler"
                ) as mock_get_handler:
                    instrumentor.instrument(meter_provider=meter_provider)

                    mock_get_handler.assert_called_once()
                    call_kwargs = mock_get_handler.call_args[1]
                    assert call_kwargs["tracer_provider"] == mock_tracer_provider

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_handler_uses_default_meter_provider_when_not_provided(
        self, tracer_provider
    ):
        """Handler should use default meter provider when not explicitly provided."""
        instrumentor = CrewAIInstrumentor()

        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            with mock.patch(
                "opentelemetry.metrics.get_meter_provider"
            ) as mock_get_meter:
                mock_meter_provider = mock.MagicMock()
                mock_get_meter.return_value = mock_meter_provider

                with mock.patch(
                    "opentelemetry.instrumentation.crewai.instrumentation.get_telemetry_handler"
                ) as mock_get_handler:
                    instrumentor.instrument(tracer_provider=tracer_provider)

                    mock_get_handler.assert_called_once()
                    call_kwargs = mock_get_handler.call_args[1]
                    assert call_kwargs["meter_provider"] == mock_meter_provider

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_reinstrumentation_is_prevented_by_base_class(
        self, tracer_provider, meter_provider, caplog
    ):
        """Re-instrumenting should be prevented by the base instrumentor class."""
        import logging

        instrumentor = CrewAIInstrumentor()

        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

            first_handler = crewai_module._handler

            # Try to re-instrument - should be prevented by base class
            with caplog.at_level(logging.WARNING):
                instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                )

            # Handler should be the same (no re-instrumentation occurred)
            assert crewai_module._handler is first_handler

            # Warning should be logged by the base class
            assert any(
                "already instrumented" in record.message.lower()
                for record in caplog.records
            )

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_wrapper_functions_use_global_handler(self, stub_handler):
        """Wrapper functions should use the global _handler instance."""
        # Set up global handler
        crewai_module._handler = stub_handler

        # Test _wrap_crew_kickoff
        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"
        mock_wrapped = mock.MagicMock(return_value=mock.MagicMock(raw="result"))

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {"inputs": {}})

        assert len(stub_handler.started_workflows) == 1
        assert len(stub_handler.stopped_workflows) == 1
