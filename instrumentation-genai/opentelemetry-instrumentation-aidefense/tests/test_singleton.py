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

"""Tests for TelemetryHandler singleton behavior in AI Defense instrumentation."""

from unittest import mock

import opentelemetry.instrumentation.aidefense.instrumentation as aidefense_module
from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor
from opentelemetry.util.genai.handler import TelemetryHandler, get_telemetry_handler


class TestTelemetryHandlerSingleton:
    """Test suite for TelemetryHandler singleton behavior."""

    def test_handler_is_none_before_instrumentation(self):
        """Handler should be None before instrument() is called."""
        assert aidefense_module._handler is None

    def test_handler_is_set_after_instrumentation(
        self, tracer_provider, meter_provider
    ):
        """Handler should be set after instrument() is called."""
        instrumentor = AIDefenseInstrumentor()

        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        assert aidefense_module._handler is not None

        instrumentor.uninstrument()

    def test_handler_is_telemetry_handler_instance(
        self, tracer_provider, meter_provider
    ):
        """Handler should be an instance of TelemetryHandler."""
        instrumentor = AIDefenseInstrumentor()

        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        assert isinstance(aidefense_module._handler, TelemetryHandler)

        instrumentor.uninstrument()

    def test_handler_receives_providers(self, tracer_provider, meter_provider):
        """Handler should be initialized with the provided providers."""
        instrumentor = AIDefenseInstrumentor()

        with mock.patch(
            "opentelemetry.instrumentation.aidefense.instrumentation.get_telemetry_handler"
        ) as mock_get_handler:
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

            mock_get_handler.assert_called_once_with(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

        instrumentor.uninstrument()

    def test_handler_uses_default_tracer_provider_when_not_provided(
        self, meter_provider
    ):
        """Handler should use default tracer provider when not explicitly provided."""
        instrumentor = AIDefenseInstrumentor()

        with mock.patch("opentelemetry.trace.get_tracer_provider") as mock_get_tracer:
            mock_tracer_provider = mock.MagicMock()
            mock_get_tracer.return_value = mock_tracer_provider

            with mock.patch(
                "opentelemetry.instrumentation.aidefense.instrumentation.get_telemetry_handler"
            ) as mock_get_handler:
                instrumentor.instrument(meter_provider=meter_provider)

                mock_get_handler.assert_called_once()
                call_kwargs = mock_get_handler.call_args[1]
                assert call_kwargs["tracer_provider"] == mock_tracer_provider

        instrumentor.uninstrument()

    def test_handler_uses_default_meter_provider_when_not_provided(
        self, tracer_provider
    ):
        """Handler should use default meter provider when not explicitly provided."""
        instrumentor = AIDefenseInstrumentor()

        with mock.patch("opentelemetry.metrics.get_meter_provider") as mock_get_meter:
            mock_meter_provider = mock.MagicMock()
            mock_get_meter.return_value = mock_meter_provider

            with mock.patch(
                "opentelemetry.instrumentation.aidefense.instrumentation.get_telemetry_handler"
            ) as mock_get_handler:
                instrumentor.instrument(tracer_provider=tracer_provider)

                mock_get_handler.assert_called_once()
                call_kwargs = mock_get_handler.call_args[1]
                assert call_kwargs["meter_provider"] == mock_meter_provider

        instrumentor.uninstrument()

    def test_reinstrumentation_is_prevented_by_base_class(
        self, tracer_provider, meter_provider, caplog
    ):
        """Re-instrumenting should be prevented by the base instrumentor class."""
        import logging

        instrumentor = AIDefenseInstrumentor()

        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        first_handler = aidefense_module._handler

        with caplog.at_level(logging.WARNING):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

        assert aidefense_module._handler is first_handler

        assert any(
            "already instrumented" in record.message.lower()
            for record in caplog.records
        )

        instrumentor.uninstrument()


class TestCrossHandlerPropagation:
    """Tests demonstrating that aidefense shares the singleton handler."""

    def test_singleton_identity_across_calls(self):
        """get_telemetry_handler() returns the same instance on repeated calls."""
        handler1 = get_telemetry_handler()
        handler2 = get_telemetry_handler()
        assert handler1 is handler2

    def test_singleton_constructor_and_factory(self):
        """TelemetryHandler() and get_telemetry_handler() return the same object."""
        handler1 = get_telemetry_handler()
        handler2 = TelemetryHandler()
        assert handler1 is handler2

    def test_aidefense_uses_shared_singleton(self, tracer_provider, meter_provider):
        """After instrumentation, aidefense's _handler IS the shared singleton."""
        instrumentor = AIDefenseInstrumentor()

        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        assert aidefense_module._handler is get_telemetry_handler()

        instrumentor.uninstrument()

    def test_handler_state_propagated(self, tracer_provider, meter_provider):
        """State added to the singleton before instrumentation is visible via aidefense's _handler."""
        singleton = get_telemetry_handler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        sentinel_callback = lambda invocation: None  # noqa: E731
        singleton.register_completion_callback(sentinel_callback)

        instrumentor = AIDefenseInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        assert aidefense_module._handler is singleton
        assert sentinel_callback in aidefense_module._handler._completion_callbacks

        instrumentor.uninstrument()
