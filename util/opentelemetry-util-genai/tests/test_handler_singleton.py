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

"""Tests for get_telemetry_handler() singleton contract."""

import os
import threading
from unittest import mock

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Clear the singleton before and after each test."""
    TelemetryHandler._reset_for_testing()
    yield
    TelemetryHandler._reset_for_testing()


@pytest.fixture(autouse=True)
def environment():
    """Disable evals and use default emitters for test isolation."""
    original_evals = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"
    )
    original_emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS")

    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"

    yield

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = (
            original_evals
        )

    if original_emitters is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = original_emitters


class TestGetTelemetryHandlerSingleton:
    """Tests proving the singleton contract of get_telemetry_handler()."""

    def test_singleton_identity(self):
        """Two calls return the exact same object."""
        handler1 = get_telemetry_handler()
        handler2 = get_telemetry_handler()
        assert handler1 is handler2

    def test_singleton_constructor_and_factory(self):
        """TelemetryHandler() and get_telemetry_handler() return the same object."""
        handler1 = get_telemetry_handler()
        handler2 = TelemetryHandler()
        assert handler1 is handler2

    def test_singleton_constructor_only(self):
        """Two direct TelemetryHandler() calls return the same object."""
        handler1 = TelemetryHandler()
        handler2 = TelemetryHandler()
        assert handler1 is handler2

    def test_singleton_ignores_subsequent_providers(self):
        """Providers passed on the second call are ignored."""
        handler1 = get_telemetry_handler(tracer_provider=TracerProvider())
        handler2 = get_telemetry_handler(tracer_provider=TracerProvider())
        assert handler1 is handler2

    def test_returns_telemetry_handler_instance(self):
        """The returned object is an instance of TelemetryHandler."""
        handler = get_telemetry_handler()
        assert isinstance(handler, TelemetryHandler)

    def test_reset_allows_new_instance(self):
        """_reset_for_testing allows creating a fresh singleton."""
        handler1 = get_telemetry_handler()
        TelemetryHandler._reset_for_testing()
        handler2 = get_telemetry_handler()
        assert handler1 is not handler2
        assert isinstance(handler2, TelemetryHandler)

    def test_thread_safe_creation(self):
        """Concurrent first calls all receive the same handler instance."""
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        results = [None] * num_threads

        def get_handler(idx):
            barrier.wait()
            results[idx] = get_telemetry_handler()

        threads = [
            threading.Thread(target=get_handler, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is not None for r in results)
        assert all(r is results[0] for r in results), (
            "All threads must receive the same singleton instance"
        )

    def test_thread_safe_single_new(self):
        """TelemetryHandler.__new__ allocates exactly one instance under contention."""
        num_threads = 10
        barrier = threading.Barrier(num_threads)
        new_count = 0
        original_new = TelemetryHandler.__new__

        def counting_new(cls, *args, **kwargs):
            nonlocal new_count
            instance = original_new(cls, *args, **kwargs)
            if not getattr(instance, "_initialized", False):
                new_count += 1
            return instance

        with mock.patch.object(TelemetryHandler, "__new__", counting_new):

            def get_handler():
                barrier.wait()
                get_telemetry_handler()

            threads = [
                threading.Thread(target=get_handler)
                for _ in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert new_count == 1, (
            f"TelemetryHandler created {new_count} fresh instances, expected 1"
        )
