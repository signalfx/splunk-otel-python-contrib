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

"""Tests for OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION behavior."""

import threading

import pytest

from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    Workflow,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    TelemetryHandler._reset_for_testing()
    yield
    TelemetryHandler._reset_for_testing()


def _make_llm_invocation() -> LLMInvocation:
    inv = LLMInvocation(request_model="test-model")
    inv.input_messages.append(
        InputMessage(role="user", parts=[Text(content="hi")])
    )
    inv.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content="hello")],
            finish_reason="stop",
        )
    )
    return inv


def _make_embedding_invocation() -> EmbeddingInvocation:
    return EmbeddingInvocation(
        request_model="embed-model", input_texts=["hello"]
    )


def _make_workflow() -> Workflow:
    return Workflow(name="test-workflow")


def _make_agent_invocation() -> AgentInvocation:
    return AgentInvocation(name="test-agent")


# ---------------------------------------------------------------------------
# Default behavior (flag off) — finalization runs inline
# ---------------------------------------------------------------------------


class TestDefaultInlineBehavior:
    """With flag off (default), stop_* calls must be fully synchronous."""

    def test_flag_off_executor_is_none(self):
        handler = get_telemetry_handler()
        assert handler._finalizer_executor is None
        assert handler._finalizer_semaphore is None

    def test_flag_off_emitter_called_synchronously(self):
        handler = get_telemetry_handler()
        called = []
        original_on_end = handler._emitter.on_end

        def recording_on_end(inv):
            called.append(True)
            original_on_end(inv)

        handler._emitter.on_end = recording_on_end

        inv = _make_llm_invocation()
        handler.start_llm(inv)
        handler.stop_llm(inv)

        # Must already be called by the time stop_llm returns
        assert called, "on_end was not called before stop_llm returned"

    def test_flag_off_callback_called_synchronously(self):
        from opentelemetry.util.genai.callbacks import CompletionCallback

        class _Recorder(CompletionCallback):
            def __init__(self):
                self.count = 0

            def on_completion(self, invocation):
                self.count += 1

        handler = get_telemetry_handler()
        recorder = _Recorder()
        handler.register_completion_callback(recorder)

        inv = _make_llm_invocation()
        handler.start_llm(inv)
        handler.stop_llm(inv)

        assert recorder.count == 1


# ---------------------------------------------------------------------------
# Async path (flag on) — stop_* returns before finalization completes
# ---------------------------------------------------------------------------


@pytest.fixture()
def async_handler(monkeypatch):
    """Create a fresh handler with async finalization enabled."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION", "true")
    return get_telemetry_handler()


class TestAsyncFinalizationEnabled:
    def test_executor_created_when_flag_on(self, async_handler):
        assert async_handler._finalizer_executor is not None
        assert async_handler._finalizer_semaphore is not None

    def test_on_end_runs_inline_completion_callbacks_offloaded(
        self, async_handler
    ):
        """on_end (span.end) runs inline; _notify_completion is offloaded to background."""
        from opentelemetry.util.genai.callbacks import CompletionCallback

        on_end_thread = []
        callback_thread = []
        barrier = threading.Barrier(2)

        original_on_end = async_handler._emitter.on_end

        def tracking_on_end(inv):
            on_end_thread.append(threading.current_thread().name)
            original_on_end(inv)

        async_handler._emitter.on_end = tracking_on_end

        class _BlockingCallback(CompletionCallback):
            def on_completion(self, invocation):
                callback_thread.append(threading.current_thread().name)
                barrier.wait(timeout=5)

        async_handler.register_completion_callback(_BlockingCallback())

        inv = _make_llm_invocation()
        async_handler.start_llm(inv)
        async_handler.stop_llm(inv)

        # on_end must have already run on the caller's thread before stop_llm returned
        assert on_end_thread, "on_end was not called"
        assert on_end_thread[0] == threading.current_thread().name, (
            "on_end must run inline on the caller's thread"
        )

        # callback is running on a different (background) thread — unblock it
        barrier.wait(timeout=5)
        async_handler.shutdown(wait=True)

        assert callback_thread, "callback was never called"
        assert callback_thread[0] != threading.current_thread().name, (
            "_notify_completion must run on a background thread"
        )

    def test_end_time_captured_inline(self, async_handler):
        """end_time must be set before stop_llm returns, regardless of async mode."""
        inv = _make_llm_invocation()
        async_handler.start_llm(inv)
        async_handler.stop_llm(inv)
        assert inv.end_time is not None

    def test_pop_current_span_inline(self, async_handler):
        """_pop_current_span runs inline so OTel context is restored immediately."""
        from opentelemetry.util.genai.handler import _current_genai_span

        inv = _make_llm_invocation()
        async_handler.start_llm(inv)
        async_handler.stop_llm(inv)
        span_after_stop = _current_genai_span.get()

        # After stop, current span should be restored to the parent (None at top level)
        assert span_after_stop is not inv.span, (
            "_pop_current_span must have run inline"
        )

    def test_callback_eventually_called(self, async_handler):
        """Completion callbacks must run on the background thread eventually."""
        from opentelemetry.util.genai.callbacks import CompletionCallback

        class _Recorder(CompletionCallback):
            def __init__(self):
                self.count = 0
                self.event = threading.Event()

            def on_completion(self, invocation):
                self.count += 1
                self.event.set()

        recorder = _Recorder()
        async_handler.register_completion_callback(recorder)

        inv = _make_llm_invocation()
        async_handler.start_llm(inv)
        async_handler.stop_llm(inv)

        # After shutdown the callback must have run
        async_handler.shutdown(wait=True)
        assert recorder.count == 1

    def test_shutdown_drains_pending_work(self, async_handler):
        """shutdown(wait=True) blocks until all queued tasks complete."""
        completed = []
        original_on_end = async_handler._emitter.on_end

        def tracking_on_end(inv):
            completed.append(id(inv))
            original_on_end(inv)

        async_handler._emitter.on_end = tracking_on_end

        invocations = []
        for _ in range(5):
            inv = _make_llm_invocation()
            invocations.append(inv)
            async_handler.start_llm(inv)
            async_handler.stop_llm(inv)

        async_handler.shutdown(wait=True)
        assert len(completed) == 5

    def test_fail_llm_on_error_inline(self, async_handler):
        """fail_llm runs on_error inline; only _notify_completion is offloaded."""
        called_thread = []
        original_on_error = async_handler._emitter.on_error

        def tracking_on_error(error, inv):
            called_thread.append(threading.current_thread().name)
            original_on_error(error, inv)

        async_handler._emitter.on_error = tracking_on_error

        inv = _make_llm_invocation()
        async_handler.start_llm(inv)
        async_handler.fail_llm(inv, Error(type=Exception, message="boom"))

        assert called_thread, "on_error was not called"
        assert called_thread[0] == threading.current_thread().name, (
            "on_error must run inline"
        )

    def test_stop_embedding_on_end_inline(self, async_handler):
        called_thread = []
        original = async_handler._emitter.on_end

        def tracking(inv):
            called_thread.append(threading.current_thread().name)
            original(inv)

        async_handler._emitter.on_end = tracking
        inv = _make_embedding_invocation()
        async_handler.start_embedding(inv)
        async_handler.stop_embedding(inv)

        assert called_thread, "on_end was not called"
        assert called_thread[0] == threading.current_thread().name

    def test_stop_workflow_on_end_inline(self, async_handler):
        called_thread = []
        original = async_handler._emitter.on_end

        def tracking(inv):
            called_thread.append(threading.current_thread().name)
            original(inv)

        async_handler._emitter.on_end = tracking
        wf = _make_workflow()
        async_handler.start_workflow(wf)
        async_handler.stop_workflow(wf)

        assert called_thread, "on_end was not called"
        assert called_thread[0] == threading.current_thread().name


# ---------------------------------------------------------------------------
# Queue-full fallback — inline execution when semaphore is exhausted
# ---------------------------------------------------------------------------


class TestQueueFullFallback:
    def test_falls_back_to_inline_when_queue_full(self, monkeypatch):
        """When the semaphore is exhausted, finalization runs inline."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION_QUEUE_SIZE", "1"
        )
        handler = get_telemetry_handler()

        # Drain the semaphore completely so every acquire() will fail
        handler._finalizer_semaphore.acquire()

        called_inline = []
        original_on_end = handler._emitter.on_end

        def tracking_on_end(inv):
            called_inline.append(threading.current_thread().name)
            original_on_end(inv)

        handler._emitter.on_end = tracking_on_end

        inv = _make_llm_invocation()
        handler.start_llm(inv)
        handler.stop_llm(inv)

        # Must have been called synchronously (on the main thread)
        assert called_inline, "on_end was not called at all"
        assert called_inline[0] == threading.current_thread().name, (
            "Fallback must run on the caller's thread, not a background thread"
        )

        # Release so teardown can shut down cleanly
        handler._finalizer_semaphore.release()


# ---------------------------------------------------------------------------
# shutdown() behavior
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_idempotent(self, monkeypatch):
        """Calling shutdown() twice must not raise."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION", "true"
        )
        handler = get_telemetry_handler()
        handler.shutdown(wait=True)
        handler.shutdown(wait=True)  # should not raise

    def test_shutdown_clears_executor(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION", "true"
        )
        handler = get_telemetry_handler()
        assert handler._finalizer_executor is not None
        handler.shutdown(wait=True)
        assert handler._finalizer_executor is None
        assert handler._finalizer_semaphore is None

    def test_stop_after_shutdown_falls_back_to_inline(self, monkeypatch):
        """After shutdown, stop_llm must still emit telemetry inline."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ASYNC_FINALIZATION", "true"
        )
        handler = get_telemetry_handler()
        handler.shutdown(wait=True)  # executor is now None

        called = []
        original_on_end = handler._emitter.on_end

        def tracking(inv):
            called.append(True)
            original_on_end(inv)

        handler._emitter.on_end = tracking

        inv = _make_llm_invocation()
        handler.start_llm(inv)
        handler.stop_llm(inv)

        assert called, "on_end must still run inline after shutdown"
