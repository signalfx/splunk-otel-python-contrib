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

"""Proxy that forwards evaluations to a separate process.

This module contains the EvalManagerProxy class which implements the
CompletionCallback protocol and forwards GenAI invocations to a child
process for evaluation. This separation prevents DeepEval's LLM calls
from being instrumented alongside the application's telemetry.
"""

from __future__ import annotations

import logging
import os
import threading
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Optional

from opentelemetry.util.genai.callbacks import CompletionCallback
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EvaluationResult,
    GenAI,
    LLMInvocation,
    Workflow,
)

from .env import read_aggregation_flag, read_interval, read_queue_size
from .serialization import (
    deserialize_evaluation_result,
    serialize_invocation,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai.handler import TelemetryHandler

_LOGGER = logging.getLogger(__name__)

# Environment variable to control process mode
OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS"
)

# Timeout for worker startup (seconds)
OTEL_INSTRUMENTATION_GENAI_EVALS_WORKER_TIMEOUT = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_WORKER_TIMEOUT"
)

# Timeout for pending results (seconds)
OTEL_INSTRUMENTATION_GENAI_EVALS_RESULT_TIMEOUT = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULT_TIMEOUT"
)


def _read_worker_timeout() -> float:
    """Read worker startup timeout from environment."""
    value = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_EVALS_WORKER_TIMEOUT, "30"
    )
    try:
        return float(value)
    except ValueError:
        return 30.0


def _read_result_timeout() -> float:
    """Read result timeout from environment."""
    value = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_EVALS_RESULT_TIMEOUT, "60"
    )
    try:
        return float(value)
    except ValueError:
        return 60.0


class EvalManagerProxy(CompletionCallback):
    """Proxy that forwards evaluations to a separate process.

    This class implements the CompletionCallback protocol and is used
    when separate process mode is enabled. When the application process
    completes an LLM/Agent invocation, this proxy serializes the
    invocation data and sends it to a child process where the actual
    evaluation runs in isolation from OpenTelemetry instrumentation.

    The child process runs with OTEL_SDK_DISABLED=true, so DeepEval's
    OpenAI calls won't pollute the application's telemetry.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        interval: float | None = None,
        aggregate_results: bool | None = None,
        queue_size: int | None = None,
    ) -> None:
        """Initialize the proxy.

        Args:
            handler: The TelemetryHandler to use for emitting results.
            interval: Polling interval for the worker (passed to child).
            aggregate_results: Whether to aggregate results (passed to child).
            queue_size: Max pending evaluations before dropping.
        """
        self._handler = handler
        self._interval = interval if interval is not None else read_interval()
        self._aggregate_results = (
            aggregate_results
            if aggregate_results is not None
            else read_aggregation_flag()
        )
        self._queue_size = (
            queue_size if queue_size is not None else read_queue_size()
        )

        # IPC channels
        self._parent_conn: Optional[Connection] = None
        self._child_conn: Optional[Connection] = None
        self._worker_process: Optional[Process] = None

        # Result receiver thread
        self._result_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Pending evaluations (for correlation)
        self._pending: dict[str, GenAI] = {}
        self._pending_lock = threading.Lock()

        # Track worker state
        self._worker_ready = threading.Event()
        self._worker_failed = False

        # Start the worker
        self._start_worker()

    @property
    def has_evaluators(self) -> bool:
        """Check if the worker has evaluators configured."""
        # We assume evaluators are configured if the worker started successfully
        return self._worker_ready.is_set() and not self._worker_failed

    def _start_worker(self) -> None:
        """Spawn the evaluator worker process."""
        _LOGGER.info("Starting evaluator worker process")

        try:
            self._parent_conn, self._child_conn = Pipe(duplex=True)

            # Prepare configuration to pass to child
            config = {
                "interval": self._interval,
                "aggregate_results": self._aggregate_results,
                "evaluators_config": os.environ.get(
                    "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", ""
                ),
            }

            # Import the worker main function
            from .worker import _evaluator_worker_main

            self._worker_process = Process(
                target=_evaluator_worker_main,
                args=(self._child_conn, config),
                daemon=True,
                name="genai-evaluator-worker",
            )
            self._worker_process.start()

            # Start result receiver thread
            self._result_thread = threading.Thread(
                target=self._result_receiver_loop,
                daemon=True,
                name="genai-eval-result-receiver",
            )
            self._result_thread.start()

            # Wait for worker to be ready (or fail)
            # We consider it ready after a short delay if no error is received
            timeout = _read_worker_timeout()
            if self._worker_process.is_alive():
                # Give worker time to initialize
                self._worker_ready.wait(timeout=min(2.0, timeout))
                if self._worker_process.is_alive():
                    self._worker_ready.set()
                    _LOGGER.info(
                        "Evaluator worker process started successfully"
                    )

        except Exception as exc:
            _LOGGER.exception("Failed to start evaluator worker: %s", exc)
            self._worker_failed = True

    def on_completion(self, invocation: GenAI) -> None:
        """Forward invocation to evaluator process.

        This method is called by TelemetryHandler after completing an
        LLM or Agent invocation. It serializes the invocation data
        (excluding non-picklable objects like Span and context_token)
        and sends it to the worker process via IPC.

        Args:
            invocation: The completed GenAI invocation.
        """
        # Early exit if worker not ready
        if not self._worker_ready.is_set() or self._worker_failed:
            return

        if not self._worker_process or not self._worker_process.is_alive():
            _LOGGER.warning("Evaluator worker process not running")
            return

        # Only evaluate supported types
        if not isinstance(
            invocation, (LLMInvocation, AgentInvocation, Workflow)
        ):
            _LOGGER.debug(
                "Skipping evaluation for unsupported type: %s",
                type(invocation).__name__,
            )
            return

        # Check if sampling allows evaluation
        if not invocation.sample_for_evaluation:
            _LOGGER.debug(
                "Skipping evaluation: sample_for_evaluation is False"
            )
            return

        # Check queue size limit
        with self._pending_lock:
            if len(self._pending) >= self._queue_size:
                invocation.evaluation_error = "client_evaluation_queue_full"
                _LOGGER.warning(
                    "Evaluation queue full (size=%d). Dropping invocation.",
                    self._queue_size,
                )
                return

        # Serialize invocation data (not the Span object)
        try:
            serializable = serialize_invocation(invocation)
        except Exception as exc:
            _LOGGER.warning("Failed to serialize invocation: %s", exc)
            invocation.evaluation_error = (
                "client_evaluation_serialization_error"
            )
            return

        run_id = str(invocation.run_id)

        with self._pending_lock:
            self._pending[run_id] = invocation

        try:
            self._parent_conn.send(("evaluate", serializable))
            _LOGGER.debug("Sent invocation for evaluation: %s", run_id)
        except Exception as exc:
            _LOGGER.warning("Failed to send invocation to worker: %s", exc)
            with self._pending_lock:
                self._pending.pop(run_id, None)
            invocation.evaluation_error = "client_evaluation_send_error"

    def _result_receiver_loop(self) -> None:
        """Receive evaluation results from worker process."""
        _LOGGER.debug("Result receiver loop starting")

        while not self._shutdown_event.is_set():
            try:
                if self._parent_conn and self._parent_conn.poll(timeout=0.1):
                    message = self._parent_conn.recv()
                    self._handle_result_message(message)
            except EOFError:
                _LOGGER.info("Worker connection closed")
                break
            except Exception as exc:
                if not self._shutdown_event.is_set():
                    _LOGGER.debug("Error receiving result: %s", exc)
                continue

        _LOGGER.debug("Result receiver loop exiting")

    def _handle_result_message(self, message: tuple) -> None:
        """Process result message from worker.

        Correlates results with pending invocations by run_id and
        calls handler.evaluation_results() to emit telemetry via
        the CompositeEmitter's evaluation emitters.

        Args:
            message: Tuple of (message_type, payload).
        """
        if not isinstance(message, tuple) or len(message) != 2:
            _LOGGER.warning("Invalid message format from worker")
            return

        msg_type, payload = message

        if msg_type == "results":
            self._handle_results(payload)
        elif msg_type == "error":
            self._handle_error(payload)
        elif msg_type == "heartbeat_ack":
            _LOGGER.debug("Received heartbeat ack")
        else:
            _LOGGER.warning("Unknown message type from worker: %s", msg_type)

    def _handle_results(self, payload: dict) -> None:
        """Handle evaluation results from worker."""
        run_id = payload.get("run_id")
        results_data = payload.get("results", [])

        with self._pending_lock:
            invocation = self._pending.pop(run_id, None)

        if not invocation:
            _LOGGER.debug("No pending invocation for run_id: %s", run_id)
            return

        # Reconstruct EvaluationResult objects
        eval_results: list[EvaluationResult] = []
        for r in results_data:
            try:
                eval_results.append(deserialize_evaluation_result(r))
            except Exception as exc:
                _LOGGER.warning("Failed to deserialize result: %s", exc)

        if eval_results:
            _LOGGER.debug(
                "Received %d results for run_id: %s",
                len(eval_results),
                run_id,
            )
            # Emit telemetry via TelemetryHandler → CompositeEmitter
            try:
                self._handler.evaluation_results(invocation, eval_results)
            except Exception as exc:
                _LOGGER.warning("Failed to emit evaluation results: %s", exc)

    def _handle_error(self, payload: dict) -> None:
        """Handle error message from worker."""
        run_id = payload.get("run_id")
        error_msg = payload.get("message", "Unknown error")

        _LOGGER.warning(
            "Evaluation error for run_id %s: %s", run_id, error_msg
        )

        if run_id:
            with self._pending_lock:
                invocation = self._pending.pop(run_id, None)
            if invocation:
                invocation.evaluation_error = (
                    f"client_evaluation_worker_error: {error_msg}"
                )

    def wait_for_all(self, timeout: float | None = None) -> None:
        """Wait for all pending evaluations to complete.

        Args:
            timeout: Maximum time to wait in seconds.
        """
        import time

        if timeout is None:
            timeout = _read_result_timeout()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_lock:
                if not self._pending:
                    return
            time.sleep(0.1)

        _LOGGER.warning(
            "Timeout waiting for %d pending evaluations",
            len(self._pending),
        )

    def shutdown(self) -> None:
        """Gracefully shutdown the evaluator process."""
        _LOGGER.info("Shutting down evaluator proxy")
        self._shutdown_event.set()

        if self._parent_conn:
            try:
                self._parent_conn.send(("shutdown", None))
            except Exception:
                pass

        if self._worker_process:
            self._worker_process.join(timeout=5.0)
            if self._worker_process.is_alive():
                _LOGGER.warning("Worker process did not exit, terminating")
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)

        if self._result_thread and self._result_thread.is_alive():
            self._result_thread.join(timeout=1.0)

        _LOGGER.info("Evaluator proxy shutdown complete")


def is_separate_process_enabled() -> bool:
    """Check if separate process mode is enabled via environment variable."""
    value = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS, "false"
    )
    return value.lower() in ("true", "1", "yes", "on")


__all__ = [
    "EvalManagerProxy",
    "is_separate_process_enabled",
    "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS",
]
