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

"""Evaluator worker process that runs evaluations in isolation.

This module contains the worker process entry point and the EvalWorker class
that receives serialized GenAI invocations from the parent process, runs
evaluations, and sends results back via IPC.

The worker process runs with OTEL_SDK_DISABLED=true to prevent any LLM calls
made during evaluation (e.g., by DeepEval) from being instrumented.
"""

from __future__ import annotations

import logging
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from opentelemetry.util.genai.evals.manager import Manager
    from opentelemetry.util.genai.types import EvaluationResult

_LOGGER = logging.getLogger(__name__)


def _evaluator_worker_main(conn: Connection, config: dict) -> None:
    """Main entry point for evaluator worker process.

    This function is the target for the child process. It disables
    OpenTelemetry SDK and DeepEval telemetry to prevent any LLM calls
    made during evaluation from polluting the application's telemetry.

    Args:
        conn: The IPC connection to the parent process.
        config: Configuration dict with interval, aggregate_results, etc.
    """
    # CRITICAL: Disable OpenTelemetry auto-instrumentation in this process
    # These must be set BEFORE importing any OTel modules
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
    os.environ["OTEL_TRACES_EXPORTER"] = "none"
    os.environ["OTEL_METRICS_EXPORTER"] = "none"
    os.environ["OTEL_LOGS_EXPORTER"] = "none"
    # Disable DeepEval's own telemetry
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

    # Configure logging for the worker process
    logging.basicConfig(
        level=logging.DEBUG
        if os.environ.get("OTEL_LOG_LEVEL") == "DEBUG"
        else logging.INFO,
        format="[eval-worker] %(levelname)s: %(message)s",
    )

    _LOGGER.info("Evaluator worker process starting")

    try:
        # Import after setting env vars to ensure they take effect
        from opentelemetry.util.genai.evals._noop_handler import (
            NoOpTelemetryHandler,
        )
        from opentelemetry.util.genai.evals.manager import Manager

        # Create a no-op handler - we don't emit telemetry from this process
        noop_handler = NoOpTelemetryHandler()

        # Create manager with configuration from parent
        evaluator_manager = Manager(
            noop_handler,
            interval=config.get("interval"),
            aggregate_results=config.get("aggregate_results"),
        )

        if not evaluator_manager.has_evaluators:
            _LOGGER.warning("No evaluators configured in worker process")
            conn.send(("error", {"message": "No evaluators configured"}))
            return

        _LOGGER.info(
            "Evaluator worker initialized with evaluators: %s",
            list(evaluator_manager._evaluators.keys()),
        )

        # Run the worker loop
        worker = EvalWorker(conn, evaluator_manager)
        worker.run()

    except Exception as exc:
        _LOGGER.exception("Evaluator worker failed to initialize")
        try:
            conn.send(("error", {"message": str(exc)}))
        except Exception:
            pass
    finally:
        _LOGGER.info("Evaluator worker process exiting")


class EvalWorker:
    """Worker that processes evaluation requests in the child process.

    Receives serialized GenAI invocations from the parent process,
    delegates to the existing Manager.evaluate_now() for evaluation,
    and sends results back via IPC.
    """

    def __init__(self, conn: Connection, manager: Manager) -> None:
        """Initialize the worker.

        Args:
            conn: The IPC connection to the parent process.
            manager: The evaluation Manager instance.
        """
        self._conn = conn
        self._manager = manager
        self._running = True

    def run(self) -> None:
        """Main worker loop - poll for messages and process them."""
        _LOGGER.debug("Worker loop starting")
        while self._running:
            try:
                if self._conn.poll(timeout=0.5):
                    message = self._conn.recv()
                    self._handle_message(message)
            except EOFError:
                _LOGGER.info("Connection closed by parent, shutting down")
                break
            except Exception as exc:
                _LOGGER.exception("Error in worker loop: %s", exc)
                continue
        _LOGGER.debug("Worker loop exiting")

    def _handle_message(self, message: tuple) -> None:
        """Handle incoming message from parent process.

        Args:
            message: Tuple of (message_type, payload).
        """
        if not isinstance(message, tuple) or len(message) != 2:
            _LOGGER.warning("Invalid message format: %s", message)
            return

        msg_type, payload = message

        if msg_type == "evaluate":
            self._process_evaluation(payload)
        elif msg_type == "shutdown":
            _LOGGER.info("Received shutdown request")
            self._running = False
            self._manager.shutdown()
        elif msg_type == "heartbeat":
            # Respond to heartbeat
            self._conn.send(
                ("heartbeat_ack", {"timestamp": payload.get("timestamp")})
            )
        else:
            _LOGGER.warning("Unknown message type: %s", msg_type)

    def _process_evaluation(self, payload: dict) -> None:
        """Process an evaluation request.

        Reconstructs the GenAI invocation from serialized data,
        runs evaluation via Manager.evaluate_now(), and sends
        results back to the parent process.

        Args:
            payload: Serialized invocation dict.
        """
        run_id = payload.get("run_id", "unknown")
        _LOGGER.debug("Processing evaluation for run_id: %s", run_id)

        try:
            from opentelemetry.util.genai.evals.serialization import (
                deserialize_invocation,
                serialize_evaluation_result,
            )

            # Reconstruct invocation from serialized data
            invocation = deserialize_invocation(payload)

            # Run evaluation synchronously using existing Manager logic
            results: Sequence[EvaluationResult] = self._manager.evaluate_now(
                invocation
            )

            # Send results back as serializable dicts
            serialized_results = [
                serialize_evaluation_result(r) for r in results
            ]

            self._conn.send(
                (
                    "results",
                    {
                        "run_id": run_id,
                        "results": serialized_results,
                    },
                )
            )
            _LOGGER.debug(
                "Sent %d evaluation results for run_id: %s",
                len(serialized_results),
                run_id,
            )

        except Exception as exc:
            _LOGGER.exception(
                "Evaluation failed for run_id %s: %s", run_id, exc
            )
            self._conn.send(
                (
                    "error",
                    {
                        "run_id": run_id,
                        "message": str(exc),
                        "type": type(exc).__name__,
                    },
                )
            )


__all__ = ["EvalWorker", "_evaluator_worker_main"]
