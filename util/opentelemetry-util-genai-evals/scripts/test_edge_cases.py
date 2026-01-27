#!/usr/bin/env python3
"""Test edge cases for separate process evaluation manager.

This script validates the behavior of EvalManagerProxy when running evaluations
in a separate process. It tests failure scenarios, timing characteristics, and
process isolation to ensure the evaluation system doesn't impact application
performance.

Prerequisites:
    - Install packages in editable mode:
        pip install -e util/opentelemetry-util-genai
        pip install -e util/opentelemetry-util-genai-evals
        pip install -e util/opentelemetry-util-genai-evals-deepeval
    - Optional: pip install psutil (for detailed process info)

Usage:
    cd <repo-root>

    # Normal flow test - verify proxy and worker are running
    python util/opentelemetry-util-genai-evals/scripts/test_edge_cases.py

    # Kill worker test - verify fast exit when worker is dead
    TEST=kill_worker python util/opentelemetry-util-genai-evals/scripts/test_edge_cases.py

    # Queue full test - verify backpressure behavior
    TEST=queue_full python util/opentelemetry-util-genai-evals/scripts/test_edge_cases.py

    # Timing test - measure overhead of stop_llm() with IPC
    TEST=timing python util/opentelemetry-util-genai-evals/scripts/test_edge_cases.py

    # Shutdown test - verify graceful worker shutdown
    TEST=shutdown python util/opentelemetry-util-genai-evals/scripts/test_edge_cases.py

Expected Results:
    - normal: Shows process separation (different PIDs), pending evaluation count
    - kill_worker: stop_llm() should complete quickly (<10ms) after worker killed
    - queue_full: evaluation_error should be 'client_evaluation_queue_full'
    - timing: Average overhead should be <1ms, max <5ms under normal conditions
    - shutdown: Worker should terminate within 5 seconds

Environment Variables:
    TEST - Test to run: normal, kill_worker, queue_full, timing, shutdown (default: normal)
    OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS - Set to 'true' by this script
    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS - Evaluator config (default: deepeval bias)
"""

import os
import signal
import sys
import time

# Enable separate process mode BEFORE any imports
os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS"] = "true"
os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = (
    "deepeval(LLMInvocation(bias))"
)
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"


def main():
    # Test selection
    TEST = os.environ.get("TEST", "normal")

    print(f"=== Running test: {TEST} ===\n")

    # Setup minimal telemetry
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    from opentelemetry.util.genai.handler import get_telemetry_handler
    from opentelemetry.util.genai.types import (
        InputMessage,
        LLMInvocation,
        OutputMessage,
        Text,
    )

    handler = get_telemetry_handler()

    # Access completion callbacks via handler internals
    callbacks = handler._completion_callbacks
    print(f"Completion callbacks: {[type(cb).__name__ for cb in callbacks]}")

    # Find the proxy
    proxy = None
    for cb in callbacks:
        if hasattr(cb, "_manager"):
            proxy = cb._manager
            break
        # Also check if it's directly an EvalManagerProxy
        if type(cb).__name__ == "EvalManagerProxy":
            proxy = cb
            break

    if proxy is None:
        print("ERROR: No proxy found! Callbacks are:")
        for cb in callbacks:
            print(f"  - {type(cb).__name__}: {cb}")
        sys.exit(1)

    print(f"Proxy type: {type(proxy).__name__}")
    print(f"Worker ready: {proxy._worker_ready.is_set()}")
    print(f"Worker failed: {proxy._worker_failed}")

    # Show process separation info
    print("\n--- Process Separation Verification ---")
    print(f"Parent process PID: {os.getpid()}")
    print(f"Parent process name: {sys.argv[0]}")

    if proxy._worker_process:
        worker_pid = proxy._worker_process.pid
        print(f"Worker process PID: {worker_pid}")
        print(f"Worker alive: {proxy._worker_process.is_alive()}")
        print(f"Worker daemon: {proxy._worker_process.daemon}")
        print(f"Worker name: {proxy._worker_process.name}")

        # Try to get more process info via psutil if available
        try:
            import psutil

            parent_proc = psutil.Process(os.getpid())
            worker_proc = psutil.Process(worker_pid)

            print("\n  Parent process details:")
            print(f"    PID: {parent_proc.pid}")
            print(f"    Name: {parent_proc.name()}")
            print(f"    Cmdline: {' '.join(parent_proc.cmdline()[:3])}...")
            print(
                f"    Memory: {parent_proc.memory_info().rss / 1024 / 1024:.1f} MB"
            )

            print("\n  Worker process details:")
            print(f"    PID: {worker_proc.pid}")
            print(f"    Name: {worker_proc.name()}")
            print(f"    Cmdline: {' '.join(worker_proc.cmdline()[:3])}...")
            print(
                f"    Memory: {worker_proc.memory_info().rss / 1024 / 1024:.1f} MB"
            )
            print(
                f"    Parent PID: {worker_proc.ppid()} (should match {os.getpid()})"
            )

            # Show environment isolation
            worker_env = worker_proc.environ()
            print(
                f"\n  Worker OTEL_SDK_DISABLED: {worker_env.get('OTEL_SDK_DISABLED', 'not set')}"
            )
            print(
                f"  Worker DEEPEVAL_TELEMETRY_OPT_OUT: {worker_env.get('DEEPEVAL_TELEMETRY_OPT_OUT', 'not set')}"
            )
        except ImportError:
            print(
                "  (install psutil for detailed process info: pip install psutil)"
            )
        except Exception as e:
            print(f"  (could not get detailed process info: {e})")

    print("--- End Process Verification ---\n")

    def create_invocation():
        return LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
            output_messages=[
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hi!")],
                    finish_reason="stop",
                )
            ],
            input_tokens=5,
            output_tokens=3,
        )

    if TEST == "normal":
        print("\n--- Normal flow test ---")
        llm = create_invocation()
        handler.start_llm(llm)
        start = time.perf_counter()
        handler.stop_llm(llm)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"stop_llm() took: {elapsed:.2f}ms")
        print(f"Pending evaluations: {len(proxy._pending)}")

    elif TEST == "kill_worker":
        print("\n--- Kill worker test ---")
        if proxy._worker_process:
            pid = proxy._worker_process.pid
            print(f"Killing worker process {pid}...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            print(
                f"Worker alive after kill: {proxy._worker_process.is_alive()}"
            )

        print("\nTrying to send invocation after worker killed...")
        llm = create_invocation()
        handler.start_llm(llm)
        start = time.perf_counter()
        handler.stop_llm(llm)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"stop_llm() took: {elapsed:.2f}ms (should be fast!)")
        print(f"evaluation_error: {llm.evaluation_error}")

    elif TEST == "queue_full":
        print("\n--- Queue full test ---")
        print(f"Queue size limit: {proxy._queue_size}")

        # Fill the pending dict to simulate queue full
        for i in range(proxy._queue_size + 5):
            proxy._pending[f"fake-{i}"] = create_invocation()

        print(f"Filled pending to: {len(proxy._pending)}")

        llm = create_invocation()
        handler.start_llm(llm)
        start = time.perf_counter()
        handler.stop_llm(llm)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"stop_llm() took: {elapsed:.2f}ms")
        print(f"evaluation_error: {llm.evaluation_error}")

    elif TEST == "timing":
        print("\n--- Timing test (10 iterations) ---")
        times = []
        for i in range(10):
            llm = create_invocation()
            handler.start_llm(llm)
            start = time.perf_counter()
            handler.stop_llm(llm)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  Iteration {i + 1}: {elapsed:.2f}ms")

        print(f"\nAvg: {sum(times) / len(times):.2f}ms")
        print(f"Max: {max(times):.2f}ms")
        print(f"Min: {min(times):.2f}ms")

    elif TEST == "shutdown":
        print("\n--- Graceful shutdown test ---")
        llm = create_invocation()
        handler.start_llm(llm)
        handler.stop_llm(llm)
        print(f"Pending before shutdown: {len(proxy._pending)}")

        start = time.perf_counter()
        proxy.shutdown()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"shutdown() took: {elapsed:.2f}ms")
        print(
            f"Worker alive after shutdown: {proxy._worker_process.is_alive() if proxy._worker_process else 'N/A'}"
        )

    print("\n=== Test complete ===")


if __name__ == "__main__":
    main()
