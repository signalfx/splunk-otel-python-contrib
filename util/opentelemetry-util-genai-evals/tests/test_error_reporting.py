"""Tests for error reporting functionality."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from opentelemetry.util.genai.evals.errors import ErrorEvent, ErrorTracker
from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.types import EvaluationResult, LLMInvocation


class _FailingHandler:
    """Handler that always throws an exception."""

    def __init__(self, fail_on_publish: bool = True) -> None:
        self.fail_on_publish = fail_on_publish
        self.publish_attempts = 0

    def evaluation_results(
        self, invocation: LLMInvocation, results: list[EvaluationResult]
    ) -> None:
        self.publish_attempts += 1
        if self.fail_on_publish:
            raise RuntimeError("Handler deliberately failed")


def test_error_event_creation():
    """Test that ErrorEvent can be created with required fields."""
    event = ErrorEvent(
        timestamp=1234567890.0,
        error_type="test_error",
        severity="error",
        component="test",
        message="Test error message",
    )
    assert event.timestamp == 1234567890.0
    assert event.error_type == "test_error"
    assert event.severity == "error"
    assert event.component == "test"
    assert event.message == "Test error message"


def test_error_tracker_records_errors():
    """Test that ErrorTracker records and counts errors."""
    tracker = ErrorTracker()

    # Record first error
    event1 = tracker.record_error(
        error_type="evaluator_error",
        component="worker",
        message="Test error 1",
        evaluator_name="test_eval",
    )
    assert event1.error_type == "evaluator_error"
    assert event1.evaluator_name == "test_eval"

    # Record second error of same type
    tracker.record_error(
        error_type="evaluator_error",
        component="worker",
        message="Test error 2",
        evaluator_name="test_eval",
    )

    summary = tracker.get_error_summary()
    assert summary["total_errors"] == 2
    assert summary["unique_error_types"] == 1


def test_error_tracker_tracks_multiple_types():
    """Test that ErrorTracker can track different error types."""
    tracker = ErrorTracker()

    tracker.record_error(
        error_type="evaluator_error", component="worker", message="Error 1"
    )
    tracker.record_error(
        error_type="handler_error", component="manager", message="Error 2"
    )
    tracker.record_error(
        error_type="config_error", component="manager", message="Error 3"
    )

    summary = tracker.get_error_summary()
    assert summary["total_errors"] == 3
    assert summary["unique_error_types"] == 3


def test_manager_tracks_handler_failures(monkeypatch):
    """Test that Manager tracks handler failures without crashing."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    handler = _FailingHandler(fail_on_publish=True)
    manager = Manager(handler)

    # Create a simple evaluator setup
    manager._evaluators = {"LLMInvocation": []}
    manager._aggregate_results = True

    # Create invocation and fake results
    invocation = LLMInvocation(request_model="test-model")
    buckets = [[EvaluationResult(metric_name="test", score=0.5)]]

    # This should not raise even though handler fails
    result = manager._publish_results(invocation, buckets)

    # Handler should have been called
    assert handler.publish_attempts == 1

    # Should still return the flattened results
    assert len(result) == 1
    assert result[0].metric_name == "test"

    # Should have recorded the error
    summary = manager.get_error_summary()
    assert summary["total_errors"] == 1
    assert "handler_error:manager:none" in summary["error_counts"]


def test_manager_continues_after_evaluator_failure(monkeypatch):
    """Test that Manager continues processing after an evaluator fails."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    handler = Mock()
    manager = Manager(handler)

    # Create a failing evaluator
    failing_evaluator = Mock()
    failing_evaluator.evaluate.side_effect = RuntimeError("Evaluator failed")

    # Create a succeeding evaluator
    succeeding_evaluator = Mock()
    succeeding_evaluator.evaluate.return_value = [
        EvaluationResult(metric_name="success", score=0.9)
    ]

    manager._evaluators = {
        "LLMInvocation": [failing_evaluator, succeeding_evaluator]
    }

    invocation = LLMInvocation(request_model="test-model")

    # This should not crash
    buckets = manager._evaluate_invocation(invocation)

    # Should have one bucket from the succeeding evaluator
    assert len(buckets) == 1
    assert buckets[0][0].metric_name == "success"

    # Should have recorded the error
    summary = manager.get_error_summary()
    assert summary["total_errors"] >= 1


def test_error_event_includes_exception_details():
    """Test that ErrorEvent properly captures exception information."""
    tracker = ErrorTracker()

    try:
        raise ValueError("Test exception")
    except ValueError as e:
        event = tracker.record_error(
            error_type="test_error",
            component="test",
            message="Exception occurred",
            exception=e,
        )

    assert event.exception is not None
    assert "ValueError" in event.exception
    assert "Test exception" in event.exception


def test_manager_error_summary_accessible():
    """Test that error summary is accessible from Manager."""
    handler = Mock()
    manager = Manager(handler)

    # Should not crash even with no errors
    summary = manager.get_error_summary()
    assert "total_errors" in summary
    assert summary["total_errors"] == 0


# ============================================================================
# Concurrent Error Scenario Tests
# ============================================================================


def test_error_event_includes_worker_name():
    """Test that ErrorEvent includes worker_name field for concurrent mode."""
    event = ErrorEvent(
        timestamp=1234567890.0,
        error_type="test_error",
        severity="error",
        component="concurrent_worker",
        message="Test error message",
        worker_name="genai-evaluator-0",
        async_context=True,
    )
    assert event.worker_name == "genai-evaluator-0"
    assert event.async_context is True

    # Check to_log_extra includes new fields
    log_extra = event.to_log_extra()
    assert log_extra["worker_name"] == "genai-evaluator-0"
    assert log_extra["async_context"] is True


def test_error_tracker_thread_safety():
    """Test that ErrorTracker is thread-safe for concurrent error recording."""
    tracker = ErrorTracker()
    error_count = 100
    thread_count = 4

    def record_errors(thread_id: int) -> None:
        for i in range(error_count):
            tracker.record_error(
                error_type="test_error",
                component="test",
                message=f"Error from thread {thread_id}",
                worker_name=f"worker-{thread_id}",
            )

    # Run concurrent error recording
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(record_errors, i) for i in range(thread_count)
        ]
        for f in futures:
            f.result()

    summary = tracker.get_error_summary()
    assert summary["total_errors"] == error_count * thread_count


def test_error_tracker_tracks_per_worker_errors():
    """Test that ErrorTracker tracks errors per worker thread."""
    tracker = ErrorTracker()

    # Record errors from different workers
    tracker.record_error(
        error_type="test_error",
        component="worker",
        message="Error 1",
        worker_name="genai-evaluator-0",
    )
    tracker.record_error(
        error_type="test_error",
        component="worker",
        message="Error 2",
        worker_name="genai-evaluator-0",
    )
    tracker.record_error(
        error_type="test_error",
        component="worker",
        message="Error 3",
        worker_name="genai-evaluator-1",
    )

    summary = tracker.get_error_summary()
    assert summary["errors_by_worker"]["genai-evaluator-0"] == 2
    assert summary["errors_by_worker"]["genai-evaluator-1"] == 1


def test_error_tracker_tracks_async_vs_sync_errors():
    """Test that ErrorTracker distinguishes async vs sync errors."""
    tracker = ErrorTracker()

    # Record async errors
    tracker.record_error(
        error_type="async_evaluator_error",
        component="async_evaluator",
        message="Async error 1",
        async_context=True,
    )
    tracker.record_error(
        error_type="async_evaluator_error",
        component="async_evaluator",
        message="Async error 2",
        async_context=True,
    )

    # Record sync errors
    tracker.record_error(
        error_type="evaluator_error",
        component="evaluator",
        message="Sync error 1",
        async_context=False,
    )

    summary = tracker.get_error_summary()
    assert summary["async_errors"] == 2
    assert summary["sync_errors"] == 1


def test_error_tracker_auto_detects_worker_name():
    """Test that ErrorTracker auto-detects worker name from current thread."""
    tracker = ErrorTracker()

    # Record error without explicit worker name
    event = tracker.record_error(
        error_type="test_error",
        component="test",
        message="Test error",
    )

    # Should have auto-detected the current thread name
    assert event.worker_name is not None
    assert event.worker_name == threading.current_thread().name


def test_error_tracker_clear_resets_all_counters():
    """Test that ErrorTracker.clear() resets all counters including concurrent mode fields."""
    tracker = ErrorTracker()

    # Record some errors
    tracker.record_error(
        error_type="test_error",
        component="worker",
        message="Error 1",
        worker_name="genai-evaluator-0",
        async_context=True,
    )
    tracker.record_error(
        error_type="test_error",
        component="worker",
        message="Error 2",
        worker_name="genai-evaluator-1",
        async_context=False,
    )

    # Clear all
    tracker.clear()

    summary = tracker.get_error_summary()
    assert summary["total_errors"] == 0
    assert summary["error_counts"] == {}
    assert summary["errors_by_worker"] == {}
    assert summary["async_errors"] == 0
    assert summary["sync_errors"] == 0


def test_queue_full_error_tracking(monkeypatch):
    """Test that queue full errors are tracked with backpressure context."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "1")

    handler = Mock()
    manager = Manager(handler, queue_size=1)

    # Set up a mock evaluator so has_evaluators returns True
    manager._evaluators = {"LLMInvocation": [Mock()]}

    # Fill the queue
    invocation1 = LLMInvocation(request_model="test-model")
    manager._queue.put_nowait(invocation1)

    # Try to add another - should fail and be tracked
    invocation2 = LLMInvocation(request_model="test-model")
    manager.offer(invocation2)

    summary = manager.get_error_summary()
    assert summary["total_errors"] == 1
    assert "queue_full:manager:none" in summary["error_counts"]

    # Clean up
    manager._queue.get_nowait()


def test_error_summary_includes_concurrent_fields():
    """Test that get_error_summary includes concurrent mode fields."""
    handler = Mock()
    manager = Manager(handler)

    summary = manager.get_error_summary()

    # Should include all expected fields
    assert "total_errors" in summary
    assert "error_counts" in summary
    assert "unique_error_types" in summary
    assert "errors_by_worker" in summary
    assert "async_errors" in summary
    assert "sync_errors" in summary
