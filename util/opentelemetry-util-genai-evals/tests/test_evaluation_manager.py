from __future__ import annotations

from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.types import (
    EmbeddingInvocation,
    EvaluationResult,
    LLMInvocation,
    OutputMessage,
)


class _StubHandler:
    """Minimal handler stub for testing Manager without full TelemetryHandler."""

    def __init__(self) -> None:
        self.calls: list[tuple[LLMInvocation, list[EvaluationResult]]] = []

    def evaluation_results(
        self, invocation: LLMInvocation, results: list[EvaluationResult]
    ) -> None:
        self.calls.append((invocation, list(results)))


def _make_manager(
    monkeypatch, aggregate: bool
) -> tuple[Manager, _StubHandler]:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    if aggregate:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION", "true"
        )
    else:
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION",
            raising=False,
        )
    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": []}
    manager._aggregate_results = aggregate
    return manager, handler


def test_manager_emits_single_batch_when_aggregation_enabled(monkeypatch):
    manager, handler = _make_manager(monkeypatch, aggregate=True)
    invocation = LLMInvocation(request_model="agg-model")
    buckets = [
        [EvaluationResult(metric_name="bias", score=0.1)],
        [EvaluationResult(metric_name="toxicity", score=0.2)],
    ]

    flattened = manager._publish_results(invocation, buckets)

    assert len(handler.calls) == 1
    emitted = handler.calls[0][1]
    assert [res.metric_name for res in emitted] == ["bias", "toxicity"]
    assert flattened == emitted


def test_manager_emits_per_bucket_when_aggregation_disabled(monkeypatch):
    manager, handler = _make_manager(monkeypatch, aggregate=False)
    invocation = LLMInvocation(request_model="no-agg-model")
    buckets = [
        [EvaluationResult(metric_name="bias", score=0.1)],
        [EvaluationResult(metric_name="toxicity", score=0.2)],
    ]

    flattened = manager._publish_results(invocation, buckets)

    calls = handler.calls
    assert len(calls) == 2
    assert [res.metric_name for res in calls[0][1]] == ["bias"]
    assert [res.metric_name for res in calls[1][1]] == ["toxicity"]
    assert flattened == [item for bucket in buckets for item in bucket]


# =============================================================================
# Concurrent Mode Tests
# =============================================================================


def _make_concurrent_manager(
    monkeypatch,
    concurrent: bool,
    worker_count: int = 4,
    queue_size: int = 0,
) -> tuple[Manager, _StubHandler]:
    """Helper to create Manager with concurrent mode settings."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    if concurrent:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", "true"
        )
    else:
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", raising=False
        )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS", str(worker_count)
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", str(queue_size)
    )
    handler = _StubHandler()
    manager = Manager(handler)
    return manager, handler


def test_manager_concurrent_mode_disabled_by_default(monkeypatch):
    """Test that concurrent mode is disabled when env var not set."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT", raising=False
    )
    handler = _StubHandler()
    manager = Manager(handler)

    assert manager.concurrent_mode is False
    manager.shutdown()


def test_manager_concurrent_mode_enabled_via_env(monkeypatch):
    """Test that concurrent mode is enabled via environment variable."""
    manager, _ = _make_concurrent_manager(monkeypatch, concurrent=True)

    assert manager.concurrent_mode is True
    assert manager._worker_count == 4
    manager.shutdown()


def test_manager_concurrent_mode_explicit_parameter(monkeypatch):
    """Test that concurrent mode can be set via constructor parameter."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    handler = _StubHandler()
    manager = Manager(handler, concurrent_mode=True, worker_count=2)

    assert manager.concurrent_mode is True
    assert manager._worker_count == 2
    manager.shutdown()


def test_manager_worker_count_configurable(monkeypatch):
    """Test that worker count is configurable via env var."""
    manager, _ = _make_concurrent_manager(
        monkeypatch, concurrent=True, worker_count=8
    )

    assert manager._worker_count == 8
    manager.shutdown()


def test_manager_queue_size_configurable(monkeypatch):
    """Test that queue size is configurable via env var."""
    manager, _ = _make_concurrent_manager(
        monkeypatch, concurrent=False, queue_size=500
    )

    assert manager._queue_size == 500
    assert manager._queue.maxsize == 500
    manager.shutdown()


def test_manager_default_queue_size(monkeypatch):
    """Test that queue defaults to 100 when no explicit size is provided."""
    manager, _ = _make_concurrent_manager(monkeypatch, concurrent=False)

    # Default queue size is 100 (from read_queue_size())
    assert manager._queue_size == 100
    assert manager._queue.maxsize == 100
    manager.shutdown()


def test_manager_sequential_mode_single_worker(monkeypatch):
    """Test that sequential mode (default) creates no workers when no evaluators."""
    manager, _ = _make_concurrent_manager(monkeypatch, concurrent=False)

    # No evaluators configured, so no workers
    assert len(manager._workers) == 0
    assert manager.concurrent_mode is False
    manager.shutdown()


def test_manager_shutdown_clears_workers(monkeypatch):
    """Test that shutdown clears all workers."""
    manager, _ = _make_concurrent_manager(monkeypatch, concurrent=True)

    # Initially may have workers if evaluators configured
    manager.shutdown()

    assert len(manager._workers) == 0


# =============================================================================
# Queue and Invocation Handling Tests
# =============================================================================


def test_manager_drops_invocation_when_queue_full(monkeypatch):
    """Test that invocations are dropped when queue is full."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "2")

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": ["deepeval"]}

    # Fill the queue to capacity
    invocation1 = LLMInvocation(request_model="model1")
    invocation2 = LLMInvocation(request_model="model2")
    manager.offer(invocation1)
    manager.offer(invocation2)

    # This invocation should be dropped
    invocation3 = LLMInvocation(request_model="model3")
    manager.offer(invocation3)

    # Verify the dropped invocation has the error flag
    assert invocation3.evaluation_error == "client_evaluation_queue_full"
    assert invocation1.evaluation_error is None
    assert invocation2.evaluation_error is None


def test_on_completion_skips_unsupported_invocation_types(monkeypatch):
    """Test that unsupported invocation types are skipped with proper error."""

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": ["deepeval"]}

    invocation = EmbeddingInvocation()
    invocation.sample_for_evaluation = True

    manager.on_completion(invocation)

    assert (
        invocation.evaluation_error
        == "client_evaluation_skipped_as_invocation_type_not_supported"
    )
    assert len(handler.calls) == 0


def test_on_completion_skips_tool_llm_invocations(monkeypatch):
    """Test that tool-only LLM invocations are skipped with proper error."""

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": ["deepeval"]}

    invocation = LLMInvocation(request_model="model")
    invocation.sample_for_evaluation = True

    # Create tool call message
    output_message = OutputMessage(
        role="", parts=["ToolCall"], finish_reason="tool_calls"
    )
    invocation.output_messages = [output_message]

    manager.on_completion(invocation)

    assert (
        invocation.evaluation_error
        == "client_evaluation_skipped_as_tool_llm_invocation_type_not_supported"
    )
    assert len(handler.calls) == 0


def test_on_completion_skips_invocation_with_error(monkeypatch):
    """Test that invocations with errors are skipped with proper error flag."""

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": ["deepeval"]}

    invocation = LLMInvocation(request_model="model")
    invocation.sample_for_evaluation = True
    invocation.attributes = {"error.type": "timeout"}

    manager.on_completion(invocation)

    assert (
        invocation.evaluation_error
        == "client_evaluation_skipped_as_error_on_invocation"
    )
    assert len(handler.calls) == 0


def test_on_completion_processes_valid_invocation(monkeypatch):
    """Test that valid invocations are processed without error."""

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": ["deepeval"]}

    invocation = LLMInvocation(request_model="model")
    invocation.sample_for_evaluation = True
    invocation.output_messages = []
    invocation.attributes = {}

    manager.on_completion(invocation)

    # Should be queued without error
    assert invocation.evaluation_error is None
