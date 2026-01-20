from __future__ import annotations

from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.types import (
    EmbeddingInvocation,
    EvaluationResult,
    LLMInvocation,
    OutputMessage,
)


class _StubHandler:
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


def test_manager_drops_invocation_when_queue_full(monkeypatch):
    """Test that invocations are dropped when queue is full."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "2")

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": []}

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
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": []}

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
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": []}

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
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": []}

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
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none")

    handler = _StubHandler()
    manager = Manager(handler)
    manager._evaluators = {"LLMInvocation": []}

    invocation = LLMInvocation(request_model="model")
    invocation.sample_for_evaluation = True
    invocation.output_messages = []
    invocation.attributes = {}

    manager.on_completion(invocation)

    # Should be queued without error
    assert invocation.evaluation_error is None
