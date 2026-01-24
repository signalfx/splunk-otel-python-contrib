"""Tests for separate process evaluation mode."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from opentelemetry.util.genai.types import (
    AgentInvocation,
    EvaluationResult,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    Workflow,
)


class TestSerialization:
    """Tests for serialization/deserialization utilities."""

    def test_serialize_llm_invocation(self):
        """Test serializing an LLMInvocation."""
        from opentelemetry.util.genai.evals.serialization import (
            serialize_invocation,
        )

        invocation = LLMInvocation(
            request_model="gpt-4",
            provider="openai",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
            output_messages=[
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hi there!")],
                    finish_reason="stop",
                )
            ],
        )

        result = serialize_invocation(invocation)

        assert result["type"] == "LLMInvocation"
        assert result["request_model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert len(result["input_messages"]) == 1
        assert result["input_messages"][0]["role"] == "user"
        assert len(result["output_messages"]) == 1
        assert result["output_messages"][0]["role"] == "assistant"

    def test_serialize_agent_invocation(self):
        """Test serializing an AgentInvocation."""
        from opentelemetry.util.genai.evals.serialization import (
            serialize_invocation,
        )

        invocation = AgentInvocation(
            name="test-agent",
            provider="anthropic",
            agent_type="researcher",
            input_context="Find information about X",
            output_result="I found the following...",
        )

        result = serialize_invocation(invocation)

        assert result["type"] == "AgentInvocation"
        assert result["name"] == "test-agent"
        assert result["provider"] == "anthropic"
        assert result["agent_type"] == "researcher"
        assert result["input_context"] == "Find information about X"
        assert result["output_result"] == "I found the following..."

    def test_serialize_workflow(self):
        """Test serializing a Workflow."""
        from opentelemetry.util.genai.evals.serialization import (
            serialize_invocation,
        )

        invocation = Workflow(
            name="test-workflow",
            workflow_type="sequential",
            initial_input="Process this data",
            final_output="Data processed",
        )

        result = serialize_invocation(invocation)

        assert result["type"] == "Workflow"
        assert result["name"] == "test-workflow"
        assert result["workflow_type"] == "sequential"

    def test_deserialize_llm_invocation(self):
        """Test deserializing an LLMInvocation."""
        from opentelemetry.util.genai.evals.serialization import (
            deserialize_invocation,
            serialize_invocation,
        )

        original = LLMInvocation(
            request_model="gpt-4",
            provider="openai",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
            output_messages=[
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hi there!")],
                    finish_reason="stop",
                )
            ],
        )

        serialized = serialize_invocation(original)
        restored = deserialize_invocation(serialized)

        assert isinstance(restored, LLMInvocation)
        assert restored.request_model == "gpt-4"
        assert restored.provider == "openai"
        assert len(restored.input_messages) == 1
        assert len(restored.output_messages) == 1

    def test_deserialize_agent_invocation(self):
        """Test deserializing an AgentInvocation."""
        from opentelemetry.util.genai.evals.serialization import (
            deserialize_invocation,
            serialize_invocation,
        )

        original = AgentInvocation(
            name="test-agent",
            provider="anthropic",
            input_context="Test input",
            output_result="Test output",
        )

        serialized = serialize_invocation(original)
        restored = deserialize_invocation(serialized)

        assert isinstance(restored, AgentInvocation)
        assert restored.name == "test-agent"
        assert restored.provider == "anthropic"
        assert restored.input_context == "Test input"
        assert restored.output_result == "Test output"

    def test_serialize_evaluation_result(self):
        """Test serializing EvaluationResult."""
        from opentelemetry.util.genai.evals.serialization import (
            serialize_evaluation_result,
        )

        result = EvaluationResult(
            metric_name="bias",
            score=0.1,
            label="low",
            explanation="No bias detected",
            attributes={"threshold": 0.5},
        )

        serialized = serialize_evaluation_result(result)

        assert serialized["metric_name"] == "bias"
        assert serialized["score"] == 0.1
        assert serialized["label"] == "low"
        assert serialized["explanation"] == "No bias detected"
        assert serialized["attributes"]["threshold"] == 0.5

    def test_deserialize_evaluation_result(self):
        """Test deserializing EvaluationResult."""
        from opentelemetry.util.genai.evals.serialization import (
            deserialize_evaluation_result,
            serialize_evaluation_result,
        )

        original = EvaluationResult(
            metric_name="toxicity",
            score=0.05,
            label="none",
            explanation="Content is safe",
        )

        serialized = serialize_evaluation_result(original)
        restored = deserialize_evaluation_result(serialized)

        assert restored.metric_name == "toxicity"
        assert restored.score == 0.05
        assert restored.label == "none"
        assert restored.explanation == "Content is safe"


class TestNoOpHandler:
    """Tests for NoOpTelemetryHandler."""

    def test_evaluation_results_is_noop(self):
        """Test that evaluation_results does nothing."""
        from opentelemetry.util.genai.evals._noop_handler import (
            NoOpTelemetryHandler,
        )

        handler = NoOpTelemetryHandler()
        invocation = LLMInvocation(request_model="test")
        results = [EvaluationResult(metric_name="test", score=1.0)]

        # Should not raise
        handler.evaluation_results(invocation, results)


class TestIsSeparateProcessEnabled:
    """Tests for is_separate_process_enabled function."""

    def test_default_is_disabled(self, monkeypatch):
        """Test that separate process is disabled by default."""
        from opentelemetry.util.genai.evals.proxy import (
            is_separate_process_enabled,
        )

        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", raising=False
        )
        assert is_separate_process_enabled() is False

    def test_enabled_with_true(self, monkeypatch):
        """Test enabling with 'true'."""
        from opentelemetry.util.genai.evals.proxy import (
            is_separate_process_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", "true"
        )
        assert is_separate_process_enabled() is True

    def test_enabled_with_yes(self, monkeypatch):
        """Test enabling with 'yes'."""
        from opentelemetry.util.genai.evals.proxy import (
            is_separate_process_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", "yes"
        )
        assert is_separate_process_enabled() is True

    def test_enabled_with_one(self, monkeypatch):
        """Test enabling with '1'."""
        from opentelemetry.util.genai.evals.proxy import (
            is_separate_process_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", "1"
        )
        assert is_separate_process_enabled() is True

    def test_disabled_with_false(self, monkeypatch):
        """Test disabling with 'false'."""
        from opentelemetry.util.genai.evals.proxy import (
            is_separate_process_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", "false"
        )
        assert is_separate_process_enabled() is False


class TestBootstrapCreateEvaluationManager:
    """Tests for create_evaluation_manager factory function."""

    def test_returns_manager_when_disabled(self, monkeypatch):
        """Test that Manager is returned when separate process is disabled."""
        from opentelemetry.util.genai.evals.bootstrap import (
            create_evaluation_manager,
        )
        from opentelemetry.util.genai.evals.manager import Manager

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", "false"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        handler = MagicMock()
        result = create_evaluation_manager(handler)

        assert isinstance(result, Manager)

    def test_returns_proxy_when_enabled(self, monkeypatch):
        """Test that EvalManagerProxy is returned when separate process is enabled."""
        from opentelemetry.util.genai.evals.bootstrap import (
            create_evaluation_manager,
        )
        from opentelemetry.util.genai.evals.proxy import EvalManagerProxy

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        handler = MagicMock()
        result = create_evaluation_manager(handler)

        # Clean up
        if hasattr(result, "shutdown"):
            result.shutdown()

        assert isinstance(result, EvalManagerProxy)


class TestEvalManagerProxyBasic:
    """Basic tests for EvalManagerProxy without spawning processes."""

    def test_skips_unsupported_types(self, monkeypatch):
        """Test that unsupported invocation types are skipped."""
        from opentelemetry.util.genai.evals.proxy import EvalManagerProxy
        from opentelemetry.util.genai.types import EmbeddingInvocation

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        handler = MagicMock()

        # Mock the worker process to avoid actually spawning
        with patch.object(EvalManagerProxy, "_start_worker"):
            proxy = EvalManagerProxy(handler)
            proxy._worker_ready.set()
            proxy._worker_process = MagicMock()
            proxy._worker_process.is_alive.return_value = True
            proxy._parent_conn = MagicMock()

        invocation = EmbeddingInvocation(
            request_model="text-embedding-ada-002"
        )

        # Should not send to worker
        proxy.on_completion(invocation)

        proxy._parent_conn.send.assert_not_called()

    def test_skips_when_sample_for_evaluation_false(self, monkeypatch):
        """Test that invocations with sample_for_evaluation=False are skipped."""
        from opentelemetry.util.genai.evals.proxy import EvalManagerProxy

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )

        handler = MagicMock()

        with patch.object(EvalManagerProxy, "_start_worker"):
            proxy = EvalManagerProxy(handler)
            proxy._worker_ready.set()
            proxy._worker_process = MagicMock()
            proxy._worker_process.is_alive.return_value = True
            proxy._parent_conn = MagicMock()

        invocation = LLMInvocation(
            request_model="gpt-4",
            sample_for_evaluation=False,
        )

        proxy.on_completion(invocation)

        proxy._parent_conn.send.assert_not_called()

    def test_queue_size_limit(self, monkeypatch):
        """Test that queue size limit is respected."""
        from opentelemetry.util.genai.evals.proxy import EvalManagerProxy

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "none"
        )
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE", "2")

        handler = MagicMock()

        with patch.object(EvalManagerProxy, "_start_worker"):
            proxy = EvalManagerProxy(handler, queue_size=2)
            proxy._worker_ready.set()
            proxy._worker_process = MagicMock()
            proxy._worker_process.is_alive.return_value = True
            proxy._parent_conn = MagicMock()

        # Add two pending items
        from uuid import uuid4

        proxy._pending["id1"] = LLMInvocation(
            request_model="m1", run_id=uuid4()
        )
        proxy._pending["id2"] = LLMInvocation(
            request_model="m2", run_id=uuid4()
        )

        # Third should be dropped
        invocation = LLMInvocation(request_model="m3")
        proxy.on_completion(invocation)

        assert invocation.evaluation_error == "client_evaluation_queue_full"
        proxy._parent_conn.send.assert_not_called()
