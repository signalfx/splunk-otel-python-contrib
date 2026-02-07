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

"""Tests for emitter.py - RateLimitPredictorEmitter integration."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from opentelemetry.util.genai.rate_limit.emitter import (
    RateLimitPredictorEmitter,
    load_emitters,
)


class TestLoadEmitters:
    """Test the entry point loader function."""

    def test_load_emitters_returns_list(self) -> None:
        specs = load_emitters()
        assert isinstance(specs, list)
        assert len(specs) >= 1

    def test_load_emitters_spec_has_correct_name(self) -> None:
        specs = load_emitters()
        names = [s.name for s in specs]
        assert "RateLimitPredictor" in names

    def test_load_emitters_spec_has_category(self) -> None:
        specs = load_emitters()
        for spec in specs:
            assert spec.category != ""

    def test_load_emitters_factory_callable(self) -> None:
        specs = load_emitters()
        for spec in specs:
            assert callable(spec.factory)


class TestRateLimitPredictorEmitter:
    """Test the emitter lifecycle."""

    @pytest.fixture
    def emitter(self, tmp_path: Path) -> RateLimitPredictorEmitter:
        db_path = str(tmp_path / "test_emitter.db")
        return RateLimitPredictorEmitter(
            db_path=db_path,
            warning_threshold=0.8,
            event_logger=None,
        )

    def test_emitter_handles_llm_invocation(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        from opentelemetry.util.genai.types import LLMInvocation

        obj = LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )
        # on_start should not raise
        emitter.on_start(obj)

    def test_emitter_on_end_records_usage(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        from opentelemetry.util.genai.types import LLMInvocation

        obj = LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            trace_id=123456789,
            span_id=987654321,
        )
        emitter.on_end(obj)
        # Verify data was recorded
        tokens_per_minute = emitter._tracker.get_current_tokens_per_minute(
            provider="openai", model="gpt-4o-mini"
        )
        assert tokens_per_minute == 150

    def test_emitter_on_end_handles_workflow(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        from opentelemetry.util.genai.types import Workflow

        wf = Workflow(
            name="test_workflow",
            provider="openai",
            trace_id=111222333,
        )
        # Should not raise even though workflow has no tokens
        emitter.on_end(wf)

    def test_emitter_on_error_does_not_raise(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        from opentelemetry.util.genai.types import Error, LLMInvocation

        obj = LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
        )
        error = Error(message="test error", type=RuntimeError)
        emitter.on_error(error, obj)

    def test_emitter_on_evaluation_results_is_noop(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        emitter.on_evaluation_results([], None)

    def test_emitter_handles_non_genai_gracefully(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        """Non-GenAI objects should be silently ignored."""
        emitter.on_start("not a GenAI object")
        emitter.on_end("not a GenAI object")

    def test_emitter_skips_zero_token_invocation(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        """LLM invocations with 0 input and 0 output tokens should be skipped."""
        from opentelemetry.util.genai.types import LLMInvocation

        obj = LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
            input_tokens=0,
            output_tokens=0,
            trace_id=123456,
            span_id=789,
        )
        emitter.on_end(obj)
        tokens_per_minute = emitter._tracker.get_current_tokens_per_minute(
            provider="openai", model="gpt-4o-mini"
        )
        assert tokens_per_minute == 0

    def test_emitter_workflow_end_to_end(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        """Full lifecycle: LLM invocations followed by workflow completion."""
        from opentelemetry.util.genai.types import LLMInvocation, Workflow

        trace_id_int = 0xABCDEF0123456789ABCDEF0123456789
        # Simulate 3 LLM calls in a workflow
        for i in range(3):
            llm = LLMInvocation(
                request_model="gpt-4o-mini",
                provider="openai",
                input_tokens=500,
                output_tokens=200,
                trace_id=trace_id_int,
                span_id=i + 1,
                attributes={"gen_ai.workflow.name": "e2e_test"},
            )
            emitter.on_end(llm)

        # Complete the workflow
        wf = Workflow(
            name="e2e_test",
            provider="openai",
            trace_id=trace_id_int,
        )
        emitter.on_end(wf)

        # Verify trace was marked complete
        trace_hex = f"{trace_id_int:032x}"
        trace_usage = emitter._tracker.get_trace_usage(trace_hex)
        assert trace_usage is not None
        assert trace_usage["status"] == "completed"
        assert trace_usage["total_tokens"] == 2100  # 3 * (500 + 200)

        # Verify workflow pattern was learned
        pattern = emitter._tracker.get_workflow_pattern("e2e_test")
        assert pattern is not None
        assert pattern["avg_total_tokens"] == 2100.0

    def test_emitter_extracts_trace_id_from_int(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        """trace_id as int should be converted to 32-char hex."""
        trace_id_int = 0x2DFC14DFFA52ABF855A1FDCD9C52C83A
        result = emitter._format_trace_id(trace_id_int)
        assert result == "2dfc14dffa52abf855a1fdcd9c52c83a"
        assert len(result) == 32

    def test_emitter_extracts_span_id_from_int(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        """span_id as int should be converted to 16-char hex."""
        span_id_int = 0xA1B2C3D4E5F67890
        result = emitter._format_span_id(span_id_int)
        assert result == "a1b2c3d4e5f67890"
        assert len(result) == 16

    def test_emitter_handles_none_trace_id(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        result = emitter._format_trace_id(None)
        assert result is None

    def test_emitter_handles_none_span_id(
        self, emitter: RateLimitPredictorEmitter
    ) -> None:
        result = emitter._format_span_id(None)
        assert result is None


class TestEmitterWithEventLogger:
    """Test that warnings are emitted via event logger."""

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def emitter_with_logger(
        self, tmp_path: Path, mock_logger: MagicMock
    ) -> RateLimitPredictorEmitter:
        db_path = str(tmp_path / "test_emitter_events.db")
        return RateLimitPredictorEmitter(
            db_path=db_path,
            warning_threshold=0.0,  # Always warn for testing
            event_logger=mock_logger,
        )

    def test_event_emitted_on_warning(
        self,
        emitter_with_logger: RateLimitPredictorEmitter,
        mock_logger: MagicMock,
    ) -> None:
        from opentelemetry.util.genai.types import LLMInvocation

        obj = LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            trace_id=123456789,
            span_id=987654321,
        )
        emitter_with_logger.on_end(obj)
        # With threshold=0.0, any usage triggers a warning
        assert mock_logger.emit.called
