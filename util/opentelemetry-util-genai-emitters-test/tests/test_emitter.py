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

"""Tests for the TestEmitter class."""

import json
import os
import tempfile
import threading
import time


class TestTestEmitter:
    """Tests for TestEmitter functionality."""

    def test_on_start_captures_event(self):
        """Test that on_start captures telemetry events."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
            attributes={"test_key": "test_value"},
        )

        emitter.on_start(invocation)

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "start"
        assert events[0].invocation_type == "LLMInvocation"
        assert events[0].run_id == str(invocation.run_id)

        stats = emitter.get_stats()
        assert stats["total_starts"] == 1
        assert stats["invocations_by_type"]["LLMInvocation"] == 1

    def test_on_end_captures_event(self):
        """Test that on_end captures telemetry events."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )

        emitter.on_start(invocation)
        emitter.on_end(invocation)

        events = emitter.get_events()
        assert len(events) == 2
        assert events[1].event_type == "end"

        stats = emitter.get_stats()
        assert stats["total_starts"] == 1
        assert stats["total_ends"] == 1

    def test_on_error_captures_event(self):
        """Test that on_error captures error events."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            Error,
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )

        error = Error(type=ValueError, message="Something went wrong")

        emitter.on_start(invocation)
        emitter.on_error(error, invocation)

        events = emitter.get_events()
        assert len(events) == 2
        assert events[1].event_type == "error"

        stats = emitter.get_stats()
        assert stats["total_errors"] == 1

    def test_on_evaluation_results_captures_event(self):
        """Test that on_evaluation_results captures evaluation results."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            EvaluationResult,
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )

        results = [
            EvaluationResult(
                metric_name="bias", score=0.1, explanation="Low bias detected"
            ),
            EvaluationResult(
                metric_name="toxicity",
                score=0.05,
                explanation="Very low toxicity",
            ),
        ]

        emitter.on_start(invocation)
        emitter.on_evaluation_results(results, invocation)

        events = emitter.get_events()
        assert len(events) == 2
        assert events[1].event_type == "evaluation_results"
        assert len(events[1].evaluation_results) == 2

        stats = emitter.get_stats()
        assert stats["total_evaluation_results"] == 2
        assert stats["evaluation_results_by_metric"]["bias"] == 1
        assert stats["evaluation_results_by_metric"]["toxicity"] == 1

    def test_reset_clears_data(self):
        """Test that reset clears all captured data."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )

        emitter.on_start(invocation)
        emitter.on_end(invocation)

        assert len(emitter.get_events()) == 2

        emitter.reset()

        assert len(emitter.get_events()) == 0
        stats = emitter.get_stats()
        assert stats["total_starts"] == 0
        assert stats["total_ends"] == 0

    def test_export_to_json(self):
        """Test that export_to_json creates valid JSON file."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )

        emitter.on_start(invocation)
        emitter.on_end(invocation)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            emitter.export_to_json(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

            assert "stats" in data
            assert "events" in data
            assert len(data["events"]) == 2
            assert data["stats"]["total_starts"] == 1
        finally:
            os.unlink(filepath)

    def test_thread_safety(self):
        """Test that the emitter is thread-safe."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()
        num_threads = 10
        events_per_thread = 100

        def submit_events():
            for _ in range(events_per_thread):
                invocation = LLMInvocation(
                    request_model="test-model",
                    provider="test-provider",
                    input_messages=[
                        InputMessage(
                            role="user", parts=[Text(content="Hello")]
                        )
                    ],
                )
                emitter.on_start(invocation)
                emitter.on_end(invocation)

        threads = [
            threading.Thread(target=submit_events) for _ in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = emitter.get_stats()
        expected_total = num_threads * events_per_thread
        assert stats["total_starts"] == expected_total
        assert stats["total_ends"] == expected_total

    def test_wait_for_evaluations(self):
        """Test wait_for_evaluations with timeout."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            EvaluationResult,
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )

        # Start with no results - should timeout
        result = emitter.wait_for_evaluations(expected_count=1, timeout=0.1)
        assert result is False

        # Add results in a separate thread
        def add_results():
            time.sleep(0.1)
            emitter.on_evaluation_results(
                [EvaluationResult(metric_name="bias", score=0.1)], invocation
            )

        thread = threading.Thread(target=add_results)
        thread.start()

        result = emitter.wait_for_evaluations(expected_count=1, timeout=2.0)
        thread.join()

        assert result is True

    def test_get_pending_count(self):
        """Test pending invocation tracking."""
        from opentelemetry.util.genai.emitters.test import TestEmitter
        from opentelemetry.util.genai.types import (
            EvaluationResult,
            InputMessage,
            LLMInvocation,
            Text,
        )

        emitter = TestEmitter()

        invocation1 = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hello")])
            ],
        )
        invocation2 = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            input_messages=[
                InputMessage(role="user", parts=[Text(content="Hi")])
            ],
        )

        # Start two invocations
        emitter.on_start(invocation1)
        emitter.on_start(invocation2)
        assert emitter.get_pending_count() == 2

        # Complete one with evaluation results
        emitter.on_evaluation_results(
            [EvaluationResult(metric_name="bias", score=0.1)], invocation1
        )
        assert emitter.get_pending_count() == 1

        # Complete the other
        emitter.on_evaluation_results(
            [EvaluationResult(metric_name="bias", score=0.2)], invocation2
        )
        assert emitter.get_pending_count() == 0


class TestTestData:
    """Tests for test data set."""

    def test_get_all_samples_returns_120(self):
        """Test that get_all_samples returns 120 samples."""
        from opentelemetry.util.genai.emitters.test_data import get_all_samples

        samples = get_all_samples()
        assert len(samples) == 120

    def test_samples_by_category(self):
        """Test that each category has 20 samples."""
        from opentelemetry.util.genai.emitters.test_data import (
            get_samples_by_category,
        )

        categories = [
            "neutral",
            "subtle_bias",
            "subtle_toxicity",
            "hallucination",
            "irrelevant",
            "negative_sentiment",
        ]

        for category in categories:
            samples = get_samples_by_category(category)
            assert len(samples) == 20, (
                f"Category {category} should have 20 samples"
            )

    def test_sample_structure(self):
        """Test that samples have required fields."""
        from opentelemetry.util.genai.emitters.test_data import get_all_samples

        samples = get_all_samples()

        for sample in samples:
            assert sample.id is not None
            assert sample.category is not None
            assert sample.input_prompt is not None
            assert sample.response is not None
            assert sample.context is not None
            assert sample.expected_evaluation is not None


class TestLoadEmitters:
    """Tests for emitter loading."""

    def test_load_emitters_returns_correct_format(self):
        """Test that load_emitters returns correct format with multiple categories."""
        from opentelemetry.util.genai.emitters.test import load_emitters

        result = load_emitters()

        assert isinstance(result, list)
        assert len(result) == 2  # span and evaluation categories

        # Check each spec is a dict with required keys
        categories = set()
        for spec in result:
            assert isinstance(spec, dict)
            assert "name" in spec
            assert "category" in spec
            assert "factory" in spec
            assert spec["name"] == "test"
            categories.add(spec["category"])

        # Should have both span and evaluation categories
        assert categories == {"span", "evaluation"}

    def test_get_test_emitter_singleton(self):
        """Test that get_test_emitter returns singleton."""
        from opentelemetry.util.genai.emitters.test import get_test_emitter

        emitter1 = get_test_emitter()
        emitter2 = get_test_emitter()

        assert emitter1 is emitter2
