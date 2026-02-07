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

"""Tests for predictor.py - Rate limit prediction engine."""

from pathlib import Path

import pytest

from opentelemetry.util.genai.rate_limit.predictor import (
    RateLimitPrediction,
    RateLimitPredictor,
)
from opentelemetry.util.genai.rate_limit.providers.openai import (
    OpenAIRateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.tracker import TokenTracker


@pytest.fixture
def tracker(tmp_path: Path) -> TokenTracker:
    db_path = tmp_path / "test_predictor.db"
    return TokenTracker(db_path=str(db_path))


@pytest.fixture
def predictor(tracker: TokenTracker) -> RateLimitPredictor:
    provider = OpenAIRateLimitProvider()
    return RateLimitPredictor(
        tracker=tracker, provider=provider, warning_threshold=0.8
    )


class TestRateLimitPrediction:
    """Test the RateLimitPrediction dataclass."""

    def test_prediction_creation(self) -> None:
        pred = RateLimitPrediction(
            limit_type="tokens_per_minute",
            current_usage=160_000,
            limit=200_000,
            utilization=0.80,
            will_breach=False,
            time_to_breach_seconds=None,
            recommendation="Approaching TPM limit",
        )
        assert pred.limit_type == "tokens_per_minute"
        assert pred.utilization == 0.80

    def test_prediction_is_warning(self) -> None:
        """Predictions above threshold should be warnings."""
        pred = RateLimitPrediction(
            limit_type="tokens_per_minute",
            current_usage=180_000,
            limit=200_000,
            utilization=0.90,
            will_breach=False,
            recommendation="WARNING",
        )
        assert pred.is_warning(threshold=0.8)
        assert not pred.is_warning(threshold=0.95)


class TestTPMPrediction:
    """Test tokens-per-minute prediction."""

    def test_tpm_under_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        tokens_per_minute_pred = next(
            (p for p in predictions if p.limit_type == "tokens_per_minute"),
            None,
        )
        assert tokens_per_minute_pred is not None
        assert tokens_per_minute_pred.current_usage == 1500
        assert tokens_per_minute_pred.limit == 200_000
        assert not tokens_per_minute_pred.will_breach
        assert tokens_per_minute_pred.utilization < 0.01

    def test_tpm_approaching_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        # Record tokens close to the TPM limit (200k)
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=120_000,
            output_tokens=60_000,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        tokens_per_minute_pred = next(
            (p for p in predictions if p.limit_type == "tokens_per_minute"),
            None,
        )
        assert tokens_per_minute_pred is not None
        assert tokens_per_minute_pred.utilization == 0.9
        assert tokens_per_minute_pred.is_warning(threshold=0.8)


class TestRPMPrediction:
    """Test requests-per-minute prediction."""

    def test_rpm_under_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        for i in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=10,
                output_tokens=5,
                trace_id="t1",
                span_id=f"s{i}",
            )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        requests_per_minute_pred = next(
            (p for p in predictions if p.limit_type == "requests_per_minute"),
            None,
        )
        assert requests_per_minute_pred is not None
        assert requests_per_minute_pred.current_usage == 5
        assert requests_per_minute_pred.limit == 30


class TestWeeklyPrediction:
    """Test weekly limit prediction."""

    def test_weekly_under_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=50_000,
            output_tokens=25_000,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        weekly_pred = next(
            (p for p in predictions if p.limit_type == "weekly"), None
        )
        assert weekly_pred is not None
        assert weekly_pred.current_usage == 75_000
        assert weekly_pred.limit == 100_000_000


class TestMonthlyPrediction:
    """Test monthly limit prediction."""

    def test_monthly_input_under_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100_000,
            output_tokens=50_000,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        monthly_input_pred = next(
            (p for p in predictions if p.limit_type == "monthly_input"), None
        )
        assert monthly_input_pred is not None
        assert monthly_input_pred.current_usage == 100_000
        assert monthly_input_pred.limit == 500_000_000

    def test_monthly_output_under_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100_000,
            output_tokens=50_000,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        monthly_output_pred = next(
            (p for p in predictions if p.limit_type == "monthly_output"), None
        )
        assert monthly_output_pred is not None
        assert monthly_output_pred.current_usage == 50_000
        assert monthly_output_pred.limit == 50_000_000


class TestUnknownModel:
    """Test behavior with unknown models."""

    def test_predict_all_unknown_model_returns_empty(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        tracker.record(
            provider="openai",
            model="unknown-model",
            input_tokens=100,
            output_tokens=50,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="unknown-model"
        )
        assert predictions == []


class TestWorkflowPrediction:
    """Test workflow completion predictions."""

    def test_workflow_can_complete(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """Workflow with tokens remaining below limit should be completable."""
        # Train a pattern
        tracker.update_workflow_pattern(
            workflow_name="test_wf",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=1000,
        )
        # Record some progress
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            trace_id="wf_trace",
            span_id="s1",
            workflow_name="test_wf",
        )
        prediction = predictor.predict_workflow_completion(
            trace_id="wf_trace",
            workflow_name="test_wf",
            provider="openai",
            model="gpt-4o-mini",
        )
        assert prediction is not None
        assert prediction.can_complete is True
        assert prediction.progress_percent > 0

    def test_workflow_cannot_complete(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """Workflow should report can_complete=False when remaining tokens
        exceed rate limit headroom."""
        # Train a pattern predicting 300k total tokens
        tracker.update_workflow_pattern(
            workflow_name="heavy_wf",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=300_000,
        )
        # Record usage that nearly exhausts the TPM limit (200k)
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=120_000,
            output_tokens=70_000,
            trace_id="heavy_trace",
            span_id="s1",
            workflow_name="heavy_wf",
        )
        prediction = predictor.predict_workflow_completion(
            trace_id="heavy_trace",
            workflow_name="heavy_wf",
            provider="openai",
            model="gpt-4o-mini",
        )
        assert prediction is not None
        # 300k predicted total, 190k consumed, 110k remaining
        # TPM limit 200k, current TPM 190k → headroom 10k < 110k needed
        assert prediction.can_complete is False
        assert "CRITICAL" in prediction.recommendation

    def test_workflow_prediction_no_pattern(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """Without a learned pattern, prediction should return None."""
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            trace_id="wf_no_pattern",
            span_id="s1",
            workflow_name="unknown_wf",
        )
        prediction = predictor.predict_workflow_completion(
            trace_id="wf_no_pattern",
            workflow_name="unknown_wf",
            provider="openai",
            model="gpt-4o-mini",
        )
        assert prediction is None

    def test_workflow_prediction_no_trace_usage(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """If pattern exists but no trace usage yet, should predict with 0 current tokens."""
        tracker.update_workflow_pattern(
            workflow_name="new_wf",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=5000,
        )
        prediction = predictor.predict_workflow_completion(
            trace_id="nonexistent_trace",
            workflow_name="new_wf",
            provider="openai",
            model="gpt-4o-mini",
        )
        assert prediction is not None
        assert prediction.current_tokens == 0
        assert prediction.progress_percent == 0.0
        assert prediction.can_complete is True


class TestRecommendations:
    """Test recommendation generation."""

    def test_recommendation_for_high_utilization(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        # Push close to the limit
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=120_000,
            output_tokens=70_000,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        tokens_per_minute_pred = next(
            (p for p in predictions if p.limit_type == "tokens_per_minute"),
            None,
        )
        assert tokens_per_minute_pred is not None
        assert tokens_per_minute_pred.recommendation != ""
        assert (
            "WARNING" in tokens_per_minute_pred.recommendation
            or "tokens_per_minute"
            in tokens_per_minute_pred.recommendation.lower()
        )

    def test_recommendation_for_normal_utilization(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """Low utilization should produce an OK recommendation."""
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        tokens_per_minute_pred = next(
            (p for p in predictions if p.limit_type == "tokens_per_minute"),
            None,
        )
        assert tokens_per_minute_pred is not None
        assert "OK" in tokens_per_minute_pred.recommendation

    def test_recommendation_for_breached_limit(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """Utilization at or above 100% should produce a CRITICAL recommendation."""
        # Exceed the TPM limit (200k) directly via tracker
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=150_000,
            output_tokens=60_000,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        tokens_per_minute_pred = next(
            (p for p in predictions if p.limit_type == "tokens_per_minute"),
            None,
        )
        assert tokens_per_minute_pred is not None
        assert tokens_per_minute_pred.will_breach is True
        assert "CRITICAL" in tokens_per_minute_pred.recommendation

    def test_predict_all_returns_all_limit_types(
        self, predictor: RateLimitPredictor, tracker: TokenTracker
    ) -> None:
        """predict_all should return predictions for TPM, RPM, weekly,
        monthly_input, and monthly_output for a known model."""
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            trace_id="t1",
            span_id="s1",
        )
        predictions = predictor.predict_all(
            provider="openai", model="gpt-4o-mini"
        )
        limit_types = {p.limit_type for p in predictions}
        assert limit_types == {
            "tokens_per_minute",
            "requests_per_minute",
            "weekly",
            "monthly_input",
            "monthly_output",
        }
