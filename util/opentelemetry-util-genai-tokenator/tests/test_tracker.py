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

"""Tests for tracker.py - SQLite token tracking and aggregation."""

import time
from pathlib import Path

import pytest

from opentelemetry.util.genai.rate_limit.tracker import TokenTracker


@pytest.fixture
def tracker(tmp_path: Path) -> TokenTracker:
    """Create a fresh tracker backed by a temporary database."""
    db_path = tmp_path / "test_tracker.db"
    return TokenTracker(db_path=str(db_path))


class TestRecordTokenUsage:
    """Test recording individual span-level token usage."""

    def test_record_single_usage(self, tracker: TokenTracker) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            trace_id="abc123",
            span_id="span1",
        )
        usage = tracker.get_usage_since(
            provider="openai", model="gpt-4o-mini", since=time.time() - 60
        )
        assert usage["total_tokens"] == 150
        assert usage["total_input_tokens"] == 100
        assert usage["total_output_tokens"] == 50
        assert usage["request_count"] == 1

    def test_record_multiple_usages(self, tracker: TokenTracker) -> None:
        for i in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
                trace_id="trace1",
                span_id=f"span{i}",
            )
        usage = tracker.get_usage_since(
            provider="openai", model="gpt-4o-mini", since=time.time() - 60
        )
        assert usage["total_tokens"] == 750
        assert usage["request_count"] == 5

    def test_record_with_workflow_name(self, tracker: TokenTracker) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            trace_id="trace1",
            span_id="span1",
            workflow_name="travel_planner",
        )
        usage = tracker.get_usage_since(
            provider="openai", model="gpt-4o-mini", since=time.time() - 60
        )
        assert usage["total_tokens"] == 300

    def test_record_without_trace_id(self, tracker: TokenTracker) -> None:
        """Records without trace_id should still be stored in token_usage
        but should NOT create a trace_token_usage entry."""
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
        )
        usage = tracker.get_usage_since(
            provider="openai", model="gpt-4o-mini", since=time.time() - 60
        )
        assert usage["total_tokens"] == 150
        assert usage["request_count"] == 1
        # No trace aggregation row should exist
        conn = tracker._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM trace_token_usage")
        assert cursor.fetchone()[0] == 0

    def test_record_with_explicit_timestamp(
        self, tracker: TokenTracker
    ) -> None:
        """Custom timestamp should be used for the record."""
        ts = time.time() - 120  # 2 minutes ago
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            timestamp=ts,
        )
        # Should NOT appear in last-60s window
        tokens_per_minute = tracker.get_current_tokens_per_minute(
            provider="openai", model="gpt-4o-mini"
        )
        assert tokens_per_minute == 0
        # But should appear in weekly window
        weekly = tracker.get_weekly_usage(
            provider="openai", model="gpt-4o-mini"
        )
        assert weekly["total_tokens"] == 150

    def test_record_isolates_models(self, tracker: TokenTracker) -> None:
        """Usage for different models should not interfere."""
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
        )
        tracker.record(
            provider="openai",
            model="gpt-4.1",
            input_tokens=200,
            output_tokens=100,
        )
        mini_usage = tracker.get_usage_since(
            provider="openai", model="gpt-4o-mini", since=time.time() - 60
        )
        assert mini_usage["total_tokens"] == 150
        gpt41_usage = tracker.get_usage_since(
            provider="openai", model="gpt-4.1", since=time.time() - 60
        )
        assert gpt41_usage["total_tokens"] == 300


class TestTraceAggregation:
    """Test trace-level token aggregation."""

    def test_trace_aggregation_single_span(
        self, tracker: TokenTracker
    ) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            trace_id="trace_agg_1",
            span_id="span1",
            workflow_name="test_workflow",
        )
        trace_usage = tracker.get_trace_usage("trace_agg_1")
        assert trace_usage is not None
        assert trace_usage["total_input_tokens"] == 100
        assert trace_usage["total_output_tokens"] == 50
        assert trace_usage["total_tokens"] == 150

    def test_trace_aggregation_multiple_spans(
        self, tracker: TokenTracker
    ) -> None:
        """Multiple spans in same trace should aggregate."""
        for i in range(3):
            tracker.record(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
                trace_id="trace_multi",
                span_id=f"span{i}",
                workflow_name="test_workflow",
            )
        trace_usage = tracker.get_trace_usage("trace_multi")
        assert trace_usage is not None
        assert trace_usage["total_tokens"] == 450

    def test_trace_aggregation_missing_trace(
        self, tracker: TokenTracker
    ) -> None:
        trace_usage = tracker.get_trace_usage("nonexistent")
        assert trace_usage is None


class TestRollingWindowQueries:
    """Test rolling window aggregation for TPM, weekly, monthly."""

    def test_get_current_tokens_per_minute(
        self, tracker: TokenTracker
    ) -> None:
        """tokens_per_minute = tokens in last 60 seconds."""
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            trace_id="t1",
            span_id="s1",
        )
        tokens_per_minute = tracker.get_current_tokens_per_minute(
            provider="openai", model="gpt-4o-mini"
        )
        assert tokens_per_minute == 1500

    def test_get_current_requests_per_minute(
        self, tracker: TokenTracker
    ) -> None:
        """requests_per_minute = requests in last 60 seconds."""
        for i in range(3):
            tracker.record(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
                trace_id="t1",
                span_id=f"s{i}",
            )
        requests_per_minute = tracker.get_current_requests_per_minute(
            provider="openai", model="gpt-4o-mini"
        )
        assert requests_per_minute == 3

    def test_get_weekly_usage(self, tracker: TokenTracker) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=5000,
            output_tokens=2000,
            trace_id="t1",
            span_id="s1",
        )
        weekly = tracker.get_weekly_usage(
            provider="openai", model="gpt-4o-mini"
        )
        assert weekly["total_tokens"] == 7000

    def test_get_monthly_usage(self, tracker: TokenTracker) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=10000,
            output_tokens=5000,
            trace_id="t1",
            span_id="s1",
        )
        monthly = tracker.get_monthly_usage(
            provider="openai", model="gpt-4o-mini"
        )
        assert monthly["total_tokens"] == 15000
        assert monthly["total_input_tokens"] == 10000
        assert monthly["total_output_tokens"] == 5000


class TestWorkflowPatterns:
    """Test workflow pattern learning via EMA."""

    def test_update_workflow_pattern_first_run(
        self, tracker: TokenTracker
    ) -> None:
        """First run should set the average directly."""
        tracker.update_workflow_pattern(
            workflow_name="travel_planner",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=1000,
        )
        pattern = tracker.get_workflow_pattern("travel_planner")
        assert pattern is not None
        assert pattern["avg_total_tokens"] == 1000
        assert pattern["sample_count"] == 1

    def test_update_workflow_pattern_ema(self, tracker: TokenTracker) -> None:
        """Subsequent runs should use EMA: new = α * latest + (1-α) * previous."""
        alpha = 0.3
        tracker_with_alpha = TokenTracker(
            db_path=tracker._db_path, ema_alpha=alpha
        )
        # First run
        tracker_with_alpha.update_workflow_pattern(
            workflow_name="planner",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=1000,
        )
        # Second run
        tracker_with_alpha.update_workflow_pattern(
            workflow_name="planner",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=2000,
        )
        pattern = tracker_with_alpha.get_workflow_pattern("planner")
        # EMA: 0.3 * 2000 + 0.7 * 1000 = 600 + 700 = 1300
        assert pattern is not None
        assert abs(pattern["avg_total_tokens"] - 1300.0) < 0.01
        assert pattern["sample_count"] == 2

    def test_update_workflow_pattern_multiple_runs_convergence(
        self, tracker: TokenTracker
    ) -> None:
        """EMA should converge toward the repeated value over many runs."""
        alpha = 0.3
        t = TokenTracker(db_path=tracker._db_path, ema_alpha=alpha)
        # First run with 1000 tokens
        t.update_workflow_pattern(
            workflow_name="converge",
            provider="openai",
            model="gpt-4o-mini",
            total_tokens=1000,
        )
        # Ten subsequent runs with 2000 tokens each → should converge near 2000
        for _ in range(10):
            t.update_workflow_pattern(
                workflow_name="converge",
                provider="openai",
                model="gpt-4o-mini",
                total_tokens=2000,
            )
        pattern = t.get_workflow_pattern("converge")
        assert pattern is not None
        # After 10 runs, EMA should be very close to 2000
        assert abs(pattern["avg_total_tokens"] - 2000) < 50
        assert pattern["sample_count"] == 11

    def test_get_workflow_pattern_missing(self, tracker: TokenTracker) -> None:
        pattern = tracker.get_workflow_pattern("nonexistent")
        assert pattern is None


class TestMarkTraceComplete:
    """Test marking a trace as completed and updating workflow patterns."""

    def test_mark_trace_complete(self, tracker: TokenTracker) -> None:
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
            trace_id="trace_complete",
            span_id="s1",
            workflow_name="test_wf",
        )
        tracker.mark_trace_complete("trace_complete")
        trace_usage = tracker.get_trace_usage("trace_complete")
        assert trace_usage is not None
        assert trace_usage["status"] == "completed"


class TestCleanup:
    """Test data retention and cleanup."""

    def test_cleanup_old_data(self, tracker: TokenTracker) -> None:
        """Records older than retention period should be removed."""
        # Insert a record with an old timestamp
        conn = tracker._get_connection()
        old_time = time.time() - (91 * 24 * 3600)  # 91 days ago
        conn.execute(
            """INSERT INTO token_usage
               (provider, model, input_tokens, output_tokens, total_tokens, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("openai", "gpt-4o-mini", 100, 50, 150, old_time),
        )
        conn.commit()

        # Insert a recent record
        tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            trace_id="recent",
            span_id="s1",
        )

        # Cleanup with 90-day retention
        tracker.cleanup(retention_days=90)

        usage = tracker.get_usage_since(
            provider="openai",
            model="gpt-4o-mini",
            since=old_time - 1,
        )
        # Only the recent record should remain
        assert usage["request_count"] == 1
        assert usage["total_tokens"] == 300
