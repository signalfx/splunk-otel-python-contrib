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

"""Token usage tracking with SQLite persistence.

Provides rolling-window queries (tokens_per_minute, requests_per_minute,
weekly, monthly), trace-level aggregation, and workflow pattern learning
via exponential moving average.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from opentelemetry.util.genai.rate_limit.models import init_db

_logger = logging.getLogger(__name__)

# Rolling window sizes in seconds
_WINDOW_PER_MINUTE = 60  # 1 minute
_WINDOW_WEEKLY = 604_800  # 7 days
_WINDOW_MONTHLY = 2_592_000  # 30 days


class TokenTracker:
    """SQLite-backed token usage tracker with rolling-window queries.

    Args:
        db_path: Path to the SQLite database file.
        ema_alpha: Smoothing factor for exponential moving average (0 < α ≤ 1).
                   Higher values give more weight to recent observations.
    """

    def __init__(
        self,
        db_path: str,
        ema_alpha: float = 0.3,
    ) -> None:
        self._db_path = db_path
        self._ema_alpha = ema_alpha
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-safe database connection."""
        if self._conn is None:
            self._conn = init_db(self._db_path)
        return self._conn

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        trace_id: str | None = None,
        span_id: str | None = None,
        workflow_name: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record a single span's token usage.

        Also updates trace_token_usage for trace-level aggregation.
        """
        ts = timestamp or time.time()
        total = input_tokens + output_tokens
        conn = self._get_connection()

        conn.execute(
            """INSERT INTO token_usage
               (provider, model, input_tokens, output_tokens, total_tokens,
                timestamp, trace_id, span_id, workflow_name)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                provider,
                model,
                input_tokens,
                output_tokens,
                total,
                ts,
                trace_id,
                span_id,
                workflow_name,
            ),
        )

        # Upsert trace-level aggregation
        if trace_id:
            conn.execute(
                """INSERT INTO trace_token_usage
                   (trace_id, workflow_name, provider, model,
                    total_input_tokens, total_output_tokens, total_tokens,
                    start_time, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(trace_id) DO UPDATE SET
                    total_input_tokens = total_input_tokens + excluded.total_input_tokens,
                    total_output_tokens = total_output_tokens + excluded.total_output_tokens,
                    total_tokens = total_tokens + excluded.total_tokens,
                    last_updated = excluded.last_updated""",
                (
                    trace_id,
                    workflow_name,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    total,
                    ts,
                    ts,
                ),
            )

        conn.commit()

    def get_usage_since(
        self,
        *,
        provider: str,
        model: str,
        since: float,
    ) -> dict[str, Any]:
        """Get aggregated token usage since a given timestamp.

        Returns:
            Dict with keys: total_tokens, total_input_tokens, total_output_tokens, request_count
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT
                COALESCE(SUM(total_tokens), 0),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COUNT(*)
               FROM token_usage
               WHERE provider = ? AND model = ? AND timestamp >= ?""",
            (provider, model, since),
        )
        row = cursor.fetchone()
        return {
            "total_tokens": row[0],
            "total_input_tokens": row[1],
            "total_output_tokens": row[2],
            "request_count": row[3],
        }

    def get_current_tokens_per_minute(
        self, *, provider: str, model: str
    ) -> int:
        """Get tokens used in the last 60 seconds (rolling tokens per minute)."""
        since = time.time() - _WINDOW_PER_MINUTE
        usage = self.get_usage_since(
            provider=provider, model=model, since=since
        )
        return usage["total_tokens"]

    def get_current_requests_per_minute(
        self, *, provider: str, model: str
    ) -> int:
        """Get requests made in the last 60 seconds (rolling requests per minute)."""
        since = time.time() - _WINDOW_PER_MINUTE
        usage = self.get_usage_since(
            provider=provider, model=model, since=since
        )
        return usage["request_count"]

    def get_weekly_usage(self, *, provider: str, model: str) -> dict[str, Any]:
        """Get token usage in the last 7 days (rolling weekly)."""
        since = time.time() - _WINDOW_WEEKLY
        return self.get_usage_since(
            provider=provider, model=model, since=since
        )

    def get_monthly_usage(
        self, *, provider: str, model: str
    ) -> dict[str, Any]:
        """Get token usage in the last 30 days (rolling monthly)."""
        since = time.time() - _WINDOW_MONTHLY
        return self.get_usage_since(
            provider=provider, model=model, since=since
        )

    def get_trace_usage(self, trace_id: str) -> dict[str, Any] | None:
        """Get aggregated token usage for a specific trace.

        Returns:
            Dict with trace usage data, or None if trace not found.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT trace_id, workflow_name, provider, model,
                      total_input_tokens, total_output_tokens, total_tokens,
                      start_time, end_time, status
               FROM trace_token_usage
               WHERE trace_id = ?""",
            (trace_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "trace_id": row[0],
            "workflow_name": row[1],
            "provider": row[2],
            "model": row[3],
            "total_input_tokens": row[4],
            "total_output_tokens": row[5],
            "total_tokens": row[6],
            "start_time": row[7],
            "end_time": row[8],
            "status": row[9],
        }

    def mark_trace_complete(self, trace_id: str) -> None:
        """Mark a trace as completed and update its end time."""
        conn = self._get_connection()
        now = time.time()
        conn.execute(
            """UPDATE trace_token_usage
               SET status = 'completed', end_time = ?, last_updated = ?
               WHERE trace_id = ?""",
            (now, now, trace_id),
        )
        conn.commit()

    def update_workflow_pattern(
        self,
        *,
        workflow_name: str,
        provider: str,
        model: str,
        total_tokens: int,
    ) -> None:
        """Update the learned workflow pattern using exponential moving average.

        Formula: ema_new = α × latest + (1-α) × previous
        First observation sets the average directly.

        Args:
            workflow_name: Identifier for the workflow type.
            provider: LLM provider name.
            model: Model name.
            total_tokens: Total tokens consumed by this workflow run.
        """
        conn = self._get_connection()
        now = time.time()

        cursor = conn.execute(
            "SELECT avg_total_tokens, sample_count FROM workflow_patterns WHERE workflow_name = ?",
            (workflow_name,),
        )
        row = cursor.fetchone()

        if row is None:
            # First observation
            conn.execute(
                """INSERT INTO workflow_patterns
                   (workflow_name, provider, model, avg_total_tokens, sample_count, last_updated)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (workflow_name, provider, model, float(total_tokens), now),
            )
        else:
            previous_avg = row[0]
            sample_count = row[1]
            # EMA: new = α × latest + (1-α) × previous
            new_avg = (
                self._ema_alpha * total_tokens
                + (1 - self._ema_alpha) * previous_avg
            )
            conn.execute(
                """UPDATE workflow_patterns
                   SET avg_total_tokens = ?, sample_count = ?, last_updated = ?
                   WHERE workflow_name = ?""",
                (new_avg, sample_count + 1, now, workflow_name),
            )

        conn.commit()

    def get_workflow_pattern(
        self, workflow_name: str
    ) -> dict[str, Any] | None:
        """Get the learned pattern for a workflow type.

        Returns:
            Dict with pattern data, or None if no pattern exists.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT workflow_name, provider, model, avg_total_tokens,
                      sample_count, last_updated
               FROM workflow_patterns
               WHERE workflow_name = ?""",
            (workflow_name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "workflow_name": row[0],
            "provider": row[1],
            "model": row[2],
            "avg_total_tokens": row[3],
            "sample_count": row[4],
            "last_updated": row[5],
        }

    def cleanup(self, retention_days: int = 90) -> int:
        """Remove records older than the retention period.

        Args:
            retention_days: Number of days to retain data.

        Returns:
            Number of records removed.
        """
        cutoff = time.time() - (retention_days * 24 * 3600)
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM token_usage WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        conn.commit()
        return deleted
