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

"""Tests for models.py - SQLite schema and database initialization."""

import sqlite3
from pathlib import Path

from opentelemetry.util.genai.rate_limit.models import (
    CREATE_TABLES_SQL,
    init_db,
)


class TestCreateTablesSql:
    """Verify the DDL string creates the expected tables and indexes."""

    def test_sql_creates_token_usage_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(CREATE_TABLES_SQL)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='token_usage'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_sql_creates_trace_token_usage_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(CREATE_TABLES_SQL)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trace_token_usage'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_sql_creates_workflow_patterns_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(CREATE_TABLES_SQL)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_patterns'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_sql_creates_indexes(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(CREATE_TABLES_SQL)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        assert "idx_token_usage_provider_model" in index_names
        assert "idx_token_usage_trace_id" in index_names
        assert "idx_token_usage_timestamp" in index_names
        conn.close()

    def test_sql_is_idempotent(self, tmp_path: Path) -> None:
        """Running the DDL twice should not raise."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(CREATE_TABLES_SQL)
        conn.executescript(CREATE_TABLES_SQL)  # Should not raise
        conn.close()


class TestInitDb:
    """Verify the init_db helper creates and returns a usable connection."""

    def test_init_db_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tokenator.db"
        conn = init_db(str(db_path))
        assert db_path.exists()
        conn.close()

    def test_init_db_returns_connection_with_tables(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "tokenator.db"
        conn = init_db(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = {row[0] for row in cursor.fetchall()}
        assert "token_usage" in table_names
        assert "trace_token_usage" in table_names
        assert "workflow_patterns" in table_names
        conn.close()

    def test_init_db_enables_wal_mode(self, tmp_path: Path) -> None:
        """WAL mode improves concurrent read performance."""
        db_path = tmp_path / "tokenator.db"
        conn = init_db(str(db_path))
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_token_usage_columns(self, tmp_path: Path) -> None:
        """Verify token_usage has all expected columns."""
        db_path = tmp_path / "tokenator.db"
        conn = init_db(str(db_path))
        cursor = conn.execute("PRAGMA table_info(token_usage)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "timestamp",
            "trace_id",
            "span_id",
            "workflow_name",
            "created_at",
        }
        assert expected.issubset(columns)
        conn.close()

    def test_trace_token_usage_columns(self, tmp_path: Path) -> None:
        """Verify trace_token_usage has all expected columns."""
        db_path = tmp_path / "tokenator.db"
        conn = init_db(str(db_path))
        cursor = conn.execute("PRAGMA table_info(trace_token_usage)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "trace_id",
            "workflow_name",
            "provider",
            "model",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "start_time",
            "end_time",
            "status",
            "last_updated",
        }
        assert expected.issubset(columns)
        conn.close()

    def test_workflow_patterns_columns(self, tmp_path: Path) -> None:
        """Verify workflow_patterns has all expected columns."""
        db_path = tmp_path / "tokenator.db"
        conn = init_db(str(db_path))
        cursor = conn.execute("PRAGMA table_info(workflow_patterns)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "workflow_name",
            "provider",
            "model",
            "avg_total_tokens",
            "sample_count",
            "last_updated",
        }
        assert expected.issubset(columns)
        conn.close()
