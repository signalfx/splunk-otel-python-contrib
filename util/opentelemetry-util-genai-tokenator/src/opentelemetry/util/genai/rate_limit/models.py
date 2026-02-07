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

"""SQLite database schema for the Tokenator rate limit predictor.

Tables:
    token_usage - Individual span-level token usage records
    trace_token_usage - Aggregated token usage per trace (workflow run)
    workflow_patterns - Learned EMA patterns from completed workflows
"""

from __future__ import annotations

import sqlite3

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    trace_id TEXT,
    span_id TEXT,
    workflow_name TEXT,
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_token_usage_provider_model
    ON token_usage(provider, model);

CREATE INDEX IF NOT EXISTS idx_token_usage_trace_id
    ON token_usage(trace_id);

CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp
    ON token_usage(timestamp);

CREATE TABLE IF NOT EXISTS trace_token_usage (
    trace_id TEXT PRIMARY KEY,
    workflow_name TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    start_time REAL NOT NULL,
    end_time REAL,
    status TEXT DEFAULT 'in_progress',
    last_updated REAL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS workflow_patterns (
    workflow_name TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    avg_total_tokens REAL NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    last_updated REAL DEFAULT (strftime('%s', 'now'))
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize the SQLite database with the required schema.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A connection to the initialized database with WAL mode enabled.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(CREATE_TABLES_SQL)
    return conn
