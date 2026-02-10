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

"""Tests for propagation module (restore_session_from_context)."""

import os
from unittest.mock import patch

from opentelemetry import baggage

from opentelemetry.instrumentation.fastmcp.propagation import (
    restore_session_from_context,
)
from opentelemetry.util.genai.handler import (
    clear_session_context,
    get_session_context,
)


class TestRestoreSessionFromContext:
    """Tests for restore_session_from_context function."""

    def setup_method(self):
        clear_session_context()

    def teardown_method(self):
        clear_session_context()

    def test_restores_all_session_fields(self):
        """gen_ai.conversation.id, user.id, and customer.id are restored from baggage."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            ctx = baggage.set_baggage("gen_ai.conversation.id", "sess-1")
            ctx = baggage.set_baggage("user.id", "user-1", ctx)
            ctx = baggage.set_baggage("customer.id", "cust-1", ctx)

            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id == "sess-1"
            assert session.user_id == "user-1"
            assert session.customer_id == "cust-1"

    def test_restores_partial_session(self):
        """Only the fields present in baggage are restored."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            # Build context from scratch using Context() to avoid inheriting
            # leaked baggage from other tests
            from opentelemetry.context import Context

            clean_ctx = Context()
            ctx = baggage.set_baggage("user.id", "only-user", clean_ctx)

            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id is None
            assert session.user_id == "only-user"
            assert session.customer_id is None

    def test_no_restore_when_propagation_disabled(self):
        """Session is NOT restored when propagation mode is contextvar."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "contextvar"},
        ):
            ctx = baggage.set_baggage("gen_ai.conversation.id", "should-not-appear")
            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id is None

    def test_no_restore_when_env_not_set(self):
        """Session is NOT restored when env var is absent (default=contextvar)."""
        env = os.environ.copy()
        env.pop("OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION", None)
        with patch.dict(os.environ, env, clear=True):
            ctx = baggage.set_baggage("gen_ai.conversation.id", "should-not-appear")
            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id is None

    def test_no_error_on_none_context(self):
        """None context does not raise."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            restore_session_from_context(None)
            # Should not raise

    def test_no_error_on_empty_baggage(self):
        """Empty baggage context does not raise and does not set session."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            # Use a clean context with no baggage entries
            from opentelemetry.context import Context

            ctx = Context()
            restore_session_from_context(ctx)

            session = get_session_context()
            assert session.session_id is None

    def test_does_not_re_propagate_via_baggage(self):
        """Restored session should NOT re-inject into OTel Baggage (avoids loops)."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            ctx = baggage.set_baggage("gen_ai.conversation.id", "no-loop")

            with patch(
                "opentelemetry.util.genai.handler.set_session_context"
            ) as mock_set:
                restore_session_from_context(ctx)
                mock_set.assert_called_once()
                # propagate_via_baggage should be False
                assert mock_set.call_args.kwargs["propagate_via_baggage"] is False
