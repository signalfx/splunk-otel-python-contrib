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

"""Tests for session context propagation via OTel Baggage."""

import os
from unittest.mock import patch

from opentelemetry import baggage
from opentelemetry.util.genai.handler import (
    _apply_session_context,
    _get_session_from_baggage,
    _is_baggage_propagation_enabled,
    _set_session_baggage,
    clear_session_context,
    get_session_context,
    session_context,
    set_session_context,
)
from opentelemetry.util.genai.types import LLMInvocation


class TestBaggagePropagationConfig:
    """Test OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION env var."""

    def test_default_is_contextvar(self):
        """Default propagation mode is contextvar (no baggage)."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove the key if present
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION", None
            )
            assert not _is_baggage_propagation_enabled()

    def test_contextvar_mode(self):
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "contextvar"},
        ):
            assert not _is_baggage_propagation_enabled()

    def test_baggage_mode(self):
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            assert _is_baggage_propagation_enabled()

    def test_baggage_mode_case_insensitive(self):
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "BAGGAGE"},
        ):
            assert _is_baggage_propagation_enabled()


class TestSetSessionBaggage:
    """Test _set_session_baggage helper."""

    def test_sets_baggage_entries(self):
        _set_session_baggage(
            session_id="sess-1", user_id="user-1", customer_id="cust-1"
        )
        assert baggage.get_baggage("gen_ai.conversation.id") == "sess-1"
        assert baggage.get_baggage("user.id") == "user-1"
        assert baggage.get_baggage("customer.id") == "cust-1"

    def test_partial_baggage(self):
        _set_session_baggage(session_id="sess-2")
        assert baggage.get_baggage("gen_ai.conversation.id") == "sess-2"
        # user.id and customer.id may or may not be set from previous context

    def test_none_values_not_set(self):
        _set_session_baggage(session_id=None, user_id=None, customer_id=None)
        # Should not error — just no-op


class TestGetSessionFromBaggage:
    """Test _get_session_from_baggage helper."""

    def test_extracts_from_baggage(self):
        _set_session_baggage(
            session_id="sess-3", user_id="user-3", customer_id="cust-3"
        )
        ctx = _get_session_from_baggage()
        assert ctx.session_id == "sess-3"
        assert ctx.user_id == "user-3"
        assert ctx.customer_id == "cust-3"


class TestSetSessionContextWithBaggage:
    """Test set_session_context with propagate_via_baggage parameter."""

    def test_explicit_baggage_propagation(self):
        """When propagate_via_baggage=True, sets baggage regardless of env."""
        set_session_context(
            session_id="sess-4",
            user_id="user-4",
            propagate_via_baggage=True,
        )
        assert baggage.get_baggage("gen_ai.conversation.id") == "sess-4"
        assert baggage.get_baggage("user.id") == "user-4"
        # Cleanup
        clear_session_context()

    def test_explicit_no_baggage(self):
        """When propagate_via_baggage=False, no baggage regardless of env."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            set_session_context(
                session_id="sess-5",
                propagate_via_baggage=False,
            )
            ctx = get_session_context()
            assert ctx.session_id == "sess-5"
            clear_session_context()

    def test_env_based_baggage(self):
        """When propagate_via_baggage=None, uses env var."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            set_session_context(session_id="sess-6", user_id="user-6")
            assert baggage.get_baggage("gen_ai.conversation.id") == "sess-6"
            assert baggage.get_baggage("user.id") == "user-6"
            clear_session_context()


class TestSessionContextManagerWithBaggage:
    """Test session_context context manager with baggage."""

    def test_context_manager_with_baggage(self):
        with session_context(
            session_id="sess-7",
            user_id="user-7",
            propagate_via_baggage=True,
        ) as ctx:
            assert ctx.session_id == "sess-7"
            assert baggage.get_baggage("gen_ai.conversation.id") == "sess-7"

    def test_context_manager_restores(self):
        set_session_context(session_id="outer")
        with session_context(session_id="inner", propagate_via_baggage=False):
            ctx = get_session_context()
            assert ctx.session_id == "inner"
        # After exit, outer context is restored
        ctx = get_session_context()
        assert ctx.session_id == "outer"
        clear_session_context()


class TestApplySessionContextWithBaggage:
    """Test _apply_session_context with baggage in the priority chain."""

    def test_baggage_used_when_contextvar_empty(self):
        """When contextvar is empty but baggage is set, use baggage."""
        clear_session_context()  # Clear contextvar

        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            # Set baggage directly
            _set_session_baggage(
                session_id="baggage-sess", user_id="baggage-user"
            )

            invocation = LLMInvocation(request_model="test-model")
            _apply_session_context(invocation)

            assert invocation.session_id == "baggage-sess"
            assert invocation.user_id == "baggage-user"

    def test_contextvar_takes_priority_over_baggage(self):
        """ContextVar session takes priority over baggage."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            # Set baggage
            _set_session_baggage(session_id="baggage-sess")

            # Set contextvar (higher priority)
            set_session_context(
                session_id="contextvar-sess", propagate_via_baggage=False
            )

            invocation = LLMInvocation(request_model="test-model")
            _apply_session_context(invocation)

            assert invocation.session_id == "contextvar-sess"
            clear_session_context()

    def test_explicit_invocation_value_highest_priority(self):
        """Explicit value on invocation takes highest priority."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "baggage"},
        ):
            _set_session_baggage(session_id="baggage-sess")
            set_session_context(
                session_id="contextvar-sess", propagate_via_baggage=False
            )

            invocation = LLMInvocation(
                request_model="test-model", session_id="explicit-sess"
            )
            _apply_session_context(invocation)

            assert invocation.session_id == "explicit-sess"
            clear_session_context()

    def test_env_var_fallback(self):
        """Env var is lowest priority fallback."""
        clear_session_context()

        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION": "contextvar",
                "OTEL_INSTRUMENTATION_GENAI_SESSION_ID": "env-sess",
                "OTEL_INSTRUMENTATION_GENAI_USER_ID": "env-user",
            },
        ):
            invocation = LLMInvocation(request_model="test-model")
            _apply_session_context(invocation)

            assert invocation.session_id == "env-sess"
            assert invocation.user_id == "env-user"
