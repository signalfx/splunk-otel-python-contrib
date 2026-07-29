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

"""
Tests for suppression of OpenAI instrumentation.

Covers two suppression surfaces:
1. SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY context key — set per-request
   by the LangChain instrumentor to prevent duplicate LLM spans when both
   instrumentors are active simultaneously.
2. SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION environment variable — set globally
   for zero-code deployments alongside OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=openai.
"""

import pytest

from opentelemetry import context as context_api
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)

# The env var name is the uppercase form of the context key string.
SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_ENV_VAR = (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY.upper()
)


@pytest.mark.vcr()
def test_chat_completion_suppressed(
    span_exporter, openai_client, instrument_with_content
):
    """Test that chat completions are not instrumented when suppression key is set."""
    # Set suppression key in context
    token = context_api.attach(
        context_api.set_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
        )
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say this is a test"}],
        )
        assert response is not None
    finally:
        context_api.detach(token)

    # Verify no spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_async_chat_completion_suppressed(
    span_exporter, async_openai_client, instrument_with_content
):
    """Test that async chat completions are not instrumented when suppression key is set."""
    # Set suppression key in context
    token = context_api.attach(
        context_api.set_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
        )
    )
    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say this is a test"}],
        )
        assert response is not None
    finally:
        context_api.detach(token)

    # Verify no spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.vcr()
def test_embeddings_suppressed(
    span_exporter, openai_client, instrument_with_content
):
    """Test that embeddings are not instrumented when suppression key is set."""
    # Set suppression key in context
    token = context_api.attach(
        context_api.set_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
        )
    )
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test for embeddings",
        )
        assert response is not None
    finally:
        context_api.detach(token)

    # Verify no spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_async_embeddings_suppressed(
    span_exporter, async_openai_client, instrument_with_content
):
    """Test that async embeddings are not instrumented when suppression key is set."""
    # Set suppression key in context
    token = context_api.attach(
        context_api.set_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
        )
    )
    try:
        response = await async_openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test for async embeddings",
        )
        assert response is not None
    finally:
        context_api.detach(token)

    # Verify no spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.vcr()
def test_chat_completion_not_suppressed_by_default(
    span_exporter, openai_client, instrument_with_content
):
    """Test that chat completions are instrumented normally when suppression key is not set."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    assert response is not None

    # Verify spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0
    # Should have at least the main chat completion span
    chat_spans = [s for s in spans if "chat" in s.name.lower()]
    assert len(chat_spans) > 0


# ---------------------------------------------------------------------------
# Environment variable suppression
# ---------------------------------------------------------------------------


@pytest.mark.vcr()
def test_chat_completion_suppressed_via_env_var(
    monkeypatch, span_exporter, openai_client, instrument_with_content
):
    """SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION=true suppresses spans globally."""
    monkeypatch.setenv(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_ENV_VAR, "true")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.vcr()
def test_chat_completion_not_suppressed_when_env_var_false(
    monkeypatch, span_exporter, openai_client, instrument_with_content
):
    """SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION=false leaves instrumentation active."""
    monkeypatch.setenv(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_ENV_VAR, "false"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE"])
def test_env_var_truthy_values_suppress(monkeypatch, value):
    """All truthy spellings of the env var are recognised."""
    monkeypatch.setenv(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_ENV_VAR, value)

    from opentelemetry.instrumentation.openai_v2.patch import (
        _is_instrumentation_suppressed,
    )

    assert _is_instrumentation_suppressed() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", ""])
def test_env_var_falsey_values_do_not_suppress(monkeypatch, value):
    """Falsey env var spellings leave instrumentation active."""
    monkeypatch.setenv(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_ENV_VAR, value)

    from opentelemetry.instrumentation.openai_v2.patch import (
        _is_instrumentation_suppressed,
    )

    assert _is_instrumentation_suppressed() is False


def test_env_var_unset_does_not_suppress(monkeypatch):
    """When the env var is absent, instrumentation is active by default."""
    monkeypatch.delenv(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_ENV_VAR, raising=False
    )

    from opentelemetry.instrumentation.openai_v2.patch import (
        _is_instrumentation_suppressed,
    )

    assert _is_instrumentation_suppressed() is False
