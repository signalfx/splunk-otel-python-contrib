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
Tests for SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY functionality.
This prevents duplicate telemetry when multiple instrumentations (e.g., LangChain + OpenAI) are active.
"""

import pytest

from opentelemetry import context as context_api
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
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
            messages=[{"role": "user", "content": "Say hello"}],
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
            messages=[{"role": "user", "content": "Say hello"}],
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
            input="Hello world",
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
            input="Hello world",
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
        messages=[{"role": "user", "content": "Say hello"}],
    )
    assert response is not None

    # Verify spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0
    # Should have at least the main chat completion span
    chat_spans = [s for s in spans if "chat" in s.name.lower()]
    assert len(chat_spans) > 0
