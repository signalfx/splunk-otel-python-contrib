"""
Reproducer for: StreamWrapper missing .headers when LiteLLM calls
with_raw_response.create(stream=True) on an SDOT-instrumented Azure OpenAI client.

Production traceback (lab0, 2026-05-15):
  File "litellm/llms/azure/azure.py", line 619, in async_streaming
    headers, response = await self.make_azure_openai_chat_completion_request(...)
  File "litellm/llms/azure/azure.py", line 176, in make_azure_openai_chat_completion_request
    headers = dict(raw_response.headers)
  AttributeError: 'StreamWrapper' object has no attribute 'headers'

This reproducer calls make_azure_openai_chat_completion_request verbatim to
confirm the fix in SDOT's StreamWrapper resolves the crash.

Related upstream issues:
  #4032 - StreamWrapper missing .parse()  (fixed upstream)
  #4113 - StreamWrapper missing .headers  (fixed upstream via __getattr__,
           but SDOT needed a deeper fix: preserve LegacyAPIResponse.headers
           before _parse_response discards it)

Run:
  pip install openai litellm opentelemetry-sdk splunk-otel-instrumentation-openai
  python reproduce_raw_response_streaming.py
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import AsyncAzureOpenAI

from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Minimal SSE streaming chunks that mimic Azure OpenAI
# ---------------------------------------------------------------------------
SSE_CHUNKS = [
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n',
    b"data: [DONE]\n\n",
]


def _make_mock_httpx_response() -> httpx.Response:
    """Build a fake httpx.Response that the OpenAI SDK treats as a raw streaming response.

    The request must carry X-Stainless-Raw-Response: true so the OpenAI SDK
    returns LegacyAPIResponse (sync .parse()) rather than AsyncAPIResponse
    (async .parse()). SDOT's _parse_response calls .parse() synchronously.
    """
    response_headers = {
        "content-type": "text/event-stream",
        "x-request-id": "test-request-id-abc123",
        "openai-model": "gpt-4o",
        "ms-azureml-model-session": "d0",
    }

    async def aiter_bytes(_chunk_size=None):
        for chunk in SSE_CHUNKS:
            yield chunk

    mock_request = MagicMock(spec=httpx.Request)
    # RAW_RESPONSE_HEADER value that async_to_raw_response_wrapper injects
    mock_request.headers = httpx.Headers({"X-Stainless-Raw-Response": "true"})

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers(response_headers)
    mock_response.aiter_bytes = aiter_bytes
    mock_response.aclose = AsyncMock()
    mock_response.request = mock_request
    mock_response.http_version = "HTTP/1.1"
    mock_response.elapsed = MagicMock()
    return mock_response


# ---------------------------------------------------------------------------
# Verbatim copy of LiteLLM's make_azure_openai_chat_completion_request
# (litellm/llms/azure/azure.py lines 154-179)
# ---------------------------------------------------------------------------
async def make_azure_openai_chat_completion_request(
    azure_client, data, timeout
):
    """
    Helper to:
    - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
    - call chat.completions.create by default
    """
    try:
        raw_response = (
            await azure_client.chat.completions.with_raw_response.create(
                **data, timeout=timeout
            )
        )

        headers = dict(raw_response.headers)
        response = raw_response.parse()
        return headers, response
    except Exception as e:
        raise e


# ---------------------------------------------------------------------------
# Reproducer
# ---------------------------------------------------------------------------
async def reproducer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    azure_client = AsyncAzureOpenAI(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": True,
    }

    mock_httpx_response = _make_mock_httpx_response()

    with patch.object(
        azure_client._client,
        "send",
        new_callable=AsyncMock,
        return_value=mock_httpx_response,
    ):
        # This is the exact call that crashed in production
        headers, response = await make_azure_openai_chat_completion_request(
            azure_client=azure_client,
            data=data,
            timeout=60.0,
        )

    print(f"✓ headers accessible: {dict(headers)}")
    print(f"✓ response type: {type(response).__name__}")

    collected = []
    async for chunk in response:
        for choice in chunk.choices:
            if choice.delta.content:
                collected.append(choice.delta.content)

    print(f"✓ streamed content: {''.join(collected)!r}")

    spans = exporter.get_finished_spans()
    print(f"✓ OTel spans: {[s.name for s in spans]}")

    instrumentor.uninstrument()
    print(
        "\nReproducer passed — 'StreamWrapper' has no attribute 'headers' is fixed."
    )


if __name__ == "__main__":
    asyncio.run(reproducer())
