"""
Reproducer for: StreamWrapper missing .headers when LiteLLM calls
with_raw_response.create(stream=True).

Production error (lab0, 2026-05-15):
  File "litellm/llms/azure/azure.py", line 176
    headers = dict(raw_response.headers)
  AttributeError: 'StreamWrapper' object has no attribute 'headers'

LiteLLM's Azure provider always calls:
  raw_response = await azure_client.chat.completions.with_raw_response.create(...)
  headers = dict(raw_response.headers)   # <-- fails when SDOT is active
  response = raw_response.parse()        # <-- also fails without parse()

Related upstream issues:
  #4032 - StreamWrapper missing .parse()  (fixed)
  #4113 - StreamWrapper missing .headers  (fixed upstream, not yet in SDOT)

Run:
  pip install openai opentelemetry-sdk splunk-otel-instrumentation-openai
  python reproduce_raw_response_streaming.py
"""

import asyncio
import json
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
# Minimal SSE streaming response that mimics Azure OpenAI
# ---------------------------------------------------------------------------
SSE_CHUNKS = [
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n',
    b"data: [DONE]\n\n",
]


def _make_mock_httpx_response() -> httpx.Response:
    """Return a minimal mock httpx.Response with headers and a streaming body.

    The request headers must include X-Stainless-Raw-Response: true so the
    OpenAI SDK returns LegacyAPIResponse (sync .parse()) instead of
    AsyncAPIResponse (async .parse()). SDOT's _parse_response calls .parse()
    synchronously, so it must be LegacyAPIResponse.
    """
    response_headers = {
        "content-type": "text/event-stream",
        "x-request-id": "test-request-id-abc123",
        "openai-model": "gpt-4o",
    }
    # RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response" — must be "true" so the
    # OpenAI SDK wraps the response in LegacyAPIResponse (sync .parse()).
    request_headers = httpx.Headers({"X-Stainless-Raw-Response": "true"})

    async def aiter_bytes(_chunk_size=None):
        for chunk in SSE_CHUNKS:
            yield chunk

    mock_request = MagicMock(spec=httpx.Request)
    mock_request.headers = request_headers

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
# Reproducer
# ---------------------------------------------------------------------------
async def reproducer():
    # Set up OTel
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    client = AsyncAzureOpenAI(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    mock_httpx_response = _make_mock_httpx_response()

    # Patch the underlying httpx send so no real network call is made
    with patch.object(
        client._client,
        "send",
        new_callable=AsyncMock,
        return_value=mock_httpx_response,
    ):
        # This is exactly what LiteLLM's Azure provider does:
        #   https://github.com/BerriAI/litellm/blob/main/litellm/llms/azure/azure.py#L167-L176
        raw_response = await client.chat.completions.with_raw_response.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
            stream=True,
        )

        print(f"raw_response type: {type(raw_response).__name__}")

        # Step 1: LiteLLM accesses .headers  — this is the line that crashed
        headers = dict(raw_response.headers)
        print(f"✓ raw_response.headers: {json.dumps(headers, indent=2)}")

        # Step 2: LiteLLM calls .parse() to get the stream
        response = raw_response.parse()
        print(f"✓ raw_response.parse() returned: {type(response).__name__}")

        # Step 3: iterate the stream
        collected = []
        async for chunk in response:
            for choice in chunk.choices:
                if choice.delta.content:
                    collected.append(choice.delta.content)

        text = "".join(collected)
        print(f"✓ streamed content: {text!r}")

    spans = exporter.get_finished_spans()
    print(f"✓ OTel spans recorded: {len(spans)}")
    for span in spans:
        print(f"  - {span.name}")

    instrumentor.uninstrument()
    print("\nAll assertions passed — bug is fixed.")


if __name__ == "__main__":
    asyncio.run(reproducer())
