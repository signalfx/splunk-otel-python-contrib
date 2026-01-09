"""Tests for LlamaIndex LLM instrumentation with OpenTelemetry."""

import os

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockLLM
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics


def setup_telemetry():
    """Setup OpenTelemetry with both trace and metrics exporters."""
    # Setup tracing
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Setup metrics with InMemoryMetricReader
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    return tracer_provider, meter_provider, metric_reader


def test_with_openai():
    """Test with real OpenAI API - requires OPENAI_API_KEY environment variable."""
    from llama_index.llms.openai import OpenAI

    print("=" * 80)
    print("Testing with OpenAI API")
    print("=" * 80)

    llm = OpenAI(model="gpt-3.5-turbo")
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Say hello in exactly 5 words"),
    ]

    response = llm.chat(messages)
    print(f"\nResponse: {response.message.content}")

    if hasattr(response, "raw") and response.raw:
        # Try dict-like .get() first (works with any dict-like object), fallback to getattr
        try:
            usage = response.raw.get("usage", {})
        except AttributeError:
            usage = getattr(response.raw, "usage", None)

        if usage:
            # Same pattern for usage object - try .get() first for dict-like objects
            try:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
            except AttributeError:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            print(
                f"\nToken Usage: input={prompt_tokens}, output={completion_tokens}, total={total_tokens}"
            )

    print("=" * 80)


class MockLLMWithUsage(MockLLM):
    """MockLLM that includes fake usage data for testing."""

    def _complete(self, prompt, **kwargs):
        """Override internal complete to inject usage data."""
        response = super()._complete(prompt, **kwargs)
        # Note: MockLLM uses _complete internally, but we can't easily inject
        # usage here because the ChatResponse is created later
        return response


def test_with_mock():
    """Test with MockLLM - no API key needed."""
    print("=" * 80)
    print("Testing with MockLLM")
    print("=" * 80)

    llm = MockLLM(max_tokens=50)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Say hello in 5 words"),
    ]

    response = llm.chat(messages)
    print(f"\nResponse: {response.message.content[:100]}...")
    print("=" * 80)


def test_message_extraction():
    """Test message extraction."""
    print("\n" + "=" * 80)
    print("Testing message extraction")
    print("=" * 80)

    llm = MockLLM(max_tokens=20)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Test message"),
    ]

    response = llm.chat(messages)
    print(f"\nResponse: {response.message.content[:50]}...")
    print("=" * 80)


if __name__ == "__main__":
    # Enable metrics emission
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"

    # Setup telemetry
    tracer_provider, meter_provider, metric_reader = setup_telemetry()

    # Instrument LlamaIndex
    instrumentor = LlamaindexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    print("LlamaIndex instrumentation enabled\n")

    # Run tests
    if os.environ.get("OPENAI_API_KEY"):
        print("Testing with real OpenAI API\n")
        test_with_openai()
    else:
        print("Testing with MockLLM (set OPENAI_API_KEY to test real API)\n")
        test_with_mock()

    # Test message extraction
    test_message_extraction()

    # Check metrics
    print("\n" + "=" * 80)
    print("Metrics Summary")
    print("=" * 80)

    metrics_data = metric_reader.get_metrics_data()
    found_duration = False
    found_token_usage = False

    if metrics_data:
        for rm in getattr(metrics_data, "resource_metrics", []) or []:
            for scope in getattr(rm, "scope_metrics", []) or []:
                for metric in getattr(scope, "metrics", []) or []:
                    print(f"\nMetric: {metric.name}")

                    if metric.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION:
                        found_duration = True
                        dps = getattr(metric.data, "data_points", [])
                        if dps:
                            print(f"  Duration: {dps[0].sum:.4f} seconds")
                            print(f"  Count: {dps[0].count}")

                    if metric.name == gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE:
                        found_token_usage = True
                        dps = getattr(metric.data, "data_points", [])
                        for dp in dps:
                            token_type = dp.attributes.get(
                                "gen_ai.token.type", "unknown"
                            )
                            print(
                                f"  Token type: {token_type}, Sum: {dp.sum}, Count: {dp.count}"
                            )

    print("\n" + "=" * 80)
    status = []
    if found_duration:
        status.append("Duration: OK")
    if found_token_usage:
        status.append("Token Usage: OK")
    if not found_duration and not found_token_usage:
        status.append("No metrics (use real API for metrics)")

    print("Status: " + " | ".join(status))
    print("=" * 80)
