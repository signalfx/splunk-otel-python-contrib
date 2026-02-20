"""Minimal LangChain embedding instrumentation test.

Follows the same VCR cassette integration pattern as test_langchain_llm.py
to validate that embedding instrumentation emits correct telemetry:

1. An embedding invocation succeeds using the recorded VCR cassette.
2. A span is emitted with GenAI semantic convention attributes for an embeddings op.
3. The default operation_name is 'embeddings' (from EmbeddingInvocation types.py default).
4. Core request model attribute exists and is plausible.
5. Metrics (duration at minimum) are produced and contain at least one data point.
"""

from __future__ import annotations

# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false

from typing import Any, List
import pytest
from pytest import MonkeyPatch
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.sdk.trace import ReadableSpan  # test-only type reference
from opentelemetry.trace.status import StatusCode
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


EMBEDDINGS = gen_ai_attributes.GenAiOperationNameValues.EMBEDDINGS.value


@pytest.mark.vcr()
def test_langchain_embedding_call(
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_with_content: Any,
    monkeypatch: MonkeyPatch,
):
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    model = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=SecretStr("test-api-key"),
        check_embedding_ctx_length=False,  # avoid tiktoken download in test
    )

    # Act
    result = embeddings.embed_query("What is the capital of France?")

    # Basic functional assertion – result must be a list of floats
    assert isinstance(result, list), "Expected a list of floats"
    assert len(result) > 0, "Expected non-empty embedding vector"
    assert all(isinstance(v, float) for v in result), "All values must be floats"

    # Spans
    spans: List[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore[assignment]
    assert spans, "Expected at least one span"
    embedding_span = None
    for s in spans:
        attrs_obj = getattr(s, "attributes", None)
        op_name = None
        try:
            if attrs_obj is not None:
                op_name = attrs_obj.get(gen_ai_attributes.GEN_AI_OPERATION_NAME)
        except Exception:
            op_name = None
        if op_name == EMBEDDINGS:
            embedding_span = s
            break
    assert embedding_span is not None, "No embeddings operation span found"

    # Span attribute sanity
    attrs = getattr(embedding_span, "attributes", {})
    assert attrs.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == EMBEDDINGS
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_MODEL) == model
    # If token usage captured ensure it is a non-negative integer
    tok_val = attrs.get(gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS)
    if tok_val is not None:
        assert isinstance(tok_val, int) and tok_val >= 0

    # Span name should follow "{operation_name} {request_model}" convention
    assert embedding_span.name == f"embeddings {model}"

    # Metrics – ensure at least duration histogram present with >=1 point
    metrics_data = metric_reader.get_metrics_data()
    found_duration = False
    if metrics_data:
        for rm in getattr(metrics_data, "resource_metrics", []) or []:
            for scope in getattr(rm, "scope_metrics", []) or []:
                for metric in getattr(scope, "metrics", []) or []:
                    if metric.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION:
                        dps = getattr(metric.data, "data_points", [])
                        if dps:
                            assert dps[0].sum >= 0
                            found_duration = True
    assert found_duration, "Duration metric missing"


@pytest.mark.vcr()
def test_langchain_embedding_call_error(
    span_exporter: InMemorySpanExporter,
    instrument_with_content: Any,
    monkeypatch: MonkeyPatch,
):
    """When the embedding API returns an error the wrapper must:
    1. Still emit a span with operation_name == 'embeddings'.
    2. Mark the span status as ERROR.
    3. Re-raise the original exception so the caller sees the failure.
    """
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    model = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=SecretStr("test-api-key"),
        check_embedding_ctx_length=False,  # avoid tiktoken download in test
        max_retries=0,  # fail immediately, don't retry on 401
    )

    # Act – the call should raise because the cassette returns a 401
    with pytest.raises(Exception):
        embeddings.embed_query("What is the capital of France?")

    # Spans – an embedding span must still be emitted
    spans: List[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore[assignment]
    assert spans, "Expected at least one span even on error"
    embedding_span = None
    for s in spans:
        attrs_obj = getattr(s, "attributes", None)
        op_name = None
        try:
            if attrs_obj is not None:
                op_name = attrs_obj.get(gen_ai_attributes.GEN_AI_OPERATION_NAME)
        except Exception:
            op_name = None
        if op_name == EMBEDDINGS:
            embedding_span = s
            break
    assert embedding_span is not None, (
        "No embeddings operation span found on error path"
    )

    # Span attribute sanity
    attrs = getattr(embedding_span, "attributes", {})
    assert attrs.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == EMBEDDINGS
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_MODEL) == model

    # Span must be marked as error
    assert embedding_span.status.status_code == StatusCode.ERROR
