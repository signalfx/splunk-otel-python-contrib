"""Minimal LangChain Embedding instrumentation test.

Validates that embedding invocations:

1. Succeed using a recorded VCR cassette (no real API call).
2. Emit a span with GenAI semantic convention attributes for an embedding op.
3. Emit token usage metrics (input tokens via tiktoken estimation).
4. Emit duration metrics.
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
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


EMBEDDINGS = "embedding"


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
        api_key=SecretStr("test-api-key"),
        base_url="https://chat-ai.cisco.com/openai/deployments/text-embedding-ada-002",
        model=model,
        default_headers={"api-key": "test-api-key"},
    )

    # Act
    result = embeddings.embed_documents(["What is the capital of France?"])

    # Basic functional assertion
    assert result is not None
    assert len(result) >= 1
    assert isinstance(result[0], list)
    assert len(result[0]) > 0  # should have embedding dimensions

    # Spans
    spans: List[ReadableSpan] = span_exporter.get_finished_spans()
    assert spans, "Expected at least one span"

    # Print spans as JSON
    import json

    trace_output = []
    embedding_span = None
    for s in spans:
        attrs_obj = getattr(s, "attributes", None)
        span_dict = {
            "name": s.name,
            "trace_id": format(s.context.trace_id, "032x") if s.context else None,
            "span_id": format(s.context.span_id, "016x") if s.context else None,
            "parent_span_id": format(s.parent.span_id, "016x") if s.parent else None,
            "status": str(s.status.status_code.name) if s.status else None,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "duration_s": (s.end_time - s.start_time) / 1e9
            if s.end_time and s.start_time
            else None,
            "attributes": dict(attrs_obj) if attrs_obj else {},
            "events": [
                {
                    "name": evt.name,
                    "timestamp": evt.timestamp,
                    "attributes": dict(evt.attributes) if evt.attributes else {},
                }
                for evt in (getattr(s, "events", []) or [])
            ],
        }
        trace_output.append(span_dict)
        op_name = None
        try:
            if attrs_obj is not None:
                op_name = attrs_obj.get(gen_ai_attributes.GEN_AI_OPERATION_NAME)
        except Exception:
            op_name = None
        if op_name == EMBEDDINGS:
            embedding_span = s

    print("\n" + json.dumps(trace_output, indent=2, default=str) + "\n")
    assert embedding_span is not None, "No embedding operation span found"

    # Span attribute sanity
    attrs = getattr(embedding_span, "attributes", {})
    assert attrs.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == EMBEDDINGS
    assert attrs.get(gen_ai_attributes.GEN_AI_REQUEST_MODEL) == model

    # Token usage on span — if captured, must be a non-negative integer
    tok_val = attrs.get(gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS)
    if tok_val is not None:
        assert isinstance(tok_val, int) and tok_val >= 0

    # Metrics – token usage
    metrics_data = metric_reader.get_metrics_data()
    found_token = False
    found_duration = False

    # Build full OTLP-style JSON for metrics
    resource_metrics_json: list[dict] = []
    if metrics_data:
        for rm in getattr(metrics_data, "resource_metrics", []) or []:
            resource_dict: dict = {
                "attributes": dict(rm.resource.attributes) if rm.resource else {},
                "schema_url": getattr(rm.resource, "schema_url", "")
                if rm.resource
                else "",
            }
            scope_metrics_list: list[dict] = []
            for scope in getattr(rm, "scope_metrics", []) or []:
                scope_info = getattr(scope, "scope", None)
                scope_dict: dict = {
                    "name": getattr(scope_info, "name", None),
                    "version": getattr(scope_info, "version", None),
                    "schema_url": getattr(scope_info, "schema_url", None),
                    "attributes": getattr(scope_info, "attributes", None),
                }
                metrics_list: list[dict] = []
                for metric in getattr(scope, "metrics", []) or []:
                    metric_dict: dict = {
                        "name": metric.name,
                        "description": getattr(metric, "description", ""),
                        "unit": getattr(metric, "unit", ""),
                        "data": {
                            "data_points": [],
                            "aggregation_temporality": getattr(
                                metric.data, "aggregation_temporality", None
                            ),
                        },
                    }
                    for dp in getattr(metric.data, "data_points", []):
                        dp_attrs = dict(dp.attributes)
                        dp_dict: dict = {
                            "attributes": dp_attrs,
                            "start_time_unix_nano": getattr(
                                dp, "start_time_unix_nano", None
                            ),
                            "time_unix_nano": getattr(dp, "time_unix_nano", None),
                            "count": getattr(dp, "count", None),
                            "sum": getattr(dp, "sum", None),
                            "min": getattr(dp, "min", None),
                            "max": getattr(dp, "max", None),
                            "exemplars": [],
                        }
                        if hasattr(dp, "bucket_counts"):
                            dp_dict["bucket_counts"] = list(dp.bucket_counts)
                        if hasattr(dp, "explicit_bounds"):
                            dp_dict["explicit_bounds"] = list(dp.explicit_bounds)
                        metric_dict["data"]["data_points"].append(dp_dict)

                        if metric.name == "gen_ai.client.token.usage":
                            if dp_attrs.get("gen_ai.operation.name") == EMBEDDINGS:
                                assert dp_attrs.get("gen_ai.token.type") == "input"
                                assert dp.sum > 0
                                found_token = True
                        if metric.name == "gen_ai.client.operation.duration":
                            if dp_attrs.get("gen_ai.operation.name") == EMBEDDINGS:
                                assert dp.sum >= 0
                                found_duration = True
                    metrics_list.append(metric_dict)
                scope_metrics_list.append(
                    {
                        "scope": scope_dict,
                        "metrics": metrics_list,
                        "schema_url": getattr(scope, "schema_url", ""),
                    }
                )
            resource_metrics_json.append(
                {
                    "resource": resource_dict,
                    "scope_metrics": scope_metrics_list,
                    "schema_url": getattr(rm, "schema_url", ""),
                }
            )

    otlp_metrics = {"resource_metrics": resource_metrics_json}
    print("\n" + json.dumps(otlp_metrics, indent=4, default=str) + "\n")

    assert found_duration, "Duration metric missing for embedding"
    # Token metric relies on tiktoken being available — don't hard-fail if missing
    try:
        import tiktoken  # noqa: F401

        assert found_token, (
            "Token usage metric missing for embedding (tiktoken available)"
        )
    except ImportError:
        pass  # tiktoken not installed, token metric won't be emitted
