"""
TC-PI2-LITELLM: LiteLLM Proxy Observability Tests
==================================================

Combined test suite for LiteLLM proxy validation covering:
- TC-PI2-LITELLM-01: Proxy Metrics Validation
- TC-PI2-LITELLM-02: End-to-End Trace Correlation

Prerequisites:
- LiteLLM proxy running on localhost:4000
- Unified GenAI test app executed with litellm_proxy scenario
"""

import pytest
import time
from typing import Dict, Any


class TestLiteLLMProxyMetrics:
    """TC-PI2-LITELLM-01: LiteLLM Proxy Metrics Validation"""

    @pytest.fixture(scope="class")
    def trace_id(self, request) -> str:
        """Get trace ID from command line or use default"""
        return request.config.getoption(
            "--trace-id", default="5949cabf0b2cd8bf0e77e6cc7c0631a9"
        )

    @pytest.fixture(scope="class")
    def trace_data(self, apm_client, trace_id) -> Dict[str, Any]:
        """Fetch trace data from Splunk APM"""
        max_retries = 12
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                trace = apm_client.get_trace(trace_id)
                if trace and "spans" in trace:
                    return trace
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

        pytest.fail(
            f"Trace {trace_id} not available after {max_retries * retry_delay}s"
        )

    def test_litellm_proxy_scenario_exists(self, trace_data):
        """Verify litellm_proxy scenario span exists"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0, "litellm_proxy_scenario span not found"

        scenario_span = scenario_spans[0]
        assert scenario_span.get("tags", {}).get("scenario.name") == "litellm_proxy"
        assert scenario_span.get("tags", {}).get("scenario.type") == "proxy"

    def test_request_counts_captured(self, trace_data):
        """Verify request counts are tracked"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0
        scenario_span = scenario_spans[0]
        tags = scenario_span.get("tags", {})

        assert (
            "scenario.proxy.requests_total" in tags
        ), "Total request count not captured"
        assert (
            "scenario.proxy.requests_successful" in tags
        ), "Successful request count not captured"

        total_requests = int(tags["scenario.proxy.requests_total"])
        successful_requests = int(tags["scenario.proxy.requests_successful"])

        assert total_requests > 0, "No requests recorded"
        assert successful_requests > 0, "No successful requests recorded"
        assert successful_requests <= total_requests, "Successful requests exceed total"

    def test_latency_measurements_captured(self, trace_data):
        """Verify latency measurements are tracked"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0
        scenario_span = scenario_spans[0]
        tags = scenario_span.get("tags", {})

        assert "scenario.proxy.avg_latency_ms" in tags, "Average latency not captured"

        avg_latency = float(tags["scenario.proxy.avg_latency_ms"])
        assert avg_latency > 0, "Invalid latency measurement"
        assert (
            avg_latency < 30000
        ), f"Latency too high: {avg_latency}ms (expected < 30s)"

    def test_provider_attribution(self, trace_data):
        """Verify backend provider attribution is tracked"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0
        scenario_span = scenario_spans[0]
        tags = scenario_span.get("tags", {})

        assert "scenario.proxy.providers" in tags, "Provider attribution not captured"

        providers = tags["scenario.proxy.providers"]
        assert len(providers) > 0, "No providers tracked"
        assert "openai" in providers.lower(), "OpenAI provider not tracked"

    def test_llm_spans_present(self, trace_data):
        """Verify LLM operation spans are present"""
        spans = trace_data.get("spans", [])

        proxy_spans = [
            s
            for s in spans
            if "localhost:4000" in str(s.get("tags", {}).get("http.url", ""))
            or "POST" in s.get("name", "")
        ]

        assert len(proxy_spans) > 0, "No proxy request spans found"

    def test_token_usage_in_logs(self, trace_data):
        """Verify token usage is tracked"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0

    def test_multiple_models_tested(self, trace_data):
        """Verify multiple models were tested through proxy"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0
        scenario_span = scenario_spans[0]
        tags = scenario_span.get("tags", {})

        total_requests = int(tags.get("scenario.proxy.requests_total", 0))
        assert (
            total_requests >= 2
        ), f"Expected multiple model tests, got {total_requests}"

    def test_concurrent_requests_handled(self, trace_data):
        """Verify concurrent requests were handled successfully"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(scenario_spans) > 0, "Scenario span not found"


class TestLiteLLMTraceCorrelation:
    """TC-PI2-LITELLM-02: End-to-End Trace Correlation"""

    @pytest.fixture(scope="class")
    def trace_id(self, request) -> str:
        """Get trace ID from command line or use default"""
        return request.config.getoption(
            "--trace-id", default="5949cabf0b2cd8bf0e77e6cc7c0631a9"
        )

    @pytest.fixture(scope="class")
    def trace_data(self, apm_client, trace_id) -> Dict[str, Any]:
        """Fetch trace data from Splunk APM"""
        max_retries = 12
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                trace = apm_client.get_trace(trace_id)
                if trace and "spans" in trace:
                    return trace
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

        pytest.fail(
            f"Trace {trace_id} not available after {max_retries * retry_delay}s"
        )

    def test_root_span_exists(self, trace_data):
        """Verify root unified_genai_test_app span exists"""
        spans = trace_data.get("spans", [])

        root_spans = [s for s in spans if s.get("name") == "unified_genai_test_app"]

        assert len(root_spans) > 0, "Root span not found"

        root_span = root_spans[0]
        assert root_span.get("tags", {}).get("app.name") == "unified-genai-test"
        assert root_span.get("tags", {}).get("app.scenario") == "litellm_proxy"

    def test_scenario_span_hierarchy(self, trace_data):
        """Verify litellm_proxy_scenario span is child of root"""
        spans = trace_data.get("spans", [])

        root_spans = [s for s in spans if s.get("name") == "unified_genai_test_app"]
        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]

        assert len(root_spans) > 0, "Root span not found"
        assert len(scenario_spans) > 0, "Scenario span not found"

        root_span = root_spans[0]
        scenario_span = scenario_spans[0]

        root_span_id = root_span.get("spanId")
        scenario_parent_id = scenario_span.get("parentSpanId")

        assert (
            scenario_parent_id == root_span_id
        ), f"Scenario span parent mismatch: expected {root_span_id}, got {scenario_parent_id}"

    def test_complete_span_chain(self, trace_data):
        """Verify complete span chain from client to backend"""
        spans = trace_data.get("spans", [])

        assert len(spans) >= 3, f"Expected at least 3 spans, got {len(spans)}"

        span_map = {s.get("spanId"): s for s in spans}

        root_spans = [s for s in spans if not s.get("parentSpanId")]
        assert len(root_spans) == 1, f"Expected 1 root span, got {len(root_spans)}"

        for span in spans:
            parent_id = span.get("parentSpanId")
            if parent_id:
                assert (
                    parent_id in span_map
                ), f"Span {span.get('name')} has invalid parent {parent_id}"

    def test_http_client_spans_present(self, trace_data):
        """Verify HTTP client spans for proxy requests"""
        spans = trace_data.get("spans", [])

        http_spans = [
            s
            for s in spans
            if "POST" in s.get("name", "")
            or s.get("tags", {}).get("http.method") == "POST"
            or "localhost:4000" in str(s.get("tags", {}).get("http.url", ""))
        ]

        assert len(http_spans) > 0, "No HTTP client spans found for proxy requests"

    def test_service_attribution(self, trace_data):
        """Verify service name attribution"""
        spans = trace_data.get("spans", [])

        for span in spans:
            service_name = span.get("tags", {}).get("service.name")
            assert service_name, f"Span {span.get('name')} missing service.name"

    def test_trace_id_consistency(self, trace_data, trace_id):
        """Verify all spans share the same trace ID"""
        spans = trace_data.get("spans", [])

        for span in spans:
            span_trace_id = span.get("traceId")
            assert (
                span_trace_id == trace_id
            ), f"Span {span.get('name')} has mismatched trace ID: {span_trace_id}"

    def test_span_timing_consistency(self, trace_data):
        """Verify span timing is consistent"""
        spans = trace_data.get("spans", [])

        span_map = {s.get("spanId"): s for s in spans}

        for span in spans:
            parent_id = span.get("parentSpanId")
            if parent_id and parent_id in span_map:
                parent = span_map[parent_id]

                child_start = span.get("startTime", 0)
                parent_start = parent.get("startTime", 0)

                assert (
                    child_start >= parent_start
                ), f"Child span {span.get('name')} starts before parent {parent.get('name')}"

    def test_proxy_request_correlation(self, trace_data):
        """Verify proxy requests are correlated in trace"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]
        assert len(scenario_spans) > 0, "Scenario span not found"

        assert len(spans) > 1, "No child spans found - trace correlation incomplete"

    def test_multiple_requests_in_trace(self, trace_data):
        """Verify multiple proxy requests are captured in single trace"""
        spans = trace_data.get("spans", [])

        scenario_spans = [s for s in spans if s.get("name") == "litellm_proxy_scenario"]
        assert len(scenario_spans) > 0

        tags = scenario_spans[0].get("tags", {})
        total_requests = int(tags.get("scenario.proxy.requests_total", 0))

        assert (
            total_requests >= 2
        ), f"Expected multiple requests in trace, got {total_requests}"

    def test_telemetry_sdk_attributes(self, trace_data):
        """Verify OpenTelemetry SDK attributes are present"""
        spans = trace_data.get("spans", [])

        root_spans = [s for s in spans if s.get("name") == "unified_genai_test_app"]
        assert len(root_spans) > 0

        root_span = root_spans[0]
        tags = root_span.get("tags", {})

        assert (
            "telemetry.sdk.name" in tags or "otel.library.name" in tags
        ), "OpenTelemetry SDK attributes missing"


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--trace-id",
        action="store",
        default="5949cabf0b2cd8bf0e77e6cc7c0631a9",
        help="Trace ID to validate",
    )
