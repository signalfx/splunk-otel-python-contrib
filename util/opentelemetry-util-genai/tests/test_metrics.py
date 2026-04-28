import os
import time
import unittest
from typing import Any, List, Optional, cast
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS,
)
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

STABILITY_EXPERIMENTAL: dict[str, str] = {}


class TestMetricsEmission(unittest.TestCase):
    def setUp(self):
        # Fresh tracer provider & exporter (do not rely on global replacement each time)
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        # Only set the global tracer provider once (subsequent overrides ignored but harmless)
        trace.set_tracer_provider(tracer_provider)
        self.tracer_provider = tracer_provider
        # Isolated meter provider with in-memory reader (do NOT set global to avoid override warnings)
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(
            metric_readers=[self.metric_reader]
        )
        # Reset handler singleton
        TelemetryHandler._reset_for_testing()
        # Reset handler singleton
        TelemetryHandler._reset_for_testing()

    def _invoke(
        self,
        generator: str,
        capture_mode: str,
        *,
        agent_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
    ) -> LLMInvocation:
        env = {
            **STABILITY_EXPERIMENTAL,
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: generator,
        }
        if capture_mode is not None:
            upper_mode = capture_mode.upper()
            capture_enabled = upper_mode != "NONE"
            env[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
                "true" if capture_enabled else "false"
            )
            env[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE] = (
                upper_mode
            )
        with patch.dict(os.environ, env, clear=False):
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            TelemetryHandler._reset_for_testing()
            handler = get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )
            inv = LLMInvocation(
                request_model="m",
                input_messages=[
                    InputMessage(role="user", parts=[Text(content="hi")])
                ],
            )
            inv.provider = "prov"
            inv.server_address = server_address
            inv.server_port = server_port
            # set agent identity post construction if provided
            if agent_name is not None:
                inv.agent_name = agent_name
            if agent_id is not None:
                inv.agent_id = agent_id
            handler.start_llm(inv)
            time.sleep(0.01)  # ensure measurable duration
            inv.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="ok")],
                    finish_reason="stop",
                )
            ]
            inv.input_tokens = 5
            inv.output_tokens = 7
            handler.stop_llm(inv)
            # Force flush isolated meter provider
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
            time.sleep(0.005)
            try:
                self.metric_reader.collect()
            except Exception:
                pass
        return inv

    def _invoke_failure(
        self,
        generator: str,
        *,
        error_type: type[BaseException] = RuntimeError,
    ) -> LLMInvocation:
        env = {
            **STABILITY_EXPERIMENTAL,
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: generator,
        }
        with patch.dict(os.environ, env, clear=False):
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            TelemetryHandler._reset_for_testing()
            handler = get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )
            inv = LLMInvocation(
                request_model="m",
                input_messages=[
                    InputMessage(role="user", parts=[Text(content="hi")])
                ],
            )
            inv.provider = "prov"
            handler.start_llm(inv)
            time.sleep(0.01)
            handler.fail_llm(
                inv,
                Error(
                    message="boom",
                    type=error_type,
                ),
            )
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
            time.sleep(0.005)
            try:
                self.metric_reader.collect()
            except Exception:
                pass
        return inv

    def _collect_metrics(
        self, retries: int = 3, delay: float = 0.01
    ) -> List[Any]:
        for attempt in range(retries):
            try:
                self.metric_reader.collect()
            except Exception:
                pass
            data: Any = None
            try:
                data = self.metric_reader.get_metrics_data()  # type: ignore[assignment]
            except Exception:
                data = None
            points: List[Any] = []
            if data is not None:
                data_any = cast(Any, data)
                for rm in getattr(data_any, "resource_metrics", []) or []:
                    for scope_metrics in (
                        getattr(rm, "scope_metrics", []) or []
                    ):
                        for metric in (
                            getattr(scope_metrics, "metrics", []) or []
                        ):
                            points.append(metric)
            if points or attempt == retries - 1:
                return points
            time.sleep(delay)
        return []

    def test_span_flavor_has_no_metrics(self):
        self._invoke("span", "span")
        metrics_list = self._collect_metrics()
        print(
            "[DEBUG span] collected metrics:", [m.name for m in metrics_list]
        )
        names = {m.name for m in metrics_list}
        self.assertNotIn("gen_ai.client.operation.duration", names)
        self.assertNotIn("gen_ai.client.token.usage", names)

    def test_span_metric_flavor_emits_metrics(self):
        self._invoke("span_metric", "span")
        # Probe metric to validate pipeline
        probe_hist = self.meter_provider.get_meter("probe").create_histogram(
            "probe.metric"
        )
        probe_hist.record(1)
        metrics_list = self._collect_metrics()
        print(
            "[DEBUG span_metric] collected metrics:",
            [m.name for m in metrics_list],
        )
        names = {m.name for m in metrics_list}
        self.assertIn(
            "probe.metric", names, "probe metric missing - pipeline inactive"
        )
        self.assertIn("gen_ai.client.operation.duration", names)
        self.assertIn("gen_ai.client.token.usage", names)

    def test_span_metric_event_flavor_emits_metrics(self):
        self._invoke("span_metric_event", "events")
        probe_hist = self.meter_provider.get_meter("probe2").create_histogram(
            "probe2.metric"
        )
        probe_hist.record(1)
        metrics_list = self._collect_metrics()
        print(
            "[DEBUG span_metric_event] collected metrics:",
            [m.name for m in metrics_list],
        )
        names = {m.name for m in metrics_list}
        self.assertIn(
            "probe2.metric", names, "probe2 metric missing - pipeline inactive"
        )
        self.assertIn("gen_ai.client.operation.duration", names)
        self.assertIn("gen_ai.client.token.usage", names)

    def test_llm_metrics_include_agent_identity_when_present(self):
        self._invoke(
            "span_metric",
            "span",
            agent_name="router_agent",
            agent_id="agent-123",
        )
        metrics_list = self._collect_metrics()
        # Collect token usage and duration datapoints and assert agent attrs present
        # We flatten all datapoints for easier searching
        found_token_agent = False
        found_duration_agent = False
        for metric in metrics_list:
            if metric.name not in (
                "gen_ai.client.token.usage",
                "gen_ai.client.operation.duration",
            ):
                continue
            # metric.data.data_points for Histogram-like metrics
            data = getattr(metric, "data", None)
            if not data:
                continue
            data_points = getattr(data, "data_points", []) or []
            for dp in data_points:
                attrs = getattr(dp, "attributes", {}) or {}
                if (
                    attrs.get("gen_ai.agent.name") == "router_agent"
                    and attrs.get("gen_ai.agent.id") == "agent-123"
                ):
                    if metric.name == "gen_ai.client.token.usage":
                        found_token_agent = True
                    if metric.name == "gen_ai.client.operation.duration":
                        found_duration_agent = True
        self.assertTrue(
            found_token_agent,
            "Expected token usage metric datapoint to include agent.name and agent.id",
        )
        self.assertTrue(
            found_duration_agent,
            "Expected operation duration metric datapoint to include agent.name and agent.id",
        )

    def test_llm_metrics_include_server_attributes(self):
        self._invoke(
            "span_metric",
            "span",
            server_address="llm.internal",
            server_port=8081,
        )
        metrics_list = self._collect_metrics()
        saw_duration = False
        saw_tokens = False
        for metric in metrics_list:
            if metric.name not in (
                "gen_ai.client.token.usage",
                "gen_ai.client.operation.duration",
            ):
                continue
            data = getattr(metric, "data", None)
            if not data:
                continue
            for dp in getattr(data, "data_points", []) or []:
                attrs = getattr(dp, "attributes", {}) or {}
                if (
                    attrs.get("server.address") == "llm.internal"
                    and attrs.get("server.port") == 8081
                ):
                    if metric.name == "gen_ai.client.token.usage":
                        saw_tokens = True
                    else:
                        saw_duration = True
        self.assertTrue(
            saw_duration,
            "Expected duration metric to include server.address and server.port",
        )
        self.assertTrue(
            saw_tokens,
            "Expected token usage metric to include server.address and server.port",
        )

    def test_llm_metrics_inherit_agent_identity_from_context(self):
        # Prepare environment to emit metrics
        env = {
            **STABILITY_EXPERIMENTAL,
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: "span_metric",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "true",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE: "SPAN_ONLY",
        }
        with patch.dict(os.environ, env, clear=False):
            TelemetryHandler._reset_for_testing()
            handler = get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )
            # Start an agent (push context)
            agent = AgentInvocation(
                name="context_agent",
                model="model-x",
                agent_id="agent-123",
            )
            handler.start_agent(agent)
            # Start LLM WITHOUT agent_name/id explicitly set
            inv = LLMInvocation(
                request_model="m2",
                input_messages=[
                    InputMessage(role="user", parts=[Text(content="hello")])
                ],
            )
            handler.start_llm(inv)
            time.sleep(0.01)
            inv.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="hi")],
                    finish_reason="stop",
                )
            ]
            inv.input_tokens = 3
            inv.output_tokens = 4
            handler.stop_llm(inv)
            handler.stop_agent(agent)
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
            self.metric_reader.collect()

        metrics_list = self._collect_metrics()
        inherited = False
        for metric in metrics_list:
            if metric.name not in (
                "gen_ai.client.token.usage",
                "gen_ai.client.operation.duration",
            ):
                continue
            data = getattr(metric, "data", None)
            if not data:
                continue
            for dp in getattr(data, "data_points", []) or []:
                attrs = getattr(dp, "attributes", {}) or {}
                if (
                    attrs.get("gen_ai.agent.name") == "context_agent"
                    and attrs.get("gen_ai.agent.id") == "agent-123"
                ):
                    inherited = True
                    break
        self.assertTrue(
            inherited,
            "Expected metrics to inherit agent identity from active agent context",
        )

    def test_llm_duration_metric_includes_error_type_on_failure(self):
        self._invoke_failure("span_metric")
        metrics_list = self._collect_metrics()
        duration_points: list[Any] = []
        for metric in metrics_list:
            if metric.name != "gen_ai.client.operation.duration":
                continue
            data = getattr(metric, "data", None)
            if not data:
                continue
            duration_points.extend(getattr(data, "data_points", []) or [])

        self.assertTrue(
            duration_points,
            "Expected at least one duration datapoint for failed invocation",
        )
        error_key = ErrorAttributes.ERROR_TYPE
        has_error_attr = any(
            getattr(dp, "attributes", {}).get(error_key) == "RuntimeError"
            for dp in duration_points
        )
        self.assertTrue(
            has_error_attr,
            f"Expected duration metric datapoint to include {error_key}",
        )

    def _invoke_streaming(
        self,
        generator: str,
        capture_mode: str,
        *,
        is_streaming: bool = True,
        ttfc_value: float = 0.123,
    ) -> LLMInvocation:
        """Helper to simulate streaming LLM invocation with time to first chunk."""
        env = {
            **STABILITY_EXPERIMENTAL,
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: generator,
        }
        if capture_mode is not None:
            upper_mode = capture_mode.upper()
            capture_enabled = upper_mode != "NONE"
            env[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
                "true" if capture_enabled else "false"
            )
            env[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE] = (
                upper_mode
            )
        with patch.dict(os.environ, env, clear=False):
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            TelemetryHandler._reset_for_testing()
            handler = get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )
            inv = LLMInvocation(
                request_model="gpt-4o-mini",
                input_messages=[
                    InputMessage(role="user", parts=[Text(content="hello")])
                ],
            )
            inv.provider = "openai"
            inv.request_stream = is_streaming
            if is_streaming:
                # Simulate streaming: set time to first chunk in attributes
                inv.attributes["gen_ai.response.time_to_first_chunk"] = (
                    ttfc_value
                )
            handler.start_llm(inv)
            time.sleep(0.01)  # ensure measurable duration
            inv.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hi!")],
                    finish_reason="stop",
                )
            ]
            inv.input_tokens = 3
            inv.output_tokens = 2
            handler.stop_llm(inv)
            # Force flush isolated meter provider
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
            time.sleep(0.005)
            try:
                self.metric_reader.collect()
            except Exception:
                pass
        return inv

    def test_streaming_time_to_first_chunk_metric_emitted(self):
        """Verify time_to_first_chunk metric is emitted for streaming calls."""
        ttfc_value = 0.234
        self._invoke_streaming(
            "span_metric", "span", is_streaming=True, ttfc_value=ttfc_value
        )
        metrics_list = self._collect_metrics()

        # Find the time_to_first_chunk metric
        ttfc_metric = None
        for metric in metrics_list:
            if metric.name == "gen_ai.client.operation.time_to_first_chunk":
                ttfc_metric = metric
                break

        self.assertIsNotNone(
            ttfc_metric,
            "Expected gen_ai.client.operation.time_to_first_chunk metric for streaming call",
        )

        # Verify the metric has datapoints with correct value
        data = getattr(ttfc_metric, "data", None)
        self.assertIsNotNone(data, "Expected metric to have data")
        data_points = getattr(data, "data_points", []) or []
        self.assertTrue(
            data_points, "Expected at least one datapoint for TTFC metric"
        )

        # Verify the recorded value matches
        found_correct_value = False
        for dp in data_points:
            # For histogram, check sum or bucket counts
            if hasattr(dp, "sum"):
                # The sum should be approximately the TTFC value
                if abs(dp.sum - ttfc_value) < 0.001:
                    found_correct_value = True
                    break

        self.assertTrue(
            found_correct_value,
            f"Expected TTFC metric to record value ~{ttfc_value}",
        )

        # Verify attributes match (same as duration/token metrics)
        for dp in data_points:
            attrs = getattr(dp, "attributes", {}) or {}
            self.assertEqual(
                attrs.get("gen_ai.request.model"),
                "gpt-4o-mini",
                "Expected request model in TTFC metric attributes",
            )
            self.assertEqual(
                attrs.get("gen_ai.provider.name"),
                "openai",
                "Expected provider in TTFC metric attributes",
            )

    def test_non_streaming_does_not_emit_time_to_first_chunk_metric(self):
        """Verify time_to_first_chunk metric is NOT emitted for non-streaming calls."""
        self._invoke_streaming(
            "span_metric", "span", is_streaming=False, ttfc_value=0.0
        )
        metrics_list = self._collect_metrics()

        # Verify the time_to_first_chunk metric is NOT present
        ttfc_metric = None
        for metric in metrics_list:
            if metric.name == "gen_ai.client.operation.time_to_first_chunk":
                ttfc_metric = metric
                break

        self.assertIsNone(
            ttfc_metric,
            "Expected NO gen_ai.client.operation.time_to_first_chunk metric for non-streaming call",
        )

        # Verify standard metrics are still emitted
        names = {m.name for m in metrics_list}
        self.assertIn(
            "gen_ai.client.operation.duration",
            names,
            "Expected duration metric for non-streaming call",
        )
        self.assertIn(
            "gen_ai.client.token.usage",
            names,
            "Expected token usage metric for non-streaming call",
        )


class TestMCPSessionDurationMetrics(unittest.TestCase):
    """Tests for mcp.client.session.duration and mcp.server.session.duration."""

    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        trace.set_tracer_provider(tracer_provider)
        self.tracer_provider = tracer_provider
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(
            metric_readers=[self.metric_reader]
        )
        TelemetryHandler._reset_for_testing()

    def _collect_metrics(self, retries: int = 5, delay: float = 0.1):
        for attempt in range(retries):
            try:
                self.metric_reader.collect()
            except Exception:
                pass
            data: Any = None
            try:
                data = self.metric_reader.get_metrics_data()
            except Exception:
                data = None
            points: list[Any] = []
            if data is not None:
                data_any = cast(Any, data)
                for rm in getattr(data_any, "resource_metrics", []) or []:
                    for scope_metrics in (
                        getattr(rm, "scope_metrics", []) or []
                    ):
                        for metric in (
                            getattr(scope_metrics, "metrics", []) or []
                        ):
                            points.append(metric)
            if points or attempt == retries - 1:
                return points
            time.sleep(delay)
        return []

    def _get_handler(self):
        env = {
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: "span_metric",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "true",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE: "SPAN",
        }
        with patch.dict(os.environ, env, clear=False):
            TelemetryHandler._reset_for_testing()
            return get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )

    def test_mcp_client_session_duration_on_normal_close(self):
        """mcp.client.session.duration is recorded when a client MCP session closes normally."""
        handler = self._get_handler()

        session = AgentInvocation(
            name="mcp.client",
            agent_type="mcp_client",
            system="mcp",
        )
        session.attributes["gen_ai.operation.name"] = "mcp.client_session"
        session.attributes["network.transport"] = "pipe"

        handler.start_agent(session)
        time.sleep(0.01)
        handler.stop_agent(session)

        try:
            self.meter_provider.force_flush()
        except Exception:
            pass

        metrics_list = self._collect_metrics()
        names = {m.name for m in metrics_list}

        self.assertIn(
            "mcp.client.session.duration",
            names,
            f"Expected mcp.client.session.duration in {names}",
        )

        # Verify attributes
        for metric in metrics_list:
            if metric.name != "mcp.client.session.duration":
                continue
            data = getattr(metric, "data", None)
            if not data:
                continue
            for dp in getattr(data, "data_points", []) or []:
                attrs = getattr(dp, "attributes", {}) or {}
                self.assertEqual(attrs.get("network.transport"), "pipe")
                self.assertNotIn("error.type", attrs)

    def test_mcp_client_session_duration_on_error_close(self):
        """mcp.client.session.duration includes error.type when session fails."""
        handler = self._get_handler()

        session = AgentInvocation(
            name="mcp.client",
            agent_type="mcp_client",
            system="mcp",
        )
        session.attributes["gen_ai.operation.name"] = "mcp.client_session"
        session.attributes["network.transport"] = "pipe"
        session.attributes["error.type"] = "ConnectionError"

        handler.start_agent(session)
        time.sleep(0.01)
        handler.fail_agent(
            session, Error(type=ConnectionError, message="connection lost")
        )

        try:
            self.meter_provider.force_flush()
        except Exception:
            pass

        metrics_list = self._collect_metrics()
        names = {m.name for m in metrics_list}

        self.assertIn(
            "mcp.client.session.duration",
            names,
            f"Expected mcp.client.session.duration in {names}",
        )

        # Verify error.type attribute
        for metric in metrics_list:
            if metric.name != "mcp.client.session.duration":
                continue
            data = getattr(metric, "data", None)
            if not data:
                continue
            for dp in getattr(data, "data_points", []) or []:
                attrs = getattr(dp, "attributes", {}) or {}
                self.assertEqual(attrs.get("error.type"), "ConnectionError")

    def test_mcp_server_session_duration_recorded(self):
        """mcp.server.session.duration is recorded for server-side MCP sessions."""
        handler = self._get_handler()

        session = AgentInvocation(
            name="mcp.server",
            agent_type="mcp_server",
            system="mcp",
        )
        session.attributes["gen_ai.operation.name"] = "mcp.server_session"
        session.attributes["network.transport"] = "tcp"

        handler.start_agent(session)
        time.sleep(0.01)
        handler.stop_agent(session)

        try:
            self.meter_provider.force_flush()
        except Exception:
            pass

        metrics_list = self._collect_metrics()
        names = {m.name for m in metrics_list}

        self.assertIn(
            "mcp.server.session.duration",
            names,
            f"Expected mcp.server.session.duration in {names}",
        )

    def test_non_mcp_agent_does_not_emit_session_duration(self):
        """Regular agent invocations should NOT emit mcp.*.session.duration."""
        handler = self._get_handler()

        agent = AgentInvocation(
            name="regular_agent",
            agent_type="researcher",
        )

        handler.start_agent(agent)
        time.sleep(0.01)
        handler.stop_agent(agent)

        try:
            self.meter_provider.force_flush()
        except Exception:
            pass

        metrics_list = self._collect_metrics()
        names = {m.name for m in metrics_list}

        self.assertNotIn("mcp.client.session.duration", names)
        self.assertNotIn("mcp.server.session.duration", names)
        self.assertIn("gen_ai.agent.duration", names)

    # ---- Embedding token metrics tests ----

    def _invoke_embedding(
        self,
        generator: str,
        *,
        input_tokens: int | None = 10,
    ) -> EmbeddingInvocation:
        env = {
            **STABILITY_EXPERIMENTAL,
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: generator,
        }
        with patch.dict(os.environ, env, clear=False):
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            if hasattr(get_telemetry_handler, "_default_handler"):
                delattr(get_telemetry_handler, "_default_handler")
            handler = get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )
            emb = EmbeddingInvocation(
                request_model="text-embedding-ada-002",
                input_texts=["hello world"],
                provider="openai",
            )
            handler.start_embedding(emb)
            time.sleep(0.01)  # ensure measurable duration
            emb.input_tokens = input_tokens
            handler.stop_embedding(emb)
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
            time.sleep(0.005)
            try:
                self.metric_reader.collect()
            except Exception:
                pass
        return emb

    def _invoke_embedding_failure(
        self,
        generator: str,
        *,
        input_tokens: int | None = 10,
        error_type: type[BaseException] = RuntimeError,
    ) -> EmbeddingInvocation:
        env = {
            **STABILITY_EXPERIMENTAL,
            OTEL_INSTRUMENTATION_GENAI_EMITTERS: generator,
        }
        with patch.dict(os.environ, env, clear=False):
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            if hasattr(get_telemetry_handler, "_default_handler"):
                delattr(get_telemetry_handler, "_default_handler")
            handler = get_telemetry_handler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )
            emb = EmbeddingInvocation(
                request_model="text-embedding-ada-002",
                input_texts=["hello world"],
                provider="openai",
            )
            handler.start_embedding(emb)
            time.sleep(0.01)
            emb.input_tokens = input_tokens
            handler.fail_embedding(
                emb,
                Error(message="boom", type=error_type),
            )
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
            time.sleep(0.005)
            try:
                self.metric_reader.collect()
            except Exception:
                pass
        return emb

    def test_embedding_emits_input_token_metric(self):
        """Embedding should emit token.usage with operation_name=embeddings and only input token type."""
        self._invoke_embedding("span_metric")
        metrics_list = self._collect_metrics()
        names = {m.name for m in metrics_list}
        self.assertIn("gen_ai.client.token.usage", names)
        self.assertIn("gen_ai.client.operation.duration", names)
        # Collect all embedding token datapoints
        token_types = []
        for metric in metrics_list:
            if metric.name != "gen_ai.client.token.usage":
                continue
            data = getattr(metric, "data", None)
            if not data:
                continue
            for dp in getattr(data, "data_points", []) or []:
                attrs = getattr(dp, "attributes", {}) or {}
                self.assertEqual(
                    attrs.get("gen_ai.operation.name"),
                    "embeddings",
                )
                token_types.append(attrs.get("gen_ai.token.type"))
        # Only input tokens, no output/completion
        self.assertEqual(token_types, ["input"])

    def test_embedding_failure_emits_token_metric(self):
        """Embedding failure path should also emit token usage metric."""
        self._invoke_embedding_failure("span_metric", input_tokens=8)
        metrics_list = self._collect_metrics()
        names = {m.name for m in metrics_list}
        self.assertIn("gen_ai.client.token.usage", names)
        self.assertIn("gen_ai.client.operation.duration", names)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
