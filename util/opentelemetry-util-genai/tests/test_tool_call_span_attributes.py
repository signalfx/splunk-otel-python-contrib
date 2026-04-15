"""Tests for ToolCall, MCPToolCall, and MCPOperation span attributes."""

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.emitters.metrics import MetricsEmitter
from opentelemetry.util.genai.emitters.span import SpanEmitter
from opentelemetry.util.genai.types import (
    Error,
    MCPOperation,
    MCPToolCall,
    ToolCall,
)


def _make_emitter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=False)
    return emitter, exporter


# --- Plain ToolCall tests ---


def test_tool_call_span_attributes():
    emitter, exporter = _make_emitter()
    call = ToolCall(
        name="summarize",
        id="tool-1",
        arguments={"text": "hello"},
        provider="provX",
    )
    emitter.on_start(call)
    assert call.span is not None
    attrs = dict(call.span.attributes)
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"
    assert attrs.get("gen_ai.provider.name") == "provX"
    emitter.on_end(call)


def test_tool_call_span_name_and_kind():
    """Plain ToolCall uses 'execute_tool {name}' span name and INTERNAL kind."""
    emitter, exporter = _make_emitter()
    call = ToolCall(name="summarize", id="tool-1")
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "execute_tool summarize"
    assert spans[0].kind == SpanKind.INTERNAL


# --- MCPToolCall tests ---


def test_mcp_tool_call_span_name_uses_method():
    """MCPToolCall span name follows MCP semconv: {mcp.method.name} {tool.name}."""
    emitter, exporter = _make_emitter()
    call = MCPToolCall(
        name="add",
        id="tool-1",
        mcp_method_name="tools/call",
        is_client=True,
    )
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "tools/call add"


def test_mcp_tool_call_client_span_kind():
    """MCPToolCall with is_client=True produces CLIENT SpanKind."""
    emitter, exporter = _make_emitter()
    call = MCPToolCall(
        name="add",
        mcp_method_name="tools/call",
        is_client=True,
    )
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].kind == SpanKind.CLIENT


def test_mcp_tool_call_server_span_kind():
    """MCPToolCall with is_client=False produces SERVER SpanKind."""
    emitter, exporter = _make_emitter()
    call = MCPToolCall(
        name="add",
        mcp_method_name="tools/call",
        is_client=False,
    )
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].kind == SpanKind.SERVER


def test_mcp_tool_call_has_execute_tool_operation():
    """MCPToolCall still sets gen_ai.operation.name = execute_tool."""
    emitter, exporter = _make_emitter()
    call = MCPToolCall(
        name="add",
        mcp_method_name="tools/call",
        is_client=True,
    )
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"


def test_mcp_tool_call_semconv_attributes():
    """MCPToolCall emits MCP semconv attributes on the span."""
    emitter, exporter = _make_emitter()
    call = MCPToolCall(
        name="add",
        mcp_method_name="tools/call",
        network_transport="pipe",
        mcp_session_id="sess-123",
        mcp_protocol_version="2025-03-26",
        sdot_mcp_server_name="math-tools",
        jsonrpc_request_id="42",
        server_address="localhost",
        server_port=8080,
        is_client=False,
    )
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs.get("mcp.method.name") == "tools/call"
    assert attrs.get("network.transport") == "pipe"
    assert attrs.get("mcp.session.id") == "sess-123"
    assert attrs.get("mcp.protocol.version") == "2025-03-26"
    assert attrs.get("sdot.mcp.server_name") == "math-tools"
    assert attrs.get("gen_ai.tool.name") == "add"
    assert attrs.get("jsonrpc.request.id") == "42"
    assert attrs.get("server.address") == "localhost"
    assert attrs.get("server.port") == 8080


def test_mcp_tool_call_defaults_method_name():
    """MCPToolCall defaults mcp_method_name to 'tools/call' in span name."""
    emitter, exporter = _make_emitter()
    call = MCPToolCall(
        name="add",
        mcp_method_name=None,
        is_client=True,
    )
    emitter.on_start(call)
    emitter.on_end(call)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    # When mcp_method_name is None, span name is just the tool name
    assert spans[0].name == "add"


# --- MCPOperation tests ---


def test_mcp_operation_list_tools_span():
    """MCPOperation for tools/list produces correct span name and CLIENT kind."""
    emitter, exporter = _make_emitter()
    op = MCPOperation(
        target="",
        mcp_method_name="tools/list",
        network_transport="pipe",
        is_client=True,
    )
    emitter.on_start(op)
    emitter.on_end(op)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "tools/list"
    assert spans[0].kind == SpanKind.CLIENT


def test_mcp_operation_resources_read_span():
    """MCPOperation for resources/read has resource URI as target."""
    emitter, exporter = _make_emitter()
    op = MCPOperation(
        target="file:///config.json",
        mcp_method_name="resources/read",
        network_transport="pipe",
        mcp_resource_uri="file:///config.json",
        is_client=True,
    )
    emitter.on_start(op)
    emitter.on_end(op)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "resources/read file:///config.json"
    assert spans[0].kind == SpanKind.CLIENT
    attrs = dict(spans[0].attributes)
    assert attrs.get("mcp.resource.uri") == "file:///config.json"


def test_mcp_operation_prompts_get_span():
    """MCPOperation for prompts/get has prompt name as target."""
    emitter, exporter = _make_emitter()
    op = MCPOperation(
        target="summarize",
        mcp_method_name="prompts/get",
        network_transport="pipe",
        gen_ai_prompt_name="summarize",
        is_client=False,
    )
    emitter.on_start(op)
    emitter.on_end(op)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "prompts/get summarize"
    assert spans[0].kind == SpanKind.SERVER
    attrs = dict(spans[0].attributes)
    assert attrs.get("gen_ai.prompt.name") == "summarize"


def test_mcp_operation_server_kind():
    """MCPOperation with is_client=False produces SERVER SpanKind."""
    emitter, exporter = _make_emitter()
    op = MCPOperation(
        target="",
        mcp_method_name="tools/list",
        is_client=False,
    )
    emitter.on_start(op)
    emitter.on_end(op)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].kind == SpanKind.SERVER


def test_mcp_operation_new_semconv_attrs():
    """MCPOperation emits all new GAP-2 semconv attributes."""
    emitter, exporter = _make_emitter()
    op = MCPOperation(
        target="",
        mcp_method_name="tools/list",
        network_transport="tcp",
        network_protocol_name="http",
        network_protocol_version="2",
        server_address="mcp.example.com",
        server_port=443,
        client_address="10.0.0.1",
        client_port=54321,
        mcp_session_id="sess-abc",
        jsonrpc_request_id="7",
        is_client=True,
    )
    emitter.on_start(op)
    emitter.on_end(op)

    spans = exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs.get("network.transport") == "tcp"
    assert attrs.get("network.protocol.name") == "http"
    assert attrs.get("network.protocol.version") == "2"
    assert attrs.get("server.address") == "mcp.example.com"
    assert attrs.get("server.port") == 443
    assert attrs.get("client.address") == "10.0.0.1"
    assert attrs.get("client.port") == 54321
    assert attrs.get("mcp.session.id") == "sess-abc"
    assert attrs.get("jsonrpc.request.id") == "7"


# --- MCPToolCall metrics error-path tests ---


def _make_metrics_emitter():
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    meter = meter_provider.get_meter(__name__)
    emitter = MetricsEmitter(meter=meter)
    return emitter, reader, meter_provider


def _collect_metric_names(reader, meter_provider):
    """Flush and return the set of metric names that have data points."""
    try:
        meter_provider.force_flush()
    except Exception:
        pass
    names = set()
    for metric in reader.get_metrics_data().resource_metrics:
        for sm in metric.scope_metrics:
            for m in sm.metrics:
                if hasattr(m, "data") and getattr(m.data, "data_points", None):
                    if len(m.data.data_points) > 0:
                        names.add(m.name)
    return names


def test_mcp_tool_call_error_records_generic_duration():
    """on_error for MCPToolCall must record both MCP and generic duration."""
    emitter, reader, meter_provider = _make_metrics_emitter()
    tool = MCPToolCall(
        name="divide",
        id="tc-1",
        mcp_method_name="tools/call",
        network_transport="pipe",
        is_client=True,
        framework="fastmcp",
    )
    tool.duration_s = 0.05

    emitter.on_error(
        Error(type=ValueError, message="division by zero"),
        tool,
    )

    names = _collect_metric_names(reader, meter_provider)
    assert "mcp.client.operation.duration" in names, (
        "MCP-specific duration metric should be recorded on error"
    )
    assert "gen_ai.client.operation.duration" in names, (
        "Generic execute-tool duration metric should also be recorded on error"
    )


def test_mcp_operation_error_does_not_record_generic_duration():
    """on_error for plain MCPOperation (non-tool) should only record MCP metric."""
    emitter, reader, meter_provider = _make_metrics_emitter()
    op = MCPOperation(
        target="",
        mcp_method_name="tools/list",
        network_transport="pipe",
        is_client=True,
    )
    op.duration_s = 0.03

    emitter.on_error(Error(type=RuntimeError, message="oops"), op)

    names = _collect_metric_names(reader, meter_provider)
    assert "mcp.client.operation.duration" in names
    assert "gen_ai.client.operation.duration" not in names


# --- MetricsEmitter.handles() tests ---


def test_handles_accepts_mcp_operation():
    """MetricsEmitter.handles() returns True for plain MCPOperation."""
    emitter, _, _ = _make_metrics_emitter()
    op = MCPOperation(target="", mcp_method_name="tools/list", is_client=True)
    assert emitter.handles(op) is True


def test_handles_accepts_mcp_tool_call():
    """MetricsEmitter.handles() returns True for MCPToolCall."""
    emitter, _, _ = _make_metrics_emitter()
    tc = MCPToolCall(name="add", mcp_method_name="tools/call", is_client=True)
    assert emitter.handles(tc) is True


def test_handles_accepts_plain_tool_call():
    """MetricsEmitter.handles() returns True for plain ToolCall."""
    emitter, _, _ = _make_metrics_emitter()
    tc = ToolCall(name="summarize", id="tc-1")
    assert emitter.handles(tc) is True


# --- MCPOperation on_end metrics tests ---


def _collect_metric_data_points(reader, meter_provider, metric_name):
    """Flush and return data points for a specific metric."""
    try:
        meter_provider.force_flush()
    except Exception:
        pass
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for sm in resource_metrics.scope_metrics:
            for m in sm.metrics:
                if m.name == metric_name and hasattr(m, "data"):
                    return list(m.data.data_points)
    return []


def test_mcp_operation_on_end_records_client_duration():
    """on_end for plain MCPOperation records mcp.client.operation.duration."""
    emitter, reader, meter_provider = _make_metrics_emitter()
    op = MCPOperation(
        target="",
        mcp_method_name="tools/list",
        network_transport="pipe",
        is_client=True,
    )
    op.duration_s = 0.05

    emitter.on_end(op)

    names = _collect_metric_names(reader, meter_provider)
    assert "mcp.client.operation.duration" in names


def test_mcp_operation_on_end_records_server_duration():
    """on_end for server-side MCPOperation records mcp.server.operation.duration."""
    emitter, reader, meter_provider = _make_metrics_emitter()
    op = MCPOperation(
        target="",
        mcp_method_name="resources/read",
        network_transport="tcp",
        is_client=False,
    )
    op.duration_s = 0.02

    emitter.on_end(op)

    names = _collect_metric_names(reader, meter_provider)
    assert "mcp.server.operation.duration" in names


def test_mcp_operation_metrics_have_correct_method_name():
    """MCP operation metrics include mcp.method.name attribute."""
    emitter, reader, meter_provider = _make_metrics_emitter()

    for method in (
        "tools/list",
        "resources/read",
        "prompts/get",
        "prompts/list",
    ):
        op = MCPOperation(
            target="",
            mcp_method_name=method,
            network_transport="pipe",
            is_client=True,
        )
        op.duration_s = 0.01
        emitter.on_end(op)

    data_points = _collect_metric_data_points(
        reader, meter_provider, "mcp.client.operation.duration"
    )
    recorded_methods = {
        dp.attributes.get("mcp.method.name") for dp in data_points
    }
    assert "tools/list" in recorded_methods
    assert "resources/read" in recorded_methods
    assert "prompts/get" in recorded_methods
    assert "prompts/list" in recorded_methods
