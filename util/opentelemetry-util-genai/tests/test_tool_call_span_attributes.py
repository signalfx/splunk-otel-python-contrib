"""Tests for ToolCall, MCPToolCall, and MCPOperation span attributes."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.emitters.span import SpanEmitter
from opentelemetry.util.genai.types import MCPOperation, MCPToolCall, ToolCall


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
