# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- **`resources/read` and `prompts/get` instrumentation** — Server and client-side hooks for `FastMCP.read_resource` / `Client.read_resource` and `FastMCP.get_prompt` / `Client.get_prompt`. Produces `MCPOperation` spans with `{mcp.method.name} {target}` naming.
- **Transport context bridge** — `MCPRequestContext` ContextVar populated by the transport instrumentor on the server side, allowing the server instrumentor to read `jsonrpc.request.id`, `network.transport`, etc.
- **Transport detection** — Client automatically detects `pipe` vs `tcp` transport from `Client.transport` type.
- **Baggage propagation** — Transport instrumentor now extracts W3C `baggage` header alongside `traceparent`/`tracestate`.
- **`error.type` attribute (GAP 5)** — Set on spans and metrics when an MCP operation fails: exception class name for exceptions, `"tool_error"` when `CallToolResult.isError` is true.
- **HTTP transport metadata (GAP 5)** — For `tcp` transport: `network.protocol.name`, `network.protocol.version`, `server.address`/`server.port` (client spans), `client.address`/`client.port` (server spans).
- **`mcp.session.id` attribute (GAP 5)** — Extracted from client session and server HTTP `mcp-session-id` header.
- **`mcp.protocol.version` attribute (GAP 5)** — Populated from MCP initialize handshake result.
- **`rpc.response.status_code` on MCP metrics** — Included in MCP operation duration metric attributes when set.
- **`gen_ai.prompt.name` on MCP metrics** — Included in MCP operation duration metric attributes for `prompts/get`.

### Changed
- **`list_tools` uses `MCPOperation` instead of `Step`** — Client `list_tools` now produces a `tools/list` span via `MCPOperation` with proper MCP semconv naming and `SpanKind.CLIENT`, instead of the previous `Step` type.
- **Server hooks on `fastmcp.server.server.FastMCP`** — Tool call hook now targets `FastMCP.call_tool` directly (in addition to the legacy `ToolManager.call_tool` path) for compatibility with FastMCP 3.x.
- **Renamed `mcp_server_name` → `sdot_mcp_server_name`** — **Breaking**: callers using `mcp_server_name=` on `MCPToolCall` must update to `sdot_mcp_server_name=`.

### Fixed
- **MCP span naming aligned with OTel MCP semantic conventions** — Tool call spans now use `tools/call {tool_name}` format with `SpanKind.CLIENT` (client-side) or `SpanKind.SERVER` (server-side), matching the [OTel MCP semconv spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/). Previously used `execute_tool {tool_name}` with `SpanKind.INTERNAL`.

## [0.1.1] - 2026-01-27

### Changed
- Updated to use structured message fields aligned with `opentelemetry-util-genai` v0.1.8
  - Tool calls now use `arguments` and `tool_result` fields instead of `input_data`/`output_data`
  - Step no longer captures input/output directly; tool discovery info moved to `step.attributes["mcp.tools.discovered"]`

## [0.1.0] - 2026-01-27

### Added
- Initial release of Splunk OpenTelemetry instrumentation for FastMCP (Model Context Protocol)
- Server-side instrumentation for FastMCP server initialization and tool execution
- Client-side instrumentation for FastMCP client session lifecycle and tool calls
- Automatic W3C TraceContext propagation between MCP client and server processes via `_meta` field
- Support for all MCP transports: stdio, SSE, and streamable-http
- Span generation for server operations (`mcp.server`), tool execution (`{tool_name}.tool`), and client sessions (`mcp.client`)
- Metrics collection with `gen_ai.mcp.tool.duration` histogram for tool execution duration
- Event generation for tool inputs and outputs when content capture is enabled
- Integration with `splunk-otel-util-genai` for standardized GenAI telemetry
- Environment variable configuration support:
  - `OTEL_INSTRUMENTATION_GENAI_ENABLE` - Enable/disable instrumentation
  - `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` - Capture tool arguments and results
  - `OTEL_INSTRUMENTATION_GENAI_EMITTERS` - Select emitters (span, metric, event)
- Programmatic and auto-instrumentation support
- Example applications demonstrating instrumentation usage
