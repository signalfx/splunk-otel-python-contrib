# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1]

### Fixed
- **Relaxed `splunk-otel-util-genai` dependency** — bumped from `>=0.1.4,<=0.1.8` to `>=0.1.13,<0.2`. The previous pin was inconsistent with the 0.2.0 implementation, which already depends on APIs added in util-genai 0.1.13 (`MCPOperation`, `MCPRequestContext`, `sdot_mcp_server_name`, `start_mcp_operation` / `stop_mcp_operation` lifecycle, and the async-context fix in `TelemetryHandler._push_current_span`). The outdated upper bound forced `pip install --no-deps` workarounds for users installing from source.

## [0.2.0]

### Added
- **FastMCP 3.x support** — **Breaking**: targets `fastmcp >= 3.0.0, < 4` (previously `>= 2.0.0, <= 2.14.7`).
- **Server session lifecycle tracking** — `server_instrumentor` now wraps `mcp.server.lowlevel.Server.run` with an `AgentInvocation(agent_type="mcp_server")` to track server session duration, enabling `mcp.server.session.duration` metric emission via `MetricsEmitter`.
- **`resources/read` and `prompts/get` instrumentation** — Server and client-side hooks for `FastMCP.read_resource` / `Client.read_resource` and `FastMCP.get_prompt` / `Client.get_prompt`. Produces `MCPOperation` spans with `{mcp.method.name} {target}` naming.
- **Transport context bridge** — `MCPRequestContext` ContextVar populated by the transport instrumentor on the server side, allowing the server instrumentor to read `jsonrpc.request.id`, `network.transport`, etc.
- **Transport detection** — Client automatically detects `pipe` vs `tcp` transport from `Client.transport` type.
- **Baggage propagation** — Transport instrumentor now extracts W3C `baggage` header alongside `traceparent`/`tracestate`.

### Changed
- **`list_tools` uses `MCPOperation` instead of `Step`** — Client `list_tools` now produces a `tools/list` span via `MCPOperation` with proper MCP semconv naming and `SpanKind.CLIENT`, instead of the previous `Step` type.
- **Server hooks on `FastMCP.call_tool`** — Tool call hook targets `FastMCP.call_tool` directly with re-entrant guard for FastMCP 3.x middleware recursion.
- **Renamed `mcp_server_name` → `sdot_mcp_server_name`** — **Breaking**: callers using `mcp_server_name=` on `MCPToolCall` must update to `sdot_mcp_server_name=`.

### Fixed
- **MCP session attributes for duration metrics** — `client_instrumentor` now sets `network.transport` and `error.type` on the `AgentInvocation` attributes dict so that `MetricsEmitter` can record `mcp.client.session.duration` with proper semconv attributes.
- **MCP span naming aligned with OTel MCP semantic conventions** — Tool call spans now use `tools/call {tool_name}` format with `SpanKind.CLIENT` (client-side) or `SpanKind.SERVER` (server-side), matching the [OTel MCP semconv spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/). Previously used `execute_tool {tool_name}` with `SpanKind.INTERNAL`.

## [0.1.2]

### Changed
- Pinned compatibility to `fastmcp >= 2.0.0, <= 2.14.7` and `splunk-otel-util-genai <= 0.1.8` to avoid runtime incompatibilities introduced by newer upstream releases.

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
