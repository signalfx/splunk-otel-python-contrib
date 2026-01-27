# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
