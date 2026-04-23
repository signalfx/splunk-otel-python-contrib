# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LLM span attributes for async agent flows:
  - `gen_ai.response.model` extracted from raw LLM response with fallback to request model
  - `gen_ai.response.finish_reasons` from response choices
  - `gen_ai.request.max_tokens` with fallback chain (serialized -> metadata -> Settings.llm)
  - `gen_ai.tool.definitions` via agent context propagation (gated by `OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS`)
  - Provider detection from LLM class name
- Agent tool registration in `wrap_agent_run()` for tool definitions propagation across async boundaries
- `find_agent_with_tools()` fallback in invocation manager when ContextVar does not propagate across asyncio tasks

### Fixed
- Fixed potential memory leak in `_InvocationManager` where orphaned invocation entries could accumulate in long-running processes. Added TTL-based eviction (5-minute TTL, 1-minute check interval).

### Added- Support for embedding operations → `LLMInvocation` spans with model and token metrics
- Workflow-based agent instrumentation (ReActAgent, FunctionAgent)
- Support for `agent.run()` → `AgentInvocation` and `Workflow` spans
- Support for tool calls → `ToolCall` spans with tool name, input, and output
- Integration with `splunk-otel-util-genai` for standardized GenAI telemetry
- Proper trace context propagation through callback handlers
- Rich span attributes including:
  - `gen_ai.request.model`, `gen_ai.response.model`
  - `gen_ai.input.messages`, `gen_ai.output.messages`
  - `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
  - `gen_ai.operation.name`, `gen_ai.framework`
- Content capture support for prompt and response messages
- Multiple emitter support: span, metrics, and content events
- Defensive instrumentation that doesn't break applications on errors

### Documentation

- Comprehensive README with usage examples
- LLM and embedding test examples
- Multi-agent travel planner example application
- Configuration and environment variable documentation
- Quick start guide with span_metric emitter setup

### Limitations

- Message content capture disabled by default (enable via `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`)
- System prompts may not appear as separate messages in `gen_ai.input.messages` (framework-dependent)

[Unreleased]: https://github.com/signalfx/splunk-otel-python-contrib/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/signalfx/splunk-otel-python-contrib/releases/tag/v0.1.1
[0.1.0]: https://github.com/signalfx/splunk-otel-python-contrib/releases/tag/v0.1.0
