# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Streaming TTFT (Time To First Token) support for LLM spans:
  - `gen_ai.response.time_to_first_chunk` attribute measuring latency to first streaming token
  - `gen_ai.request.stream` flag (true when streaming detected, false otherwise)
- TTFT tracking via LlamaIndex event system (`event_handler.py`):
  - `TTFTTracker` class for recording start times and calculating TTFT
  - `LlamaindexEventHandler` listening to `LLMChatStartEvent`/`LLMChatInProgressEvent` for per-chunk timing
  - ContextVar correlation bridging callback handler (event_id) with event handler (span_id)

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
- Corrected retrieval span `gen_ai.operation.name` from `"retrieve"` to `"retrieval"` per OpenTelemetry semantic conventions. Removed explicit override in callback handler; now uses the `RetrievalInvocation` dataclass default from `util-genai`.

## [0.1.1] - 2026-01-30

### Fixed
- Align workflow and agent instrumentation with `input_messages` to match updated GenAI types
- Avoid invalid `initial_input` and `input_context` arguments in workflow spans

## [0.1.0] - 2025-01-29

### Added

- Initial release of LlamaIndex instrumentation
- Callback-based instrumentation for LLM and embedding operations
- Support for LLM invocations → `LLMInvocation` spans with rich attributes
- Support for embedding operations → `LLMInvocation` spans with model and token metrics
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
