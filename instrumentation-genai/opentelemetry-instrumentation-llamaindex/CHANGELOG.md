# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/signalfx/splunk-otel-python-contrib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/signalfx/splunk-otel-python-contrib/releases/tag/v0.1.0
