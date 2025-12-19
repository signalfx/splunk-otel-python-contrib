# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-25

### Added
- Initial release of CrewAI instrumentation
- Wrapper-based instrumentation for CrewAI workflows, agents, tasks, and tools
- Support for `Crew.kickoff()` → `Workflow` spans
- Support for `Task.execute_sync()` → `Step` spans
- Support for `Agent.execute_task()` → `AgentInvocation` spans
- Support for `BaseTool.run()` and `CrewStructuredTool.invoke()` → `ToolCall` spans
- Integration with `splunk-otel-util-genai` for standardized GenAI telemetry
- Proper trace context propagation using `contextvars`
- Rich span attributes for all CrewAI components
- Defensive instrumentation that doesn't break applications on errors

### Documentation
- Comprehensive README with usage examples
- Compositional instrumentation patterns (CrewAI + OpenAI + Vector Stores)
- Configuration and environment variable documentation

### Limitations
- Synchronous workflows only (async support planned for future release)
- LLM calls not instrumented (use provider-specific instrumentation)

[Unreleased]: https://github.com/signalfx/splunk-otel-python-contrib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/signalfx/splunk-otel-python-contrib/releases/tag/v0.1.0

