# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Populate LLMInvocation semantic convention fields from span_data:
  - `gen_ai.request.temperature` from model_config
  - `gen_ai.request.max_tokens` from model_config
  - `gen_ai.response.finish_reasons` from output
  - `gen_ai.tool.definitions` from model_config (when capture enabled)

## [0.1.3] - 2026-04-07

### Fixed
- Fixed all failing tests after PR #228 (run_id removal) and PR #244 (singleton TelemetryHandler): updated singleton reset to `TelemetryHandler._reset_for_testing()`, added environment and state-cleanup fixtures, re-enabled skipped integration tests, and corrected assertions to use semantic convention enums and `_InvocationState`.
- Fixed `GenAISemanticProcessor` to populate agent `input_messages` from `span_data.input` on span start and set `tool_type=FUNCTION` on tool invocations.
- Added configurable `agent_name` kwarg to `OpenAIAgentsInstrumentor.instrument()`.

## [0.1.2] - 2026-01-30

### Fixed
- Changed dependency from `opentelemetry-util-genai` to `splunk-otel-util-genai >= 0.1.9` to fix missing `AgentCreation` type issue.

## [0.1.1] - 2026-01-29

### Added
- Support for `initial_request` in trace metadata to populate workflow `gen_ai.input.messages` attribute.

### Fixed
- Workflow's input messages are now populated from trace metadata `initial_request` when available.

## [0.1.0] - 2026-01-26

### Added
- Initial release of Splunk OpenTelemetry instrumentation for OpenAI Agents SDK.
- `GenAISemanticProcessor` for processing spans with GenAI semantic conventions.
- Support for agent, LLM (chat), tool, workflow, handoff, guardrail, and response spans.
- Metrics collection with duration and token usage histograms.
- Multi-agent workflow support with proper span hierarchy.
- Example applications: travel-concierge and travel-planner demos.

### Fixed
- Cleaned up duplicate and unused imports in span_processor.py.
