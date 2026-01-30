# Changelog

All notable changes to this project will be documented in this file.

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
