# Changelog

All notable changes to this project will be documented in this file.

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
