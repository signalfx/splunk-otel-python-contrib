# Changelog
All notable changes to this repository are documented in this file.

## Version 0.2.0 - 2026-03-03

### Changed
- Upgraded `opentelemetry-semantic-conventions` dependency from `~= 0.57b0` to `>= 0.60b1` for compatibility with `splunk-otel-python >= 2.9.0`.

## Version 0.1.7 - 2026-02-09

### Fixed
- **Content Logger** Use content_logger (Logs API) instead of event_logger (Events API) for SplunkConversationEventsEmitter and SplunkEvaluationResultsEmitter to ensure log record emission is consistent with evaluation emitter behavior.

## Version 0.1.6 - 2026-01-29

### Added
- **Workflow event support** - `SplunkConversationEventsEmitter` now handles `Workflow` objects and emits `gen_ai.client.inference.operation.details` events for workflow invocations, consistent with agent and LLM events.
- **Workflow evaluation support** - `SplunkEvaluationResultsEmitter` now accepts `Workflow` objects in its `handles()` method.

### Fixed
- **Missing workflow events** - Fixed issue where workflow events were not being emitted when using `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,splunk"` configuration in CrewAI and other agentic frameworks.

## Version 0.1.5 - 2026-01-27

- Release 0.1.5 of splunk-otel-genai-emitters-splunk

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-genai-emitters-splunk
