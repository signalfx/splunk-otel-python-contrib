# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.10 - 2026-02-06

### Fixed
- **Logging visibility for evals module** - INFO level messages from `opentelemetry.util.genai.evals.*` modules are now always visible (e.g., "Using separate process evaluation mode"). DEBUG level messages require `OTEL_INSTRUMENTATION_GENAI_DEBUG=true`.

## Version 0.1.9 - 2026-01-29

- Release 0.1.9

## Version 0.1.8 - 2026-01-27

### Changed
- **Structured message fields for agentic types** - Added `input_messages` and `output_messages` fields to `Workflow`, `AgentCreation`, and `AgentInvocation` types for properly structured message capture with valid JSON serialization.
- **Made `finish_reason` optional in `OutputMessage`** - `finish_reason` is now optional (defaults to `None`) since it's only meaningful for LLM responses, not agent/workflow outputs.
- **Deprecated legacy string fields** - Legacy fields (`initial_input`, `final_output`, `input_context`, `output_result`) are now marked as deprecated with metadata. Use structured `input_messages`/`output_messages` instead.
- **Removed input/output from `Step`** - `Step` type no longer captures input/output data as it was redundant with agent-level capture.
- **Single metric mode now default** - `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC` now defaults to `true`, emitting all evaluation scores to a unified `gen_ai.evaluation.score` histogram with evaluation type in the `gen_ai.evaluation.name` attribute. 

### Fixed
- **JSON serialization for agent/workflow messages** - Agent and workflow input/output messages are now properly serialized as valid JSON instead of Python repr strings, fixing rendering issues in observability UIs.

## Version 0.1.7 - 2026-01-26

### Added
- **Concurrent Evaluation Support** - New environment variables for parallel evaluation processing:
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` - Enable concurrent evaluation mode with multiple worker threads
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` - Configure number of worker threads (default: 4)
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` - Set bounded queue size for backpressure (0 = unbounded)
- These variables enable significant throughput improvements for LLM-as-a-Judge evaluations
- Added `OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE` environment variable to support evaluation queue size.
  Added queue size check and error attribute on span while eval for a span is dropped.
- **Explicit histogram bucket boundaries** per [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md):
  - `gen_ai.client.operation.duration`, `gen_ai.workflow.duration`, `gen_ai.agent.duration`: exponential buckets in seconds
  - `gen_ai.client.token.usage`: exponential buckets for token counts
  - `gen_ai.evaluation.score` and evaluation metrics: linear buckets [0.1 - 1.0] for score range

## Version 0.1.6 - 2026-01-13

- Added `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC` environment variable to support single evaluation metric.

## Version 0.1.5 - 2025-12-19

- Added `GEN_AI_SECURITY_EVENT_ID` semantic convention attribute for Cisco AI Defense integration
- Added `security_event_id` field to `LLMInvocation` dataclass with semconv metadata

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-util-genai
