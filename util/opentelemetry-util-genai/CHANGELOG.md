# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.12

### Added
- **Client-side streaming attributes** — New attributes for observing streaming LLM requests:
  - `gen_ai.request.stream` (boolean) — Whether the request uses streaming mode
  - `gen_ai.response.time_to_first_chunk` (float) — Client-side time in seconds from request sent to first chunk received
  - Added `request_stream` field to `LLMInvocation` dataclass with semconv metadata
  - Added `GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK` to span emitter's allowed supplemental keys
- **Tool definitions attribute** — New `gen_ai.tool.definitions` attribute for capturing tool/function schemas:
  - `GEN_AI_TOOL_DEFINITIONS` constant in attributes module
  - `tool_definitions` field on `LLMInvocation` dataclass with opt-in semconv_content metadata
  - `should_capture_tool_definitions()` helper in utils module for early gating (requires both message content and tool definitions capture enabled)
  - Environment variable `OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS` (default: `true`)

## Version 0.1.11

### Fixed
- **OTel context detachment errors in async instrumentation** — Replaced `tracer.start_as_current_span()` with `tracer.start_span()` in `SpanEmitter` to prevent `ValueError: <Token> was created in a different Context` errors when spans are started and stopped in different `asyncio.Task`s. Removed all `context_token` / `cm.__exit__()` detach blocks.
- **Centralized span parenting in TelemetryHandler** — Added `_current_genai_span` ContextVar with `_inherit_parent_span` / `_push_current_span` / `_pop_current_span` helpers. Sync instrumentations (CrewAI, FastMCP) get automatic parent-child linking without setting `parent_span`. In sync contexts, spans are also attached to OTel context for downstream non-GenAI instrumentation compatibility; skipped in async contexts to avoid cross-task detach errors.

## Version 0.1.10

### Added
- **Error classification for interrupt/resume support** — New `ErrorClassification` enum (`REAL_ERROR`, `INTERRUPT`, `CANCELLATION`) on `Error` dataclass enables classification-aware span status. Interrupts and cancellations leave span status as `UNSET` (default) instead of setting `StatusCode.ERROR`.
- **Classification-aware step status** — `_error_step` now sets `gen_ai.step.status` to `"interrupted"`, `"cancelled"`, or `"failed"` based on error classification.
- **Entity-agnostic finish reason attributes** — `gen_ai.finish_reason` and `gen_ai.finish_reason_description` set on any span type when an error is classified.

## Version 0.1.9 - 2026-03-04

### Added
- **Conversation Context & Association Properties** — New APIs for conversation tracking and custom context propagation across GenAI operations:
  - `genai_context()` — Context manager for automatic context propagation
  - `set_genai_context()` / `get_genai_context()` / `clear_genai_context()` — Manual context management
  - `GenAIContext` dataclass holding `conversation_id` and `properties` dict
  - Added `conversation_id` and `association_properties` fields to `GenAI` base type
  - Association properties emitted on spans as `gen_ai.association.properties.<key>`
  - New environment variables:
    - `OTEL_INSTRUMENTATION_GENAI_CONTEXT_INCLUDE_IN_METRICS` (`all` or comma-separated keys)
    - `OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION` (disable context propagation, default: `true`)
  - Priority order: explicit invocation value > contextvars

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
