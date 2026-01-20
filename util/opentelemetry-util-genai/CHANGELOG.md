# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.7 - 2026-01-17

### Added
- **Concurrent Evaluation Support** - New environment variables for parallel evaluation processing:
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` - Enable concurrent evaluation mode with multiple worker threads
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` - Configure number of worker threads (default: 4)
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` - Set bounded queue size for backpressure (0 = unbounded)
- These variables enable significant throughput improvements for LLM-as-a-Judge evaluations

## Version 0.1.6 - 2026-01-13

- Added `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC` environment variable to support single evaluation metric.

## Version 0.1.5 - 2025-12-19

- Added `GEN_AI_SECURITY_EVENT_ID` semantic convention attribute for Cisco AI Defense integration
- Added `security_event_id` field to `LLMInvocation` dataclass with semconv metadata

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-util-genai
