# Changelog

All notable changes to this repository are documented in this file.

- Added `OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE` environment variable to support evaluation queue size.
  Added queue size check and error attribute on span while eval for a span is dropped. 

## Version 0.1.6 - 2026-01-13

- Added `OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC` environment variable to support single evaluation metric.

## Version 0.1.5 - 2025-12-19

- Added `GEN_AI_SECURITY_EVENT_ID` semantic convention attribute for Cisco AI Defense integration
- Added `security_event_id` field to `LLMInvocation` dataclass with semconv metadata

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-util-genai
