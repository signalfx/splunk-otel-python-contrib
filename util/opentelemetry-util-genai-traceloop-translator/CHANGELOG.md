# Changelog

All notable changes to this repository are documented in this file.

## Version 0.2.0 - 2026-03-03

### Changed
- Upgraded `opentelemetry-semantic-conventions` dependency from `~= 0.52b1` to `>= 0.60b1` for compatibility with `splunk-otel-python >= 2.9.0`.

## Version 0.1.8 - 2026-02-06
- Resolved issue with missing Agent evaluations & skip already instrumented spans.
- Removed redundant calls which was causing duplicate metrics & evaluations.

## Version 0.1.7 - 2026-01-29
- Fixed TraceloopSpanProcessor to correctly read input/output messages from spans after structured message types refactor.

## Version 0.1.6 - 2026-01-24
- Fixed issue with missing Agent evaluations in Traceloop Translator.
- Enhanced span processing to correctly identify and evaluate Agent spans.
- Improved input/output message capture for better evaluation accuracy.

## Version 0.1.5 - 2025-11-07

- Initial 0.1.5 release of splunk-otel-util-genai-translator-traceloop