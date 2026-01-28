# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.6 - 2026-01-28

### Added

- **Rate limiter test example** - New `rate-limiter-test` example demonstrating evaluation rate limiting configuration and behavior

## Version 0.1.5 - 2026-01-27

### Changed

- **Use structured message fields** - Workflow and AgentInvocation now use `input_messages`/`output_messages` structured fields instead of legacy string fields for proper JSON serialization.
- **Improved JSON serialization** - `_serialize()` helper now uses `json.dumps(obj, default=str)` for reliable JSON output with non-serializable objects.
- **Removed input/output from Step** - Step spans no longer capture redundant input/output data.

### Fixed

- **Agent/workflow message rendering** - Fixed issue where agent and workflow `gen_ai.input.messages` were rendered as Python repr strings instead of valid JSON.

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-instrumentation-langchain
