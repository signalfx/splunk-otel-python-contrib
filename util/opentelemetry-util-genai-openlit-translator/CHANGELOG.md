# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.1 - 2026-01-27

### Changed
- Updated to use structured `input_messages`/`output_messages` fields for Workflow, AgentCreation, and AgentInvocation types
- Aligned with `opentelemetry-util-genai` v0.1.8 structured message format
- Legacy span attributes are now wrapped in `InputMessage`/`OutputMessage` for backward compatibility

### Fixed
- Fixed bug where `invoke_agent` operation returned `None` instead of the invocation object

## Version 0.1.0 - 2025-01-24

- Initial 0.1.0 release of splunk-otel-util-genai-translator-openlit