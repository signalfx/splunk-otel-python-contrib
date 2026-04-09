# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

### Added
- Initial release of Strands Agents instrumentation
- Support for Agent lifecycle telemetry via wrapt wrappers
- Support for LLM call telemetry via Strands hooks
- Support for tool call telemetry via Strands hooks
- Configurable built-in tracer suppression via `OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER`
- Optional BedrockAgentCoreApp entrypoint instrumentation for workflow-level spans
- **Bedrock AgentCore components instrumentation:**
  - MemoryClient operations (retrieve_memory_records, create_event, batch_create_memory_records, list_events)
  - CodeInterpreter operations (start, stop, execute_code, install_packages, upload_file)
  - BrowserClient operations (start, stop, take_control, release_control, get_session)
