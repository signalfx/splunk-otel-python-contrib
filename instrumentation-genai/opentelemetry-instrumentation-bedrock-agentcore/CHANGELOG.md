# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial release of Bedrock AgentCore instrumentation
- Support for BedrockAgentCoreApp.entrypoint workflow spans
- Support for MemoryClient operations (retrieve_memories, create_event, create_blob_event, list_events)
- Support for CodeInterpreter operations (start, stop, execute_code, install_packages, upload_file)
- Support for BrowserClient operations (start, stop, take_control, release_control, get_session)
- Added AgentCore instrumentation testing reference with configuration, wrapped SDK surface, telemetry relationship model, and attribute assertions.
- Support for MemorySessionManager operations with safe metadata capture and retrieval spans for long-term memory search.

### Fixed
- Support `filename` as a CodeInterpreter upload-file argument when setting AgentCore filename metadata.
- Preserve an empty retrieval query value when content capture is disabled instead of storing `None`.
- Suppress CodeInterpreter `clear_context` results even when content capture is enabled.
- Suppress Memory event results and Browser control-plane results to avoid capturing payloads or infrastructure configuration.
- Gate AgentCore entrypoint input/output messages behind content capture.
- Suppress CodeInterpreter create, Browser session/update-stream, and conversational Memory content where responses or arguments can include sensitive data.
- Suppress CodeInterpreter get/list control-plane results and add AgentCore package tests to CI.
- Suppress generic MemoryClient control-plane arguments/results and align Browser session/list response parsing with SDK response keys.
