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

### Fixed
- Support `filename` as a CodeInterpreter upload-file argument when setting AgentCore filename metadata.
- Preserve an empty retrieval query value when content capture is disabled instead of storing `None`.
- Suppress CodeInterpreter `clear_context` results even when content capture is enabled.
