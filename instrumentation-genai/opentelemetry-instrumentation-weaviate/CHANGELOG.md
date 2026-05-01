# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- **`is_content_enabled()` now uses `get_content_capturing_mode()`** — Returns `True` for any mode other than `NO_CONTENT`, aligning with the unified `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` mode values (`SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT`). Legacy `true`/`false` values continue to work.

## [0.1.0] - 2026-01-22

### Added
- Initial release of OpenTelemetry Weaviate instrumentation
- Support for Weaviate client versions 3.x and 4.x
- Automatic tracing of Weaviate operations
