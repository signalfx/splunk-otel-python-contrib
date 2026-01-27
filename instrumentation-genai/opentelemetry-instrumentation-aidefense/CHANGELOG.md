# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-08

### Added

- **Gateway Mode**: Automatic capture of `X-Cisco-AI-Defense-Event-Id` header when LLM calls are proxied through AI Defense Gateway
  - Wraps httpx HTTP client (covers OpenAI, Azure OpenAI, Cohere, Mistral SDKs)
  - Wraps botocore HTTP client (covers AWS Bedrock)
  - Automatically detects AI Defense Gateway URLs via regex patterns
  - Adds `gen_ai.security.event_id` to the current span (e.g., LangChain spans)
  - Support for custom gateway URL patterns via `OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS` environment variable

- **Supported LLM Providers for Gateway Mode**:
  - OpenAI (`api.openai.com`)
  - Azure OpenAI (`*.openai.azure.com`)
  - AWS Bedrock (`bedrock-runtime.*.amazonaws.com`)
  - Google Vertex AI (`*aiplatform.googleapis.com`)
  - Cohere (`api.cohere.com`)
  - Mistral (`api.mistral.ai`)

- New example: `examples/gateway/multi_agent_travel_planner/` demonstrating Gateway Mode

### Changed

- Refactored `AIDefenseInstrumentor` to support both SDK Mode and Gateway Mode
- SDK Mode instrumentation now gracefully handles missing `cisco-aidefense-sdk` dependency
- Improved URL pattern matching with compiled regex for performance

## [0.1.0] - 2025-12-15

### Added

- Initial release with SDK Mode support
- Instrumentation for `ChatInspectionClient` methods:
  - `inspect_prompt`
  - `inspect_response`
  - `inspect_conversation`
- Instrumentation for `HttpInspectionClient` methods:
  - `inspect_request`
  - `inspect_response`
  - `inspect_request_from_http_library`
  - `inspect_response_from_http_library`
- Capture of `gen_ai.security.event_id` span attribute for security event correlation
- Example: `examples/multi_agent_travel_planner/` demonstrating SDK Mode with LangGraph
