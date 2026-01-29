# Changelog

## [0.1.0] - 2025-01-29

### Added
- Initial release of Langsmith to GenAI semantic convention translator
- `LangsmithSpanProcessor` for translating Langsmith span attributes to GenAI semantic conventions
- Auto-enable functionality on package installation
- Support for attribute transformations:
  - Content & Messages (entity input/output to gen_ai.input/output.messages)
  - Model & System attributes (ls_provider, ls_model_name)
  - Request parameters (temperature, max_tokens, top_p, top_k)
  - Token usage metrics
  - Tool/function calling attributes
  - Session and run tracking
  - Agent and workflow attributes
- Message reconstruction from LangChain serialization formats
- Content normalization for various provider formats (OpenAI, Anthropic, Google)
- Support for conditional transformation rules
- Environment variable `OTEL_INSTRUMENTATION_GENAI_LANGSMITH_DISABLE` to disable auto-registration
