# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.8 - 2025-12-22

### Added
- `DEEPEVAL_LLM_EXTRA_HEADERS` environment variable support for custom HTTP headers
  - Enables custom headers for API gateways (e.g., `system-code` for Azure OpenAI proxies)
  - Format: JSON string (e.g., `'{"system-code": "APP-123"}'`)
  - Note: LiteLLM does not natively support `extra_headers` via environment variables
- Warning logs for `DEEPEVAL_LLM_EXTRA_HEADERS` parsing failures
  - Logs at WARNING level without exposing sensitive header values
  - Provides guidance on expected JSON format

## Version 0.1.7 - 2025-12-15

### Added
- OAuth2 authentication support for custom LLM providers via LiteLLM
  - `DEEPEVAL_LLM_TOKEN_URL` - OAuth2 token endpoint
  - `DEEPEVAL_LLM_CLIENT_ID` - OAuth2 client ID
  - `DEEPEVAL_LLM_CLIENT_SECRET` - OAuth2 client secret
  - `DEEPEVAL_LLM_GRANT_TYPE` - OAuth2 grant type (default: `client_credentials`)
  - `DEEPEVAL_LLM_SCOPE` - OAuth2 scope (optional)
  - `DEEPEVAL_LLM_AUTH_METHOD` - Token auth method (`basic` or `post`)
- Static API key authentication via `DEEPEVAL_LLM_API_KEY`
- Custom LLM provider configuration
  - `DEEPEVAL_LLM_BASE_URL` - Custom LLM endpoint
  - `DEEPEVAL_LLM_MODEL` - Model name (default: `gpt-4o-mini`)
  - `DEEPEVAL_LLM_PROVIDER` - LiteLLM provider prefix (default: `openai`)
  - `DEEPEVAL_LLM_AUTH_HEADER` - Auth header name (default: `api-key`)
  - `DEEPEVAL_LLM_CLIENT_APP_NAME` - App key for providers that require it

## Version 0.1.6 - 2025-11-07

- Initial 0.1.6 release of splunk-otel-genai-evals-deepeval