# Changelog

## [0.1.0] - 2026-01-30

### Added
- Initial release of `opentelemetry-util-auth` package
- `OAuth2TokenManager` for OAuth2 client credentials flow
- Support for two authentication methods:
  - Basic Authentication (default)
  - POST body authentication (Azure AD, enterprise IdPs)
- Automatic token refresh before expiry
- Thread-safe token caching
- Comprehensive logging support
- Environment variable configuration via `LLM_*` variables
- Helper method `get_llm_base_url()` for constructing LLM endpoints

