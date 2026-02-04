# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.13 - 2026-02-04

### Added
- **NativeEvaluator Async Support** - Added `evaluate_async()` method for non-blocking evaluation
  - Enables concurrent evaluation when used with async evaluation pipeline
  - Returns same results as synchronous `evaluate()` method

### Changed
- **Performance Benchmarks Documented** - Comprehensive comparison of evaluator modes:
  - Native Batched: 15.35 evals/s (7x faster than Deepeval library)
  - Native Non-Batched: 6.89 evals/s (3x faster than Deepeval library)
  - Deepeval Library: 2.24 evals/s (slower due to 2-3 LLM calls per metric)
- **eval_perf_test.py** - Improved wait logic and progress reporting

## Version 0.1.12 - 2026-01-30

### Fixed
- **Hallucination Post-Processing** - Score inversion now correctly applied
  - Changed metric name to `"hallucination [geval]"` to match post-processing name check when GEval appends `[GEval]` suffix

- **Sentiment `passed` Value** - Now correctly reflects actual sentiment
  - Modified `_derive_passed` to check sentiment labels first: `Positive`/`Neutral` → `true`, `Negative` → `false`

- **Relevance Label** - Now shows "Relevant"/"Irrelevant" instead of "Pass"/"Fail"

- **Toxicity Label** - Standardized to "Non Toxic"

## Version 0.1.11 - 2026-01-27

- Release 0.1.11 of splunk-otel-genai-evals-deepeval

## Version 0.1.10 - 2026-01-26

### Changed
- **Hallucination GEval Metric** - Improved accuracy and industry-standard scoring
  - Score inversion: GEval outputs higher=better (1.0=no hallucination), now inverted to lower=better (0.0=no hallucination) to match DeepEval's HallucinationMetric convention
  - Enhanced criteria with explicit guidance to distinguish logical inference from fabrication
  - Updated evaluation steps with self-verification and conservative flagging to reduce false positives
  - New attribute `deepeval.hallucination.geval_score` preserves the original GEval score for debugging
  
- **Sentiment GEval Metric** - Clearer scoring scale and thresholds
  - Updated to use 0-1 scale: 0=negative, 0.5=neutral, 1=positive (instead of -1 to +1)
  - Clear threshold boundaries: 0.0-0.35=Negative, 0.35-0.65=Neutral, 0.65-1.0=Positive
  - Improved criteria and steps for better intensity and mixed sentiment handling
  - Backward-compatible `deepeval.sentiment.compound` attribute still available

## Version 0.1.9 - 2026-01-17

### Added
- **Async Evaluation Support** - Native async methods for concurrent evaluation processing
  - `evaluate_async()` - Async entry point supporting concurrent mode
  - `evaluate_llm_async()` - Async LLM invocation evaluation
  - `evaluate_agent_async()` - Async agent invocation evaluation
  - `supports_async` property returns `True` for concurrent processing
  
- **Async DeepEval Runner** - New `run_evaluation_async()` function in `deepeval_runner.py`
  - Executes DeepEval in thread pool to avoid blocking event loop
  - Automatically enables DeepEval's internal `AsyncConfig(run_async=True)` in concurrent mode
  - Configurable `max_concurrent=10` for parallel metric evaluation

### Changed
- `DeepevalEvaluator` now implements native async evaluation methods
- DeepEval's internal async mode controlled by `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` flag
- When concurrent mode is enabled:
  - Multiple workers process invocations in parallel
  - DeepEval runs metrics concurrently within each invocation
  - Significant throughput improvement for LLM-as-a-Judge evaluations

### Notes
- Requires `splunk-otel-util-genai-evals>=0.1.5` for async evaluation support
- DeepEval's `run_async=True` spawns internal threads; callers should add buffer wait time
  after queue completion for complete metric results

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