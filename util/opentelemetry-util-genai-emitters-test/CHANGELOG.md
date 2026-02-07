# Changelog

All notable changes to this repository are documented in this file.

## Version 0.2.1 - 2026-02-04

### Changed
- **eval_perf_test.py** - Improved performance test framework
  - Fixed wait logic to properly wait for all evaluations to complete
  - Added progress reporting during evaluation wait
  - Added debug output for troubleshooting
  - Improved handling of async evaluation completion

## Version 0.2.0 - 2026-01-27

### Added
- **Performance Test Framework** - `eval_perf_test.py` for benchmarking evaluator modes
  - Supports trace-based sampling with configurable sample rates
  - Synthetic test data with 6 categories (bias, toxicity, hallucination, etc.)
  - Real-time progress monitoring and throughput reporting
  - JSON export of test results

### Changed
- **TestEmitter** - Enhanced for performance testing
  - Added evaluation result capture
  - Added statistics tracking APIs
  - Added wait-for-completion helpers

## Version 0.1.0 - 2026-01-15

- Initial release of opentelemetry-util-genai-emitters-test
