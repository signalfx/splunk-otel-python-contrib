# OpenTelemetry GenAI Test Emitter

This package provides a test emitter for OpenTelemetry GenAI utilities. It captures all telemetry (spans, metrics, events, evaluation results) in memory for testing, validation, and performance benchmarking.

## Features

- **Complete Telemetry Capture**: Captures all `on_start`, `on_end`, `on_error`, and `on_evaluation_results` calls
- **Thread-Safe**: All operations are thread-safe for concurrent testing
- **JSON Export**: Export captured telemetry to JSON files for analysis
- **Statistics**: Built-in counters for successes, errors, and result types
- **Evaluation Performance Testing**: CLI tool for validating evaluation metrics

## Installation

This package is for **development and testing only** and is not published to PyPI.

```bash
# Install in editable mode from the repository
pip install -e ./util/opentelemetry-util-genai-emitters-test
```

## Usage

### Using the Test Emitter

The emitter is automatically registered via entry points. Enable it by setting:

```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span,test"
```

Or use it directly in code:

```python
from opentelemetry.util.genai.emitters.test import TestEmitter, get_test_emitter

# Get the singleton instance
emitter = get_test_emitter()

# After running your GenAI operations...

# Get statistics
stats = emitter.get_stats()
print(f"Total invocations: {stats['total_starts']}")
print(f"Evaluation results: {stats['total_evaluation_results']}")

# Export to JSON
emitter.export_to_json("test_results.json")

# Reset for next test
emitter.reset()
```

### Evaluation Performance Testing

The package includes a CLI tool for testing evaluation framework performance and validating that evaluation metrics work correctly against known test samples.

#### Prerequisites

1. **Install the package with dependencies**:
   ```bash
   pip install -e ./util/opentelemetry-util-genai-emitters-test
   pip install -e ./util/opentelemetry-util-genai-evals-deepeval
   ```

2. **Configure DeepEval LLM** (required for running evaluations):
   ```bash
   # Option 1: Use OpenAI
   export OPENAI_API_KEY="your-openai-key"
   
   # Option 2: Use a local model (e.g., LM Studio, Ollama)
   export DEEPEVAL_LLM_BASE_URL=http://localhost:1234/v1
   export DEEPEVAL_LLM_MODEL=your-model-name
   ```

#### Running the Test

```bash
python -m opentelemetry.util.genai.emitters.eval_perf_test [options]
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--samples N` | Number of test samples to use | 20 |
| `--concurrent` | Enable concurrent evaluation mode | Disabled |
| `--workers N` | Number of concurrent workers | 4 |
| `--queue-size N` | Evaluation queue size | 100 |
| `--timeout N` | Maximum timeout in seconds | 300 |
| `--output FILE` | JSON file for results export | None |
| `--output-dir DIR` | Directory for output files | /var/tmp |
| `--verbose` | Enable verbose logging | Disabled |
| `--category CAT` | Test only a specific category | All |
| `--min-invocations N` | Minimum invocations per trace | 1 |
| `--max-invocations N` | Maximum invocations per trace | 5 |
| `--sample-rate N` | Evaluate every Nth trace | 2 (50%) |

#### Test Categories

The test suite includes 120 samples across 6 categories:

- **neutral**: Balanced, unbiased responses
- **subtle_bias**: Responses with subtle bias patterns
- **subtle_toxicity**: Responses with subtle toxic content
- **hallucination**: Responses with factual inaccuracies
- **irrelevant**: Off-topic or irrelevant responses
- **negative_sentiment**: Responses with negative tone

Each sample has expected metric values for validation.

#### Metrics Tested

The framework tests all DeepEval metrics:

- **Bias**: Detects biased content
- **Toxicity**: Detects toxic/harmful content
- **Hallucination (GEval)**: Detects factual inaccuracies
- **Answer Relevancy**: Measures response relevance
- **Sentiment (GEval)**: Analyzes response sentiment

#### Example Run

```bash
# Run with 120 samples in concurrent mode with 4 workers
python -m opentelemetry.util.genai.emitters.eval_perf_test \
    --samples 120 \
    --concurrent \
    --workers 4 \
    --sample-rate 2 \
    --output results.json
```

#### Example Output

```
======================================================================
EVALUATION PERFORMANCE TEST
======================================================================

📁 Using balanced sampling across all categories
📊 Sample count: 120

📊 Configuration:
   Concurrent Mode: True
   Workers: 4
   Queue Size: 100
   Invocations per trace: 1-5
   Trace sample rate: 1/2 (every 2nd trace)

🔗 Organizing 120 samples into traces...
   Total traces: 40
   Sampled traces: 20 (50%)
   Sampled invocations: 54

🚀 Submitting invocations across 40 traces...
✅ Submitted all invocations in 0.05s

⏳ Waiting for evaluations (idle timeout: 60s)...
   [   0.0s] Queue:  62 | Pending:  66 | Results:   0
   [  18.0s] Queue:  59 | Pending:  63 | Results:  30
   ...
✅ All evaluations completed in 298.53s

======================================================================
EVALUATION PERFORMANCE TEST RESULTS
======================================================================

📋 Configuration:
   Concurrent Mode: True
   Workers: 4
   Invocations per trace: 1-5
   Trace sample rate: 1/2

🔗 Trace Statistics:
   Total Traces: 40
   Sampled Traces: 20
   Sampled Invocations: 54

⏱️  Timing:
   Submit Time: 0.05s
   Evaluation Time: 298.53s
   Total Time: 298.58s

📊 Metrics:
   Total Invocations: 240
   Sampled Invocations: 54
   Evaluations Received: 660
   Throughput (evals/s): 2.21

✅ Validation Summary:
   Total Validations: 330
   Passed: 184 (55.8%)
   Failed: 146

📏 Score Deviation:
   Mean Absolute Error (MAE): 0.3545
   Root Mean Square Error (RMSE): 0.5189
   Close Enough (labels match): 212 (64.2%)

📊 By Metric:
   ✗ Bias: 24/66 passed (36%)
   ✗ Toxicity: 62/66 passed (94%)
   ✓ Sentiment [GEval]: 66/66 passed (100%)

📄 Results exported to: results.json
📄 Failures exported to: /var/tmp/failures.json
```

#### Output Files

1. **results.json**: Complete test results including configuration, timing, metrics, and validation summary
2. **failures.json**: Detailed failure information with:
   - Sample ID and category
   - Expected vs actual scores
   - Failure type (false_positive, false_negative, evaluation_error)
   - Input prompt and response for debugging

#### Idle Timeout

The test uses an idle timeout of 60 seconds. If there's no change in the evaluation queue state for 60 seconds, the test stops waiting early. This prevents hanging on stalled evaluations.

#### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` | Enable concurrent mode | false |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` | Worker count | 4 |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` | Queue size | 100 |
| `DEEPEVAL_LLM_BASE_URL` | Custom LLM endpoint | None |
| `DEEPEVAL_LLM_MODEL` | Model name for DeepEval | None |
| `DEEPEVAL_FILE_SYSTEM` | File system mode | READ_ONLY |

## License

Apache License 2.0
