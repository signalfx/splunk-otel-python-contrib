# Evaluation Performance Testing Framework

## Overview

Create a comprehensive performance testing framework for the GenAI evaluator system to validate concurrent evaluation capabilities, measure throughput and latency, and ensure proper telemetry emission under load.

## Goals

1. **Performance Benchmarking**: Validate evaluator performance with concurrent GenAI invocations using opentelemetry-util-genai telemetry handler
2. **Telemetry Validation**: Capture and analyze all emitted telemetry and evaluation results using a test emitter coming from a opentelemetry-util-genai-emitters-test. Should ensur that for each enqued to Evaluation Manager invocation there is an evaluation result reported back to the telemetry handler and emitter. Should count errors by type and successes, wait for all enqued telemetry to succeed and print a clear summary of the run. Should capture all of the telemetry to a json file for each test run.
3. **Configuration Testing**: Test different evaluator configurations (concurrent mode, worker counts, queue sizes)
4. **Evaluator LLM configration Support**: Enable testing using locally run (LM Studio) LLMs to minimize the cost or some LLMs by using DEEPEVAL configuration
5. **Test Data Set** create test data set which will be used to run test invocations. Try to make it not fully obsviously triggering one of the evaluation metrics, but be more subtle and realistic.
6. **Monitoring Evaluator** the Evaluation Manager runs the deepeval tests async for each of the invocation with the workers concurrency. We need to test lall of the the CONCURRENT_EVAL_BENCHMARK.md and CONCURRENT_EVAL_ARCHITECTURE.md, and ensure we test the max queue. Fix any existing methods for waiting for all evals in the Telemetry Handler or Evaluation Manager or add new ones to continuously monitor the evaluation progress, queue lenght, and evaluation results in the test emitter. Make the test output concise but show the progress and test summary. 

## Architecture

### Components

Get more details in the AGENTS.md about the Splunk Distro for OpenTelemetry architecture and configuration.

```text
┌─────────────────────────────────────────────────────────────┐
│                    eval_perf_test.py                         │
│  - Loads synthetic LLM invocations from test data           │
│  - Configures evaluator (workers, queue, etc.)              │
│  - Monitors performance metrics                              │
└──────────────┬──────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────────┐
│           opentelemetry-util-genai-handler                   │
│  - start_llm() / stop_llm()                                  │
│  - Triggers completion callbacks                             │
└──────────────┬──────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────────┐
│       opentelemetry-util-genai-evals (Manager)               │
│  - Queues invocations for evaluation                         │
│  - Distributes work to concurrent workers                    │
│  - Calls registered evaluators                               │
└──────────────┬──────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────────┐
│  opentelemetry-util-genai-evals-deepeval (Evaluator)         │
│  - Runs bias, toxicity, hallucination metrics                │
│  - Calls local or cloud LLM for evaluation                   │
└──────────────┬──────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────────┐
│     opentelemetry-util-genai-emitters-test                   │
│  - Captures ALL telemetry (spans, metrics, events)           │
│  - Captures ALL evaluation results                           │
│  - Provides API for test to query captured data              │
└─────────────────────────────────────────────────────────────┘
```

### Test Data Strategy

Create 120 synthetic LLM invocation samples organized into 6 categories (20 samples each):

1. **Neutral Baseline** (20 samples): Clean, unbiased, factual responses
2. **Subtle Bias** (20 samples): Mild preference or stereotyping
3. **Subtle Toxicity** (20 samples): Passive-aggressive or mildly dismissive
4. **Hallucination** (20 samples): Fabricated details mixed with facts
5. **Irrelevant** (20 samples): Off-topic responses
6. **Negative Sentiment** (20 samples): Pessimistic or discouraging tone

### Models available locally

```text
curl http://localhost:1234/api/v1/chat
-H "Content-Type: application/json"
-d '{
"model": "jina-embeddings-v4-text-retrieval",
"system_prompt": "You answer only in rhymes.",
"input": "What is your favorite color?"
}'

curl http://localhost:1234/api/v1/chat
-H "Content-Type: application/json"
-d '{
"model": "liquid/lfm2.5-1.2b",
"system_prompt": "You answer only in rhymes.",
"input": "What is your favorite color?"
}'

curl http://localhost:1234/api/v1/chat
-H "Content-Type: application/json"
-d '{
"model": "mistralai/ministral-3-14b-reasoning",
"system_prompt": "You answer only in rhymes.",
"input": "What is your favorite color?"
}'

curl http://127.0.0.1:1234/v1/embeddings
-H "Content-Type: application/json"
-d '{
"model": "text-embedding-nomic-embed-text-v1.5",
"input": "Some text to embed"
}'
```

### Implementation plan

#### Phase 1: Test Infrastructure (Completed)

1. **Test Emitter Package** (`util/opentelemetry-util-genai-emitters-test/`)
   - Created package structure with `pyproject.toml`
   - Implemented `TestEmitter` class that captures all telemetry events
   - Provides statistics tracking, JSON export, and wait-for-completion APIs
   - Registered as entry point for automatic discovery

2. **Test Data Set** (`test_data.py`)
   - Created 120 synthetic samples across 6 categories (20 each):
     - Neutral baseline - clean, factual responses
     - Subtle bias - mild stereotyping/preference
     - Subtle toxicity - passive-aggressive/dismissive
     - Hallucination - fabricated details mixed with facts
     - Irrelevant - off-topic responses
     - Negative sentiment - pessimistic tone
   - Each sample includes: input prompt, context, response, expected evaluations

3. **Queue Monitoring** (Added to `evals/manager.py`)
   - `get_queue_depth()` - current items in queue
   - `get_pending_count()` - unfinished tasks count
   - `get_status()` - comprehensive status dictionary
   - `wait_for_all_with_progress()` - wait with callback for progress monitoring

4. **Performance Test Script** (`eval_perf_test.py`)
   - CLI interface with configurable options
   - Submits invocations through TelemetryHandler
   - Monitors evaluation progress in real-time
   - Collects and reports metrics (throughput, completion rate)
   - Exports results to JSON

#### Phase 2: Usage

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Install packages in development mode
pip install -e ./util/opentelemetry-util-genai
pip install -e ./util/opentelemetry-util-genai-evals
pip install -e ./util/opentelemetry-util-genai-emitters-test
pip install -e ./util/opentelemetry-util-genai-evals-deepeval

# Configure local LLM (LM Studio)
export DEEPEVAL_LLM_BASE_URL=http://localhost:1234/v1
export DEEPEVAL_LLM_MODEL=liquid/lfm2.5-1.2b

# Run performance test (with trace-based sampling)
python -m opentelemetry.util.genai.emitters.eval_perf_test \
    --samples 120 \
    --concurrent \
    --workers 4 \
    --queue-size 100 \
    --timeout 300 \
    --min-invocations 1 \
    --max-invocations 5 \
    --sample-rate 2 \
    --output results.json \
    --verbose
```

##### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--samples N` | 20 | Number of samples to test |
| `--concurrent` | false | Enable concurrent evaluation mode |
| `--workers N` | 4 | Number of concurrent workers |
| `--queue-size N` | 100 | Evaluation queue size |
| `--timeout N` | 300 | Timeout in seconds |
| `--output FILE` | None | Output JSON file for results |
| `--verbose` | false | Enable verbose logging |
| `--category CAT` | all | Test specific category only |
| `--min-invocations N` | 1 | Minimum invocations per trace |
| `--max-invocations N` | 5 | Maximum invocations per trace |
| `--sample-rate N` | 2 | Evaluate every Nth trace (2=50%, 4=25%) |
| `--evaluators STR` | all | Evaluator config string |

##### Trace-Based Sampling

The test framework groups invocations into traces with random counts (1-5 by default).
Sampling happens at the trace level:
- `--sample-rate 2` means every 2nd trace is evaluated (50%)
- `--sample-rate 4` means every 4th trace is evaluated (25%)
- All invocations within a sampled trace are evaluated together

This allows testing of realistic production scenarios where evaluation sampling
is trace-based, ensuring all operations within a trace are evaluated consistently.

##### Default Evaluators

By default, all available DeepEval metrics are enabled:
- **bias**: Detects biased content
- **toxicity**: Detects harmful/toxic content  
- **hallucination**: Detects fabricated information
- **answer_relevancy**: Checks response relevance
- **faithfulness**: Checks faithfulness to source context

Use `--evaluators` to specify a subset:
```bash
# Run only bias and toxicity evaluations
--evaluators "deepeval(LLMInvocation(bias,toxicity))"
```

##### Troubleshooting: Entry Point Registration Issues

If you encounter an error like:
```
AttributeError: module 'opentelemetry.util.genai.emitters.test' has no attribute 'test_emitters'
```

This indicates stale entry points from a previous package installation. Fix by:

```bash
# Ensure you're in the correct virtual environment
source .venv/bin/activate
which python  # Should show .venv/bin/python, NOT pyenv shims

# Uninstall any old package versions (both possible names)
pip uninstall -y splunk-otel-genai-emitters-test opentelemetry-util-genai-emitters-test

# Reinstall to refresh entry points
pip install -e ./util/opentelemetry-util-genai-emitters-test

# Verify entry points are correct
python -c "from importlib.metadata import entry_points; eps = entry_points(group='opentelemetry_util_genai_emitters'); print([f'{e.name}: {e.value}' for e in eps])"
# Should show: test: opentelemetry.util.genai.emitters.test:load_emitters
```

**Common Causes:**
- Virtual environment not properly activated (pyenv intercepting pip/python)
- Old package with different name still installed
- Entry points cached from previous installation

**Verification:**
```bash
# Check which Python/pip you're using
which python  # Should be .venv/bin/python
which pip     # Should be .venv/bin/pip

# List installed packages
pip list | grep -i splunk-otel-genai-emitters-test
```

#### Phase 3: Metrics Collected

- **Timing**: Submit time, evaluation time, total time
- **Throughput**: Invocations/sec, evaluations/sec  
- **Completion**: Total invocations, evaluations received, pending count
- **Errors**: By type (queue_full, evaluator_error, etc.)
- **Results by Metric**: Counts for bias, toxicity, hallucination, etc.

### Performance Benchmark Results

The following benchmarks compare different evaluator modes using the NativeEvaluator implementation.

> **Note:** The `deepeval(...)` evaluator plugin defaults to using `NativeEvaluator` (not the Deepeval library).
> To use the actual Deepeval library, set `OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION=deepeval`.

#### Test Configuration

| Parameter | Value |
|-----------|-------|
| Samples | 30 |
| Workers | 4 |
| Metrics | bias, toxicity, hallucination, answer_relevancy, sentiment |
| Metrics per sample | 10 (5 metrics × 2 invocations per trace) |
| Total evaluations | 300 |
| Sample rate | 1 (100% evaluation) |
| Local LLM | `liquid/lfm2.5-1.2b` via LM Studio at `http://localhost:1234/v1` |

#### Results Summary

| Mode | Implementation | Total Time | Throughput | LLM Calls per Invocation |
|------|---------------|------------|------------|--------------------------|
| **Native Batched** (default) | NativeEvaluator | 6.51s | **15.35 evals/s** | 1 call (all metrics) |
| **Native Non-Batched** | NativeEvaluator | 14.52s | **6.89 evals/s** | 5 calls (one per metric) |
| **Deepeval Library** | DeepevalEvaluator | 44.60s | **2.24 evals/s** | ~12-15 calls (2-3 per metric) |

#### Key Findings

1. **Native Batched is ~7x faster** than the Deepeval library
2. **Native Non-Batched is ~3x faster** than the Deepeval library
3. **Native Batched is ~2.2x faster** than Native Non-Batched mode
4. Both native modes use the same `NativeEvaluator` implementation
5. The performance difference between implementations is due to LLM call count

#### Why Native is Faster than Deepeval Library

| Implementation | LLM Calls per Metric | With 5 metrics | With 10 invocations (100 evaluations) |
|---------------|---------------------|----------------|---------------------------------------|
| **Native Batched** | 0.2 calls (all metrics in one prompt) | 1 call | **10 LLM calls total** |
| **Native Non-Batched** | 1 call per metric | 5 calls | **50 LLM calls total** |
| **Deepeval Library** | 2-3 calls per metric | 12-15 calls | **120-150 LLM calls total** |

The Deepeval library makes multiple LLM calls per metric because each metric implementation performs:
1. **Extraction step**: Extract relevant statements/claims from the response
2. **Evaluation step**: Evaluate the extracted content against criteria
3. **Reasoning step** (optional): Generate explanation for the score

Testing with BiasMetric + ToxicityMetric (2 metrics) showed **5 LLM calls** (~2.5 calls/metric on average).

This explains why the Deepeval library is ~3x slower than Native Non-Batched (which also uses 1 call per metric)
and ~7x slower than Native Batched (which uses 1 call for all metrics combined).

#### Detailed Results

##### Batched Mode (Default)

```
Environment:
  OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
  OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
  OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE=batched (default)

Results:
  Total invocations: 10
  Evaluations completed: 100
  Completion rate: 100%
  Evaluation time: 6.51s
  Throughput: 15.35 evals/s
```

##### Non-Batched Mode

```
Environment:
  OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
  OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
  OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE=non-batched

Results:
  Total invocations: 10
  Evaluations completed: 100
  Completion rate: 100%
  Evaluation time: 14.52s
  Throughput: 6.89 evals/s
```

##### Deepeval Library Mode

```
Environment:
  OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
  OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
  OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION=deepeval

Results:
  Total invocations: 10
  Evaluations completed: 100
  Completion rate: 100%
  Evaluation time: 44.60s
  Throughput: 2.24 evals/s
```

**Note:** The Deepeval library mode is significantly slower due to:
- Each metric makes 2-3 LLM calls (extraction + evaluation + reasoning)
- Total of ~12-15 LLM calls per invocation (for 5 metrics) vs 1 call for native batched

#### Configuration Reference

| Environment Variable | Values | Description |
|---------------------|--------|-------------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE` | `batched` (default), `non-batched` | Controls whether metrics are evaluated in one LLM call or separately |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_IMPLEMENTATION` | `deepeval` (default), `native` | Which evaluator implementation to use (deepeval = Deepeval library, native = NativeEvaluator) |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` | `true`, `false` | Enable/disable concurrent evaluation |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` | integer (default: 4) | Number of concurrent evaluation workers |

#### Recommendations

- **Use Native Batched mode (default)** for production - provides 7x better throughput than Deepeval library
- **Use Native Non-Batched** when per-metric timing is needed or for debugging (still 3x faster than Deepeval)
- **Use Deepeval Library** only if you need access to Deepeval-specific features or advanced metrics
- Increase worker count for high-throughput scenarios
- Native implementation is recommended for cost-sensitive production environments

#### Phase 4: Test Scenarios

1. **Baseline Sequential Test**
   ```bash
   OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=false
   ```

2. **Concurrent Mode Test**
   ```bash
   OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
   OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
   ```

3. **Queue Pressure Test**
   ```bash
   OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=50
   # Submit 100+ samples to test backpressure
   ```

4. **High Concurrency Test**
   ```bash
   OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=8
   ```