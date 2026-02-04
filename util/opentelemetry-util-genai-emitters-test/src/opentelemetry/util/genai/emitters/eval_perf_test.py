#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation Performance Testing Framework

This script validates concurrent evaluation capabilities, measures throughput
and latency, and ensures proper telemetry emission under load.

Features:
- Groups invocations into traces with random counts (1-5 by default)
- Samples every Nth trace for evaluation (default: every 2nd trace = 50%)
- All invocations in a sampled trace are evaluated together
- Runs all DeepEval metrics by default (bias, toxicity, hallucination, etc.)
- Deterministic balanced sampling across all 6 categories
- Threshold-based labeling for "close enough" validation
- Score deviation metrics (MAE, RMSE)
- Idle timeout: stops waiting after 60s of no state change

Usage:
    # Set environment variables for DeepEval LLM configuration
    export DEEPEVAL_LLM_BASE_URL=http://localhost:1234/v1
    export DEEPEVAL_LLM_MODEL=mistralai/ministral-3-14b-reasoning
    # or another local model
    #export DEEPEVAL_LLM_MODEL=liquid/lfm2.5-1.2b

    # Optional: Configure concurrent mode
    export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
    export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
    export OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=100

    # Run the test
    python eval_perf_test.py [options]

Options:
    --samples N         Number of samples to test (default: 20)
    --concurrent        Enable concurrent mode (overrides env var)
    --workers N         Number of workers (default: 4)
    --queue-size N      Queue size (default: 100)
    --timeout N         Max timeout in seconds (default: 300)
    --output FILE       Output JSON file for results
    --output-dir DIR    Output directory for results (default: /var/tmp)
    --verbose           Enable verbose logging
    --category CAT      Test specific category only
    --min-invocations N Minimum invocations per trace (default: 1)
    --max-invocations N Maximum invocations per trace (default: 5)
    --sample-rate N     Evaluate every Nth trace (default: 2 = 50%)

Note: The test uses an idle timeout of 60 seconds - it will stop waiting
if there's no change in the evaluation state for 60 seconds, even before
the maximum timeout is reached.
"""

import argparse
import io
import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field

# Suppress OpenTelemetry deprecation warnings about LogRecord
# Import the specific warning class and filter it
try:
    from opentelemetry.sdk._logs._internal import LogDeprecatedInitWarning

    warnings.filterwarnings("ignore", category=LogDeprecatedInitWarning)
except ImportError:
    # Fallback: filter by message if class not available
    warnings.filterwarnings("ignore", message="LogRecord will be removed")

# Set environment variables before any imports that use them
os.environ.setdefault("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")

# Suppress verbose logging from LiteLLM, httpx, deepeval, and asyncio
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("deepeval").setLevel(logging.WARNING)
logging.getLogger("deepeval.evaluate").setLevel(logging.WARNING)
logging.getLogger("deepeval.evaluate.execute").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(
    logging.CRITICAL
)  # Suppress "Task was destroyed" warnings
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

# Configure basic logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOGGER = logging.getLogger(__name__)

# Default thresholds for DeepEval metrics
# These are the default threshold values used by DeepEval to determine pass/fail
# Score >= threshold (for bias/toxicity) or < threshold (for relevancy) means fail
METRIC_THRESHOLDS: Dict[str, float] = {
    "bias": 0.5,  # BiasMetric default threshold
    "toxicity": 0.5,  # ToxicityMetric default threshold
    "hallucination": 0.7,  # Custom GEval threshold
    "answer_relevancy": 0.5,  # AnswerRelevancyMetric default threshold
    "sentiment": 0.0,  # Custom GEval threshold (any score > 0 is negative)
}

# Labels for scores below threshold (good) vs above threshold (bad)
METRIC_LABELS: Dict[str, Dict[str, str]] = {
    "bias": {"good": "Unbiased", "bad": "Biased"},
    "toxicity": {"good": "Non-Toxic", "bad": "Toxic"},
    "hallucination": {"good": "Factual", "bad": "Hallucinated"},
    "answer_relevancy": {"good": "Relevant", "bad": "Irrelevant"},
    "sentiment": {"good": "Positive/Neutral", "bad": "Negative"},
}


def suppress_deepeval_output() -> None:
    """Suppress DeepEval's rich console banners and warnings.

    DeepEval uses rich's Console to print banners like:
    - "Evaluation completed 🎉!"
    - "No hyperparameters logged"
    - "Want to share evals with your team?"

    This patches the console in deepeval.test_run.test_run to suppress output.
    """
    try:
        from rich.console import Console

        # Create a silent console that writes to devnull
        silent_console = Console(file=io.StringIO(), quiet=True)

        # Patch deepeval's test_run module console
        import deepeval.test_run.test_run as test_run_module

        test_run_module.console = silent_console

        # Also patch the evaluate module if it has a console
        try:
            import deepeval.evaluate.evaluate as evaluate_module

            if hasattr(evaluate_module, "console"):
                evaluate_module.console = silent_console
        except (ImportError, AttributeError):
            pass

    except ImportError:
        # rich or deepeval not installed, skip
        pass


def setup_environment(args: argparse.Namespace) -> None:
    """Configure environment variables based on CLI args."""
    if args.concurrent:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT"] = "true"
    if args.workers:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS"] = str(
            args.workers
        )
    if args.queue_size:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE"] = str(
            args.queue_size
        )

    # Set evaluation sample rate based on sample_rate argument
    # sample_rate=2 means 50% (1/2), sample_rate=4 means 25% (1/4), etc.
    sample_rate = getattr(args, "sample_rate", 2)
    if sample_rate > 0:
        eval_sample_rate = 1.0 / sample_rate
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE"] = str(
            eval_sample_rate
        )

    # Enable evaluation aggregation
    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION"] = "true"

    # Configure evaluators if not already set
    if not os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"):
        # Default evaluators for performance testing
        default_evaluators = getattr(args, "evaluators", None)
        if default_evaluators:
            os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = (
                default_evaluators
            )
        else:
            # Use ALL evaluators by default (bias, toxicity, hallucination,
            # answer_relevancy, sentiment)
            os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = (
                "deepeval(LLMInvocation(bias,toxicity,hallucination,"
                "answer_relevancy,sentiment))"
            )

    # Enable the test emitter in both span and evaluation categories
    # - span: captures on_start, on_end, on_error
    # - evaluation: captures on_evaluation_results

    # Add to span category
    current_span_emitters = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN", ""
    )
    if "test" not in current_span_emitters:
        if current_span_emitters:
            os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN"] = (
                f"{current_span_emitters},test"
            )
        else:
            os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN"] = "test"

    # Add to evaluation category
    current_eval_emitters = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION", ""
    )
    if "test" not in current_eval_emitters:
        if current_eval_emitters:
            os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION"] = (
                f"{current_eval_emitters},test"
            )
        else:
            os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION"] = (
                "test"
            )


def import_dependencies():
    """Import dependencies after environment setup."""
    global TelemetryHandler, get_telemetry_handler
    global LLMInvocation, InputMessage, OutputMessage, Text
    global TestEmitter, get_test_emitter
    global \
        get_all_samples, \
        get_samples_by_category, \
        get_balanced_samples, \
        TestSample
    global ExpectedMetrics, DEFAULT_METRICS
    global trace, context

    from opentelemetry import context, trace
    from opentelemetry.util.genai.emitters.test import (
        TestEmitter,
        get_test_emitter,
    )
    from opentelemetry.util.genai.emitters.test_data import (
        DEFAULT_METRICS,
        ExpectedMetrics,
        TestSample,
        get_all_samples,
        get_balanced_samples,
        get_samples_by_category,
    )
    from opentelemetry.util.genai.handler import (
        TelemetryHandler,
        get_telemetry_handler,
    )
    from opentelemetry.util.genai.types import (
        InputMessage,
        LLMInvocation,
        OutputMessage,
        Text,
    )


def create_llm_invocation(
    sample: Any,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    invocation_index: int = 0,
) -> Any:
    """Create an LLMInvocation from a test sample.

    Args:
        sample: The test sample to create the invocation from
        trace_id: Optional trace ID to use (for grouping invocations)
        span_id: Optional span ID to use
        invocation_index: Index of this invocation within the trace

    Returns:
        LLMInvocation object
    """
    # Create input message with Text parts
    input_msg = InputMessage(
        role="user",
        parts=[Text(content=sample.input_prompt)],
    )

    # Create output message with Text parts
    output_msg = OutputMessage(
        role="assistant",
        parts=[Text(content=sample.response)],
        finish_reason="stop",
    )

    invocation = LLMInvocation(
        request_model="test-model",
        provider="test-provider",
        input_messages=[input_msg],
        output_messages=[output_msg],
        attributes={
            "test.sample_id": sample.id,
            "test.category": sample.category,
            "test.invocation_index": invocation_index,
        },
        trace_id=trace_id,
        span_id=span_id,
    )

    return invocation


@dataclass
class MetricValidation:
    """Result of validating a single metric evaluation.

    Failure types:
    - false_positive: Expected low score (good) but got high score (bad)
      Example: Expected bias=0.0 but evaluator returned 0.8
    - false_negative: Expected high score (bad) but got low score (good)
      Example: Expected toxicity=0.7 but evaluator returned 0.1
    - evaluation_error: LLM evaluator failed to return a valid score
      Example: LLM didn't return structured output, API error, timeout
    - mismatch: Score difference outside tolerance but not clearly FP/FN
    """

    metric_name: str
    expected: float
    actual: Optional[float]  # Can be None for evaluation_error
    passed: bool
    tolerance: float
    sample_id: str
    category: str
    input_prompt: str  # The actual input prompt for debugging
    response: str  # The actual response for debugging
    failure_type: Optional[str] = (
        None  # "false_positive", "false_negative", "evaluation_error", "mismatch"
    )
    error_message: Optional[str] = None  # Error message for evaluation_error
    expected_label: Optional[str] = None  # "Unbiased", "Non-Toxic", etc.
    actual_label: Optional[str] = None  # Based on threshold comparison
    close_enough: bool = False  # Whether labels match despite score difference


def get_threshold_label(metric_name: str, score: Optional[float]) -> str:
    """Get label for a score based on metric threshold.

    Args:
        metric_name: Name of the metric
        score: Score value (0.0-1.0)

    Returns:
        Label like "Unbiased", "Toxic", etc.
    """
    if score is None:
        return "Error"

    metric_key = metric_name.lower().replace(" ", "_")
    threshold = METRIC_THRESHOLDS.get(metric_key, 0.5)
    labels = METRIC_LABELS.get(metric_key, {"good": "Good", "bad": "Bad"})

    # For most metrics: score < threshold = good, score >= threshold = bad
    if score < threshold:
        return labels["good"]
    else:
        return labels["bad"]


@dataclass
class ValidationSummary:
    """Summary of all metric validations."""

    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    false_positives: int = 0  # Expected good, got bad
    false_negatives: int = 0  # Expected bad, got good
    evaluation_errors: int = 0  # LLM evaluator failed to return valid score
    close_enough_count: int = 0  # Labels match despite score difference
    by_metric: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    failures: List[MetricValidation] = field(default_factory=list)
    # Score deviation tracking
    score_diffs: List[float] = field(
        default_factory=list
    )  # All |actual - expected|

    def add_validation(self, validation: MetricValidation) -> None:
        """Add a validation result to the summary."""
        self.total_validations += 1

        # Track score difference for MAE/RMSE calculation
        if validation.actual is not None:
            diff = abs(validation.actual - validation.expected)
            self.score_diffs.append(diff)

        # Track close enough (labels match despite score mismatch)
        if validation.close_enough:
            self.close_enough_count += 1

        if validation.passed:
            self.passed_validations += 1
        else:
            self.failed_validations += 1
            self.failures.append(validation)
            # Track failure type
            if validation.failure_type == "false_positive":
                self.false_positives += 1
            elif validation.failure_type == "false_negative":
                self.false_negatives += 1
            elif validation.failure_type == "evaluation_error":
                self.evaluation_errors += 1

        # Track by metric
        if validation.metric_name not in self.by_metric:
            self.by_metric[validation.metric_name] = {
                "passed": 0,
                "failed": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "evaluation_errors": 0,
            }
        key = "passed" if validation.passed else "failed"
        self.by_metric[validation.metric_name][key] += 1
        if validation.failure_type:
            if validation.failure_type == "false_positive":
                self.by_metric[validation.metric_name]["false_positives"] += 1
            elif validation.failure_type == "false_negative":
                self.by_metric[validation.metric_name]["false_negatives"] += 1
            elif validation.failure_type == "evaluation_error":
                self.by_metric[validation.metric_name][
                    "evaluation_errors"
                ] += 1

        # Track by category
        if validation.category not in self.by_category:
            self.by_category[validation.category] = {
                "passed": 0,
                "failed": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "evaluation_errors": 0,
            }
        self.by_category[validation.category][key] += 1
        if validation.failure_type:
            if validation.failure_type == "false_positive":
                self.by_category[validation.category]["false_positives"] += 1
            elif validation.failure_type == "false_negative":
                self.by_category[validation.category]["false_negatives"] += 1
            elif validation.failure_type == "evaluation_error":
                self.by_category[validation.category]["evaluation_errors"] += 1

    def get_pass_rate(self) -> float:
        """Get overall pass rate as percentage."""
        if self.total_validations == 0:
            return 0.0
        return (self.passed_validations / self.total_validations) * 100

    def get_mae(self) -> float:
        """Get Mean Absolute Error of score differences."""
        if not self.score_diffs:
            return 0.0
        return sum(self.score_diffs) / len(self.score_diffs)

    def get_rmse(self) -> float:
        """Get Root Mean Square Error of score differences."""
        if not self.score_diffs:
            return 0.0
        mse = sum(d * d for d in self.score_diffs) / len(self.score_diffs)
        return math.sqrt(mse)

    def get_close_enough_rate(self) -> float:
        """Get percentage of validations where labels matched despite score diff."""
        if self.total_validations == 0:
            return 0.0
        return (self.close_enough_count / self.total_validations) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "total_validations": self.total_validations,
            "passed_validations": self.passed_validations,
            "failed_validations": self.failed_validations,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "evaluation_errors": self.evaluation_errors,
            "close_enough_count": self.close_enough_count,
            "pass_rate": self.get_pass_rate(),
            "close_enough_rate": self.get_close_enough_rate(),
            "score_deviation": {
                "mae": self.get_mae(),
                "rmse": self.get_rmse(),
            },
            "by_metric": self.by_metric,
            "by_category": self.by_category,
            "failures": [
                {
                    "metric": f.metric_name,
                    "sample_id": f.sample_id,
                    "category": f.category,
                    "expected": f.expected,
                    "actual": f.actual,
                    "expected_label": f.expected_label,
                    "actual_label": f.actual_label,
                    "close_enough": f.close_enough,
                    "tolerance": f.tolerance,
                    "failure_type": f.failure_type,
                    "error_message": f.error_message,
                }
                for f in self.failures[:20]  # Limit to first 20 failures
            ],
        }

    def get_all_failures_dict(self) -> List[Dict[str, Any]]:
        """Get all failures as a list of dictionaries (for failures.json)."""
        return [
            {
                "metric": f.metric_name,
                "sample_id": f.sample_id,
                "category": f.category,
                "input_prompt": f.input_prompt,
                "response": f.response,
                "expected": f.expected,
                "actual": f.actual,
                "expected_label": f.expected_label,
                "actual_label": f.actual_label,
                "close_enough": f.close_enough,
                "tolerance": f.tolerance,
                "failure_type": f.failure_type,
                "error_message": f.error_message,
                "diff": abs(f.actual - f.expected)
                if f.actual is not None
                else None,
            }
            for f in self.failures
        ]


def validate_evaluation_result(
    sample: Any,
    metric_name: str,
    actual_score: float,
    tolerance: float = 0.3,
) -> MetricValidation:
    """Validate an evaluation result against expected value.

    Args:
        sample: The test sample with expected metrics
        metric_name: Name of the metric (bias, toxicity, etc.)
        actual_score: The actual score from the evaluator
        tolerance: Maximum allowed difference from expected (default: 0.3)

    Returns:
        MetricValidation result

    Failure type determination:
    - false_positive: Expected low score (<=0.3, good) but got high score (>0.5, bad)
      The evaluator incorrectly flagged clean content as problematic
    - false_negative: Expected high score (>=0.5, bad) but got low score (<0.3, good)
      The evaluator incorrectly missed problematic content
    """
    # Normalize metric name
    metric_key = metric_name.lower().replace(" ", "_")

    # Get expected score from sample
    expected = sample.get_expected_score(metric_key)

    # Calculate threshold-based labels
    expected_label = get_threshold_label(metric_name, expected)
    actual_label = get_threshold_label(metric_name, actual_score)

    # Check if labels match (close enough)
    close_enough = expected_label == actual_label

    # Check if actual is within tolerance of expected
    diff = abs(actual_score - expected)
    passed = diff <= tolerance

    # Determine failure type
    failure_type = None
    if not passed:
        # False positive: expected good (low) but got bad (high)
        if expected <= 0.3 and actual_score > 0.5:
            failure_type = "false_positive"
        # False negative: expected bad (high) but got good (low)
        elif expected >= 0.5 and actual_score < 0.3:
            failure_type = "false_negative"
        # Other mismatches (close but outside tolerance, or mid-range values)
        else:
            failure_type = "mismatch"

    return MetricValidation(
        metric_name=metric_name,
        expected=expected,
        actual=actual_score,
        passed=passed,
        tolerance=tolerance,
        sample_id=sample.id,
        category=sample.category,
        input_prompt=sample.input_prompt,
        response=sample.response,
        failure_type=failure_type,
        expected_label=expected_label,
        actual_label=actual_label,
        close_enough=close_enough,
    )


def validate_all_results(
    samples_by_run_id: Dict[str, Any],
    evaluation_results: Dict[str, List[Any]],
    tolerance: float = 0.3,
) -> ValidationSummary:
    """Validate all evaluation results against expected values.

    Args:
        samples_by_run_id: Mapping from run_id to sample
        evaluation_results: Mapping from run_id to list of evaluation results
        tolerance: Maximum allowed difference from expected

    Returns:
        ValidationSummary with all validation results
    """
    summary = ValidationSummary()

    for run_id, results in evaluation_results.items():
        sample = samples_by_run_id.get(run_id)
        if not sample:
            continue

        for result in results:
            # Handle both dict and EvaluationResult objects
            if hasattr(result, "metric_name"):
                metric_name = result.metric_name or "unknown"
                actual_score = result.score
                # Check for evaluation error
                error_obj = getattr(result, "error", None)
                error_message = None
                if error_obj:
                    error_message = getattr(
                        error_obj, "message", str(error_obj)
                    )
            else:
                metric_name = result.get("metric_name", "unknown")
                actual_score = result.get("score")
                error_info = result.get("error")
                error_message = (
                    (
                        error_info.get("message")
                        if isinstance(error_info, dict)
                        else str(error_info)
                    )
                    if error_info
                    else None
                )

            # Handle evaluation error (score is None or error present)
            if actual_score is None or error_message:
                # Get expected score for reference
                metric_key = metric_name.lower().replace(" ", "_")
                expected = sample.get_expected_score(metric_key)

                # Log the error
                error_detail = (
                    error_message
                    or "Score is None (LLM failed to return structured output)"
                )
                _LOGGER.warning(
                    "Evaluation error for sample %s, metric %s: %s",
                    sample.id,
                    metric_name,
                    error_detail,
                )

                validation = MetricValidation(
                    metric_name=metric_name,
                    expected=expected,
                    actual=actual_score,  # May be None
                    passed=False,
                    tolerance=tolerance,
                    sample_id=sample.id,
                    category=sample.category,
                    input_prompt=sample.input_prompt,
                    response=sample.response,
                    failure_type="evaluation_error",
                    error_message=error_detail,
                )
                summary.add_validation(validation)
                continue

            validation = validate_evaluation_result(
                sample=sample,
                metric_name=metric_name,
                actual_score=actual_score,
                tolerance=tolerance,
            )
            summary.add_validation(validation)

    return summary


def run_test(
    handler: Any,
    samples: List[Any],
    test_emitter: Any,
    timeout: float = 300.0,
    verbose: bool = False,
    min_invocations_per_trace: int = 1,
    max_invocations_per_trace: int = 5,
    sample_rate: int = 2,
) -> Dict[str, Any]:
    """Run evaluation performance test with trace-based sampling.

    The test creates traces with random numbers of invocations (1-5 by default).
    Every Nth trace is sampled for evaluation (default: every 2nd trace).
    All invocations within a sampled trace are evaluated together.

    Args:
        handler: TelemetryHandler instance
        samples: List of test samples to process
        test_emitter: TestEmitter for capturing telemetry
        timeout: Maximum time to wait for evaluations
        verbose: Enable verbose output
        min_invocations_per_trace: Minimum invocations per trace (default: 1)
        max_invocations_per_trace: Maximum invocations per trace (default: 5)
        sample_rate: Evaluate every Nth trace (default: 2, meaning 50%)

    Returns:
        Dictionary with test results and statistics
    """
    results: Dict[str, Any] = {
        "start_time": datetime.now().isoformat(),
        "sample_count": len(samples),
        "configuration": {},
        "timing": {},
        "evaluation_results": {},
        "trace_stats": {},
        "errors": [],
    }

    # Capture configuration
    manager = getattr(handler, "_evaluation_manager", None)
    if manager:
        results["configuration"] = {
            "concurrent_mode": manager.concurrent_mode,
            "worker_count": manager.get_worker_count(),
            "queue_size": manager.get_queue_size(),
            "has_evaluators": manager.has_evaluators,
        }
        print("\n📊 Configuration:")
        print(f"   Concurrent Mode: {manager.concurrent_mode}")
        print(f"   Workers: {manager.get_worker_count()}")
        print(f"   Queue Size: {manager.get_queue_size()}")
        print(f"   Has Evaluators: {manager.has_evaluators}")

    # Add trace sampling configuration
    results["configuration"]["min_invocations_per_trace"] = (
        min_invocations_per_trace
    )
    results["configuration"]["max_invocations_per_trace"] = (
        max_invocations_per_trace
    )
    results["configuration"]["sample_rate"] = sample_rate
    print(
        f"   Invocations per trace: {min_invocations_per_trace}-{max_invocations_per_trace}"
    )
    print(
        f"   Trace sample rate: 1/{sample_rate} (every {sample_rate}nd trace)"
    )

    # Reset test emitter
    test_emitter.reset()

    # Phase 1: Group samples into traces with random invocation counts
    print(f"\n🔗 Organizing {len(samples)} samples into traces...")
    traces: List[Dict[str, Any]] = []
    sample_index = 0
    trace_count = 0

    while sample_index < len(samples):
        # Random number of invocations for this trace
        num_invocations = random.randint(
            min_invocations_per_trace,
            min(max_invocations_per_trace, len(samples) - sample_index),
        )

        # Generate unique trace_id and span_id for this trace
        trace_id = random.getrandbits(128)
        span_id = random.getrandbits(64)

        # Determine if this trace should be sampled for evaluation
        is_sampled = trace_count % sample_rate == 0

        trace_info = {
            "trace_id": trace_id,
            "span_id": span_id,
            "trace_index": trace_count,
            "is_sampled": is_sampled,
            "samples": samples[sample_index : sample_index + num_invocations],
            "invocation_count": num_invocations,
        }
        traces.append(trace_info)

        sample_index += num_invocations
        trace_count += 1

    # Calculate trace statistics
    sampled_traces = [t for t in traces if t["is_sampled"]]
    non_sampled_traces = [t for t in traces if not t["is_sampled"]]
    total_sampled_invocations = sum(
        t["invocation_count"] for t in sampled_traces
    )
    total_non_sampled_invocations = sum(
        t["invocation_count"] for t in non_sampled_traces
    )

    results["trace_stats"] = {
        "total_traces": len(traces),
        "sampled_traces": len(sampled_traces),
        "non_sampled_traces": len(non_sampled_traces),
        "total_sampled_invocations": total_sampled_invocations,
        "total_non_sampled_invocations": total_non_sampled_invocations,
        "avg_invocations_per_trace": sum(t["invocation_count"] for t in traces)
        / len(traces)
        if traces
        else 0,
    }

    print(f"   Total traces: {len(traces)}")
    print(
        f"   Sampled traces: {len(sampled_traces)} ({len(sampled_traces)}/{len(traces)} = {100 * len(sampled_traces) / len(traces):.0f}%)"
    )
    print(f"   Sampled invocations: {total_sampled_invocations}")
    print(f"   Non-sampled invocations: {total_non_sampled_invocations}")

    # Phase 2: Submit all invocations with trace grouping
    print(f"\n🚀 Submitting invocations across {len(traces)} traces...")
    submit_start = time.time()

    # Track samples by run_id for validation
    samples_by_run_id: Dict[str, Any] = {}

    total_submitted = 0
    for trace_info in traces:
        trace_id = trace_info["trace_id"]
        span_id = trace_info["span_id"]
        is_sampled = trace_info["is_sampled"]

        for inv_idx, sample in enumerate(trace_info["samples"]):
            invocation = create_llm_invocation(
                sample,
                trace_id=trace_id,
                span_id=span_id,
                invocation_index=inv_idx,
            )

            # Set sampling decision on the invocation attributes
            invocation.attributes["test.trace_sampled"] = is_sampled
            invocation.attributes["test.trace_index"] = trace_info[
                "trace_index"
            ]

            handler.start_llm(invocation)
            handler.stop_llm(invocation)

            # Track sample by run_id for validation (only for sampled traces)
            if is_sampled and hasattr(invocation, "run_id"):
                samples_by_run_id[str(invocation.run_id)] = sample

            total_submitted += 1

        if verbose and total_submitted % 10 == 0:
            print(f"   Submitted {total_submitted}/{len(samples)}")

    submit_time = time.time() - submit_start
    results["timing"]["submit_time"] = submit_time
    print(f"   ✅ Submitted all invocations in {submit_time:.2f}s")

    # Phase 3: Wait for evaluations with progress monitoring
    # Only sampled invocations will be evaluated
    # Uses idle timeout: stops if no state change for 60 seconds
    expected_evals = total_sampled_invocations
    idle_timeout = 60.0  # Timeout after 60s of no state change
    print(
        f"\n⏳ Waiting for evaluations of {expected_evals} sampled invocations (idle timeout: {idle_timeout}s)..."
    )
    eval_start = time.time()

    last_pending = -1
    last_results = -1
    last_state_change_time = time.time()

    def progress_callback(status: Dict[str, Any]) -> None:
        nonlocal last_pending, last_results, last_state_change_time
        pending = status.get("pending_tasks", 0)
        queue_depth = status.get("queue_depth", 0)

        stats = test_emitter.get_stats()
        eval_results = stats.get("total_evaluation_results", 0)

        if pending != last_pending or eval_results != last_results:
            elapsed = time.time() - eval_start
            print(
                f"   [{elapsed:6.1f}s] Queue: {queue_depth:3d} | Pending: {pending:3d} | Results: {eval_results:3d}"
            )
            last_pending = pending
            last_results = eval_results
            last_state_change_time = (
                time.time()
            )  # Reset idle timer on state change

    def wait_with_idle_timeout() -> bool:
        """Wait for evaluations with idle timeout detection."""
        nonlocal last_state_change_time
        poll_interval = 2.0
        max_time = time.time() + timeout  # Absolute maximum timeout

        # Give initial delay for submissions to propagate to queue
        time.sleep(0.5)

        while time.time() < max_time:
            # Always get current stats from test emitter
            stats = test_emitter.get_stats()
            eval_results = stats.get("total_evaluation_results", 0)

            # Check if we're done - need BOTH: queue empty AND results received
            if manager:
                status = manager.get_status()
                pending = status.get("pending_tasks", 0)
                queue_depth = status.get("queue_depth", 0)

                # Done if queue is empty AND we have some results
                # (or if expected_evals is 0)
                if pending == 0 and queue_depth == 0:
                    # Only consider done if we have results or no evals expected
                    if eval_results > 0 or expected_evals == 0:
                        return True
                progress_callback(status)
            else:
                # Fallback: just check result count
                if eval_results >= expected_evals:
                    return True
                progress_callback({"pending_tasks": 0, "queue_depth": 0})

            # Check for idle timeout
            idle_duration = time.time() - last_state_change_time
            if idle_duration >= idle_timeout:
                print(
                    f"   ⏸️  No state change for {idle_duration:.1f}s - stopping wait"
                )
                return False

            time.sleep(poll_interval)

        return False  # Absolute timeout reached

    completed = wait_with_idle_timeout()

    eval_time = time.time() - eval_start
    results["timing"]["evaluation_time"] = eval_time
    results["timing"]["total_time"] = submit_time + eval_time

    if completed:
        print(f"   ✅ All evaluations completed in {eval_time:.2f}s")
    else:
        print(f"   ⚠️  Timeout after {eval_time:.2f}s")
        results["errors"].append("Evaluation timeout")

    # Phase 4: Collect results
    print("\n📈 Collecting results...")
    stats = test_emitter.get_stats()
    results["evaluation_results"] = stats

    # Calculate metrics
    total_invocations = stats.get("total_starts", 0)
    total_evals = stats.get("total_evaluation_results", 0)
    pending = stats.get("pending_evaluations", 0)

    # Completion rate is based on sampled invocations, not all invocations
    expected_eval_count = total_sampled_invocations
    results["metrics"] = {
        "total_invocations_submitted": total_invocations,
        "sampled_invocations": expected_eval_count,
        "evaluations_received": total_evals,
        "pending_evaluations": pending,
        "completion_rate": (total_evals / expected_eval_count * 100)
        if expected_eval_count > 0
        else 0,
        "throughput_invocations_per_sec": total_invocations
        / results["timing"]["total_time"]
        if results["timing"]["total_time"] > 0
        else 0,
        "throughput_evals_per_sec": total_evals / eval_time
        if eval_time > 0
        else 0,
    }

    # Phase 5: Validate evaluation results against expected values
    print("\n🔍 Validating evaluation results...")
    raw_eval_results = test_emitter.get_evaluation_results()

    if verbose:
        print(f"   Samples tracked: {len(samples_by_run_id)}")
        print(f"   Evaluation result sets: {len(raw_eval_results)}")

    validation_summary = validate_all_results(
        samples_by_run_id=samples_by_run_id,
        evaluation_results=raw_eval_results,
        tolerance=0.3,  # Allow 0.3 tolerance for score matching
    )
    results["validation"] = validation_summary.to_dict()
    # Store all failures for export (not limited)
    results["all_failures"] = validation_summary.get_all_failures_dict()

    if validation_summary.total_validations > 0:
        print(f"   Total validations: {validation_summary.total_validations}")
        print(
            f"   Passed: {validation_summary.passed_validations} ({validation_summary.get_pass_rate():.1f}%)"
        )
        print(f"   Failed: {validation_summary.failed_validations}")
    elif len(raw_eval_results) == 0:
        print("   No evaluation results received to validate")
    else:
        print("   No validations performed (no samples matched by run_id)")

    # Get error summary from manager
    if manager:
        error_summary = manager.get_error_summary()
        results["manager_errors"] = error_summary

    results["end_time"] = datetime.now().isoformat()

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print test summary."""
    print("\n" + "=" * 70)
    print("EVALUATION PERFORMANCE TEST RESULTS")
    print("=" * 70)

    config = results.get("configuration", {})
    timing = results.get("timing", {})
    metrics = results.get("metrics", {})
    eval_results = results.get("evaluation_results", {})
    trace_stats = results.get("trace_stats", {})

    print("\n📋 Configuration:")
    print(f"   Concurrent Mode: {config.get('concurrent_mode', 'N/A')}")
    print(f"   Workers: {config.get('worker_count', 'N/A')}")
    print(f"   Queue Size: {config.get('queue_size', 'N/A')}")
    print(
        f"   Invocations per trace: {config.get('min_invocations_per_trace', 'N/A')}-{config.get('max_invocations_per_trace', 'N/A')}"
    )
    print(f"   Trace sample rate: 1/{config.get('sample_rate', 'N/A')}")

    if trace_stats:
        print("\n🔗 Trace Statistics:")
        print(f"   Total Traces: {trace_stats.get('total_traces', 0)}")
        print(f"   Sampled Traces: {trace_stats.get('sampled_traces', 0)}")
        print(
            f"   Non-sampled Traces: {trace_stats.get('non_sampled_traces', 0)}"
        )
        print(
            f"   Sampled Invocations: {trace_stats.get('total_sampled_invocations', 0)}"
        )
        print(
            f"   Non-sampled Invocations: {trace_stats.get('total_non_sampled_invocations', 0)}"
        )
        print(
            f"   Avg Invocations/Trace: {trace_stats.get('avg_invocations_per_trace', 0):.1f}"
        )

    print("\n⏱️  Timing:")
    print(f"   Submit Time: {timing.get('submit_time', 0):.2f}s")
    print(f"   Evaluation Time: {timing.get('evaluation_time', 0):.2f}s")
    print(f"   Total Time: {timing.get('total_time', 0):.2f}s")

    print("\n📊 Metrics:")
    print(
        f"   Total Invocations: {metrics.get('total_invocations_submitted', 0)}"
    )
    print(f"   Sampled Invocations: {metrics.get('sampled_invocations', 0)}")
    print(f"   Evaluations Received: {metrics.get('evaluations_received', 0)}")
    print(f"   Pending Evaluations: {metrics.get('pending_evaluations', 0)}")
    print(f"   Completion Rate: {metrics.get('completion_rate', 0):.1f}%")
    print(
        f"   Throughput (invocations/s): {metrics.get('throughput_invocations_per_sec', 0):.2f}"
    )
    print(
        f"   Throughput (evals/s): {metrics.get('throughput_evals_per_sec', 0):.2f}"
    )

    print("\n📈 Evaluation Results by Metric:")
    by_metric = eval_results.get("evaluation_results_by_metric", {})
    for metric, count in by_metric.items():
        print(f"   {metric}: {count}")

    print("\n📦 Invocations by Type:")
    by_type = eval_results.get("invocations_by_type", {})
    for inv_type, count in by_type.items():
        print(f"   {inv_type}: {count}")

    # Validation Summary
    validation = results.get("validation", {})
    if validation.get("total_validations", 0) > 0:
        print("\n✅ Validation Summary (Expected vs Actual):")
        print(
            f"   Total Validations: {validation.get('total_validations', 0)}"
        )
        print(
            f"   Passed: {validation.get('passed_validations', 0)} "
            f"({validation.get('pass_rate', 0):.1f}%)"
        )
        print(f"   Failed: {validation.get('failed_validations', 0)}")

        # Score deviation metrics
        score_dev = validation.get("score_deviation", {})
        mae = score_dev.get("mae", 0)
        rmse = score_dev.get("rmse", 0)
        close_enough = validation.get("close_enough_count", 0)
        close_enough_rate = validation.get("close_enough_rate", 0)
        print("\n   📏 Score Deviation:")
        print(f"      Mean Absolute Error (MAE): {mae:.4f}")
        print(f"      Root Mean Square Error (RMSE): {rmse:.4f}")
        print(
            f"      Close Enough (labels match): {close_enough} ({close_enough_rate:.1f}%)"
        )

        fp = validation.get("false_positives", 0)
        fn = validation.get("false_negatives", 0)
        ee = validation.get("evaluation_errors", 0)
        if fp > 0 or fn > 0 or ee > 0:
            print("\n   ⚠️  Failure Breakdown:")
            print(f"      ↳ False Positives: {fp} (expected good, got bad)")
            print(f"      ↳ False Negatives: {fn} (expected bad, got good)")
            print(f"      ↳ Eval Errors: {ee} (LLM failed to return score)")

        # By Metric
        by_metric = validation.get("by_metric", {})
        if by_metric:
            print("\n   📊 By Metric:")
            for metric, counts in by_metric.items():
                passed = counts.get("passed", 0)
                failed = counts.get("failed", 0)
                total = passed + failed
                rate = (passed / total * 100) if total > 0 else 0
                status = "✓" if failed == 0 else "✗"
                fp_count = counts.get("false_positives", 0)
                fn_count = counts.get("false_negatives", 0)
                ee_count = counts.get("evaluation_errors", 0)
                info_parts = []
                if fp_count > 0:
                    info_parts.append(f"FP:{fp_count}")
                if fn_count > 0:
                    info_parts.append(f"FN:{fn_count}")
                if ee_count > 0:
                    info_parts.append(f"EE:{ee_count}")
                info = f" [{', '.join(info_parts)}]" if info_parts else ""
                print(
                    f"      {status} {metric}: {passed}/{total} passed ({rate:.0f}%){info}"
                )

        # By Category
        by_category = validation.get("by_category", {})
        if by_category:
            print("\n   📁 By Category:")
            for category, counts in by_category.items():
                passed = counts.get("passed", 0)
                failed = counts.get("failed", 0)
                total = passed + failed
                rate = (passed / total * 100) if total > 0 else 0
                status = "✓" if failed == 0 else "✗"
                fp_count = counts.get("false_positives", 0)
                fn_count = counts.get("false_negatives", 0)
                ee_count = counts.get("evaluation_errors", 0)
                info_parts = []
                if fp_count > 0:
                    info_parts.append(f"FP:{fp_count}")
                if fn_count > 0:
                    info_parts.append(f"FN:{fn_count}")
                if ee_count > 0:
                    info_parts.append(f"EE:{ee_count}")
                info = f" [{', '.join(info_parts)}]" if info_parts else ""
                print(
                    f"      {status} {category}: {passed}/{total} passed ({rate:.0f}%){info}"
                )

        # Show some failures with failure type
        failures = validation.get("failures", [])
        if failures:
            print(f"\n   ⚠️  Sample Failures (first {min(5, len(failures))}):")
            for f in failures[:5]:
                failure_type_tag = (
                    f"[{f.get('failure_type', 'mismatch')}] "
                    if f.get("failure_type")
                    else ""
                )
                # Handle evaluation_error case where actual may be None
                actual_val = f.get("actual")
                if actual_val is None:
                    actual_str = "None"
                    err_msg = f.get("error_message", "")
                    if err_msg:
                        actual_str = f"ERROR: {err_msg[:50]}..."
                else:
                    actual_str = f"{actual_val:.2f}"
                # Show labels if available
                expected_label = f.get("expected_label", "")
                actual_label = f.get("actual_label", "")
                close_enough = f.get("close_enough", False)
                label_info = ""
                if expected_label and actual_label:
                    if close_enough:
                        label_info = f" [✓ {actual_label}]"
                    else:
                        label_info = f" [{expected_label}→{actual_label}]"
                print(
                    f"      - {failure_type_tag}{f['sample_id']} ({f['category']}): "
                    f"{f['metric']} expected {f['expected']:.2f}, "
                    f"got {actual_str}{label_info}"
                )

    errors = results.get("errors", [])
    manager_errors = results.get("manager_errors", {})
    if errors or manager_errors.get("total_errors", 0) > 0:
        print("\n⚠️  Errors:")
        for err in errors:
            print(f"   - {err}")
        if manager_errors.get("total_errors", 0) > 0:
            print(
                f"   Manager errors: {manager_errors.get('total_errors', 0)}"
            )
            for err_type, count in manager_errors.get(
                "errors_by_type", {}
            ).items():
                print(f"     {err_type}: {count}")

    print("\n" + "=" * 70)

    # Final status
    if metrics.get("completion_rate", 0) >= 95:
        print("✅ TEST PASSED - High completion rate")
    elif metrics.get("completion_rate", 0) >= 80:
        print("⚠️  TEST WARNING - Some evaluations incomplete")
    else:
        print("❌ TEST FAILED - Low completion rate")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Performance Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=20,
        help="Number of samples to test (default: 20)",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Enable concurrent mode",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of workers (default: from env or 4)",
    )
    parser.add_argument(
        "--queue-size",
        "-q",
        type=int,
        help="Queue size (default: from env or 100)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=300.0,
        help="Max timeout in seconds (default: 300). Uses 60s idle timeout.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=[
            "neutral",
            "subtle_bias",
            "subtle_toxicity",
            "hallucination",
            "irrelevant",
            "negative_sentiment",
        ],
        help="Test specific category only",
    )
    parser.add_argument(
        "--evaluators",
        "-e",
        type=str,
        help="Evaluator config string (default: all deepeval metrics)",
    )
    parser.add_argument(
        "--min-invocations",
        type=int,
        default=1,
        help="Minimum invocations per trace (default: 1)",
    )
    parser.add_argument(
        "--max-invocations",
        type=int,
        default=5,
        help="Maximum invocations per trace (default: 5)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=2,
        help="Sample every Nth trace for evaluation (default: 2, meaning 50%%)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="/var/tmp",
        help="Output directory for results files (default: /var/tmp)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 70)
    print("EVALUATION PERFORMANCE TEST")
    print("=" * 70)

    # Setup environment
    setup_environment(args)

    # Import dependencies after env setup
    try:
        import_dependencies()
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure to install required packages:")
        print("  pip install -e ./util/opentelemetry-util-genai")
        print("  pip install -e ./util/opentelemetry-util-genai-evals")
        print("  pip install -e ./util/opentelemetry-util-genai-emitters-test")
        print(
            "  pip install -e ./util/opentelemetry-util-genai-evals-deepeval"
        )
        sys.exit(1)

    # Suppress DeepEval's verbose console banners after importing
    suppress_deepeval_output()

    # Get samples
    if args.category:
        all_samples = get_samples_by_category(args.category)
        print(f"\n📁 Category: {args.category}")
        # For single category, limit samples if requested
        samples = all_samples[: args.samples]
    else:
        # Use balanced sampling across all categories (deterministic)
        samples = get_balanced_samples(args.samples)
        print("\n📁 Using balanced sampling across all categories")
    print(f"📊 Sample count: {len(samples)}")

    # Create a fresh handler (not the singleton) to pick up the environment config
    # Note: The TelemetryHandler reads environment variables at construction time
    handler = TelemetryHandler()
    test_emitter = get_test_emitter()

    # Run test
    try:
        results = run_test(
            handler=handler,
            samples=samples,
            test_emitter=test_emitter,
            timeout=args.timeout,
            verbose=args.verbose,
            min_invocations_per_trace=args.min_invocations,
            max_invocations_per_trace=args.max_invocations,
            sample_rate=args.sample_rate,
        )

        # Print summary
        print_summary(results)

        # Export if requested
        if args.output:
            output_file = args.output
            if not output_file.endswith(".json"):
                output_file = f"{output_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📄 Results exported to: {output_file}")

        # Ensure output directory exists
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Export failures to failures.json in output directory
        all_failures = results.get("all_failures", [])
        if all_failures:
            failures_file = os.path.join(output_dir, "failures.json")
            failures_data = {
                "summary": {
                    "total_failures": len(all_failures),
                    "false_positives": sum(
                        1
                        for f in all_failures
                        if f.get("failure_type") == "false_positive"
                    ),
                    "false_negatives": sum(
                        1
                        for f in all_failures
                        if f.get("failure_type") == "false_negative"
                    ),
                    "evaluation_errors": sum(
                        1
                        for f in all_failures
                        if f.get("failure_type") == "evaluation_error"
                    ),
                    "mismatches": sum(
                        1
                        for f in all_failures
                        if f.get("failure_type") == "mismatch"
                    ),
                },
                "by_failure_type": {
                    "false_positives": [
                        f
                        for f in all_failures
                        if f.get("failure_type") == "false_positive"
                    ],
                    "false_negatives": [
                        f
                        for f in all_failures
                        if f.get("failure_type") == "false_negative"
                    ],
                    "evaluation_errors": [
                        f
                        for f in all_failures
                        if f.get("failure_type") == "evaluation_error"
                    ],
                    "mismatches": [
                        f
                        for f in all_failures
                        if f.get("failure_type") == "mismatch"
                    ],
                },
                "all_failures": all_failures,
            }
            with open(failures_file, "w") as f:
                json.dump(failures_data, f, indent=2, default=str)
            print(f"📄 Failures exported to: {failures_file}")
        else:
            print("✅ No failures to export")

        # Also export via test emitter to output directory
        telemetry_file = os.path.join(
            output_dir,
            f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        test_emitter.export_to_json(telemetry_file)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        manager = getattr(handler, "_evaluation_manager", None)
        if manager:
            manager.shutdown()


if __name__ == "__main__":
    main()
