#!/usr/bin/env python3
"""
Example demonstrating different GenAI invocation types with OpenTelemetry.

Usage:
    python invocation_example.py llm        # Run LLM invocation example
    python invocation_example.py agent      # Run Agent invocation example
    python invocation_example.py workflow   # Run Workflow invocation example
    python invocation_example.py all        # Run all examples (default)
    python invocation_example.py eval       # Run evaluation test cases from JSON
    python invocation_example.py eval 10    # Run first 10 evaluation test cases

    # Export to OTLP endpoint instead of console:
    python invocation_example.py llm --exporter otlp
    python invocation_example.py eval 10 --exporter otlp

    # Set session context for telemetry correlation:
    python invocation_example.py llm --session-id my-session-123
    python invocation_example.py all --session-id conv-456 --exporter otlp

This example shows:
1. Simple LLM invocation with input/output messages
2. Agent invocation with steps and tool calls
3. Workflow orchestration with multiple agents
4. Evaluation replay from travel_agent_test_cases.json
"""

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Set environment variables before importing OTel modules
os.environ.setdefault(
    "OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=invocation_example"
)
os.environ.setdefault("OTEL_SERVICE_NAME", "demo-app-invocation-example")
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", "SPAN_AND_EVENT"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event"
)
# Enable evaluation monitoring for metrics
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING", "true")
# Use batched evaluator for faster evaluation
# os.environ.setdefault(
#     "OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE", "batched"
# )

from opentelemetry import _logs as logs
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# OTLP exporters (optional, for --exporter otlp)
try:
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

from opentelemetry.util.genai.handler import (
    get_telemetry_handler,
    session_context,
)
from opentelemetry.util.genai.types import (
    AgentInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Step,
    Text,
    ToolCall,
    ToolCallResponse,
    Workflow,
)


def setup_telemetry(exporter: str = "console"):
    """Set up OpenTelemetry tracing, metrics, and logging.

    Args:
        exporter: Either "console" or "otlp". Default is "console".
    """
    use_otlp = exporter == "otlp"

    if use_otlp and not OTLP_AVAILABLE:
        print("Warning: OTLP exporters not available. Install with:")
        print("  pip install opentelemetry-exporter-otlp-proto-grpc")
        print("Falling back to console exporters.")
        use_otlp = False

    if use_otlp:
        print(
            f"Using OTLP exporters (endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')})"
        )
    else:
        print("Using console exporters")

    # Tracing
    trace_provider = TracerProvider()
    if use_otlp:
        trace_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter())
        )
    else:
        trace_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
    trace.set_tracer_provider(trace_provider)

    # Metrics
    if use_otlp:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(), export_interval_millis=5000
        )
    else:
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=5000
        )
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Logging (for events)
    logger_provider = LoggerProvider()
    if use_otlp:
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(OTLPLogExporter())
        )
    else:
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(ConsoleLogExporter())
        )
    logs.set_logger_provider(logger_provider)

    return trace_provider, meter_provider, logger_provider


def run_llm_invocation():
    """
    Demonstrate a simple LLM invocation.

    This creates a single LLM call with input messages and simulated output.
    """
    print("\n" + "=" * 80)
    print("LLM INVOCATION EXAMPLE")
    print("=" * 80 + "\n")

    handler = get_telemetry_handler()

    # Create an LLM invocation
    print("Starting LLM invocation...")
    llm = LLMInvocation(
        request_model="gpt-4o-mini",
        provider="openai",
        framework="custom",
        input_messages=[
            InputMessage(
                role="system",
                parts=[Text(content="You are a helpful assistant.")],
            ),
            InputMessage(
                role="user",
                parts=[Text(content="What is the capital of France?")],
            ),
        ],
    )

    handler.start_llm(llm)
    time.sleep(0.1)  # Simulate API call latency

    # Simulate LLM response
    llm.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[
                Text(
                    content="The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center."
                )
            ],
            finish_reason="stop",
        )
    ]
    llm.input_tokens = 25
    llm.output_tokens = 35

    handler.stop_llm(llm)

    print("LLM invocation completed!")
    print("  Model: gpt-4o-mini")
    print("  Input tokens: 25")
    print("  Output tokens: 35")
    print("\n")


def run_agent_invocation():
    """
    Demonstrate an agent invocation with steps and tool calls.

    This creates an agent that:
    1. Receives a user query
    2. Executes a step to process the query
    3. Makes an LLM call with a tool call
    4. Returns the result
    """
    print("\n" + "=" * 80)
    print("AGENT INVOCATION EXAMPLE")
    print("=" * 80 + "\n")

    handler = get_telemetry_handler()

    # Create agent invocation
    print("Starting agent: weather_assistant...")
    agent = AgentInvocation(
        name="weather_assistant",
        agent_type="assistant",
        framework="custom",
        model="gpt-4o-mini",
        input_messages=[
            InputMessage(
                role="user",
                parts=[
                    Text(content="User wants to know the weather in New York")
                ],
            ),
        ],
    )

    handler.start_agent(agent)
    time.sleep(0.05)

    # Step: Process user query
    print("  Executing step: process_query...")
    step = Step(
        name="process_query",
        step_type="planning",
        objective="Determine what weather information the user needs",
        source="agent",
        status="in_progress",
    )
    handler.start_step(step)
    time.sleep(0.05)

    # LLM call within the agent/step context
    print("  Making LLM call with tool use...")
    llm = LLMInvocation(
        request_model="gpt-4o-mini",
        provider="openai",
        framework="custom",
        agent_name="weather_assistant",
        agent_id=str(agent.run_id),
        input_messages=[
            InputMessage(
                role="system",
                parts=[
                    Text(
                        content="You are a weather assistant. Use the get_weather tool to fetch weather data."
                    )
                ],
            ),
            InputMessage(
                role="user",
                parts=[Text(content="What's the weather like in New York?")],
            ),
        ],
    )

    handler.start_llm(llm)
    time.sleep(0.1)

    # Simulate LLM response with tool call
    llm.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[
                ToolCall(
                    id="call_weather_001",
                    name="get_weather",
                    arguments={"location": "New York", "units": "fahrenheit"},
                )
            ],
            finish_reason="tool_calls",
        )
    ]
    llm.input_tokens = 45
    llm.output_tokens = 22

    handler.stop_llm(llm)

    # Second LLM call with tool response
    print("  Processing tool response...")
    llm2 = LLMInvocation(
        request_model="gpt-4o-mini",
        provider="openai",
        framework="custom",
        agent_name="weather_assistant",
        agent_id=str(agent.run_id),
        input_messages=[
            InputMessage(
                role="system",
                parts=[
                    Text(
                        content="You are a weather assistant. Use the get_weather tool to fetch weather data."
                    )
                ],
            ),
            InputMessage(
                role="user",
                parts=[Text(content="What's the weather like in New York?")],
            ),
            InputMessage(
                role="assistant",
                parts=[
                    ToolCall(
                        id="call_weather_001",
                        name="get_weather",
                        arguments={
                            "location": "New York",
                            "units": "fahrenheit",
                        },
                    )
                ],
            ),
            InputMessage(
                role="tool",
                parts=[
                    ToolCallResponse(
                        id="call_weather_001",
                        response='{"temperature": 72, "conditions": "sunny", "humidity": 45}',
                    )
                ],
            ),
        ],
    )

    handler.start_llm(llm2)
    time.sleep(0.1)

    llm2.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[
                Text(
                    content="The weather in New York is currently sunny with a temperature of 72°F and 45% humidity. It's a beautiful day!"
                )
            ],
            finish_reason="stop",
        )
    ]
    llm2.input_tokens = 85
    llm2.output_tokens = 32

    handler.stop_llm(llm2)

    # Complete step
    step.output_data = "Weather information retrieved and formatted"
    step.status = "completed"
    handler.stop_step(step)

    # Complete agent
    agent.output_result = (
        "The weather in New York is sunny, 72°F with 45% humidity."
    )
    handler.stop_agent(agent)

    print("Agent invocation completed!")
    print("  Agent: weather_assistant")
    print("  Steps executed: 1")
    print("  LLM calls: 2 (planning + response)")
    print("\n")


def run_workflow_invocation():
    """
    Demonstrate a workflow orchestrating multiple agents.

    This creates a workflow that:
    1. Routes user query to appropriate agent
    2. Executes classifier agent
    3. Executes handler agent based on classification
    """
    print("\n" + "=" * 80)
    print("WORKFLOW INVOCATION EXAMPLE")
    print("=" * 80 + "\n")

    handler = get_telemetry_handler()

    # Start workflow
    print("Starting workflow: customer_support_pipeline...")
    workflow = Workflow(
        name="customer_support_pipeline",
        workflow_type="sequential",
        description="Multi-agent customer support workflow",
        framework="custom",
        initial_input="User: I need help with my subscription",
    )

    handler.start_workflow(workflow)
    time.sleep(0.05)

    # Agent 1: Classifier
    print("  Agent: classifier_agent...")
    classifier = AgentInvocation(
        name="classifier_agent",
        agent_type="classifier",
        framework="custom",
        model="gpt-4o-mini",
        input_context="I need help with my subscription",
    )

    handler.start_agent(classifier)
    time.sleep(0.05)

    # LLM call for classification
    llm_classify = LLMInvocation(
        request_model="gpt-4o-mini",
        provider="openai",
        framework="custom",
        agent_name="classifier_agent",
        agent_id=str(classifier.run_id),
        input_messages=[
            InputMessage(
                role="system",
                parts=[
                    Text(
                        content="Classify the user query into: billing, technical, or general."
                    )
                ],
            ),
            InputMessage(
                role="user",
                parts=[Text(content="I need help with my subscription")],
            ),
        ],
    )

    handler.start_llm(llm_classify)
    time.sleep(0.05)

    llm_classify.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[Text(content="Category: billing")],
            finish_reason="stop",
        )
    ]
    llm_classify.input_tokens = 30
    llm_classify.output_tokens = 5

    handler.stop_llm(llm_classify)

    classifier.output_result = "billing"
    handler.stop_agent(classifier)

    # Agent 2: Billing Support
    print("  Agent: billing_support_agent...")
    billing_agent = AgentInvocation(
        name="billing_support_agent",
        agent_type="support",
        framework="custom",
        model="gpt-4o-mini",
        input_context="Handle billing query: I need help with my subscription",
    )

    handler.start_agent(billing_agent)
    time.sleep(0.05)

    # Step: Handle billing query
    billing_step = Step(
        name="handle_billing_query",
        step_type="execution",
        objective="Resolve the user's billing issue",
        source="agent",
        status="in_progress",
        input_data="I need help with my subscription",
    )
    handler.start_step(billing_step)
    time.sleep(0.05)

    # LLM call for billing support
    llm_billing = LLMInvocation(
        request_model="gpt-4o-mini",
        provider="openai",
        framework="custom",
        agent_name="billing_support_agent",
        agent_id=str(billing_agent.run_id),
        input_messages=[
            InputMessage(
                role="system",
                parts=[
                    Text(
                        content="You are a billing support specialist. Help users with subscription issues."
                    )
                ],
            ),
            InputMessage(
                role="user",
                parts=[Text(content="I need help with my subscription")],
            ),
        ],
    )

    handler.start_llm(llm_billing)
    time.sleep(0.1)

    llm_billing.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[
                Text(
                    content="I'd be happy to help with your subscription! I can see your account is active. What specific issue are you experiencing? Common topics include: cancellation, upgrades, payment methods, or billing history."
                )
            ],
            finish_reason="stop",
        )
    ]
    llm_billing.input_tokens = 40
    llm_billing.output_tokens = 48

    handler.stop_llm(llm_billing)

    billing_step.output_data = "Initial response sent to user"
    billing_step.status = "completed"
    handler.stop_step(billing_step)

    billing_agent.output_result = "User engaged, awaiting further input"
    handler.stop_agent(billing_agent)

    # Complete workflow
    workflow.final_output = (
        "Customer routed to billing support, initial response provided"
    )
    handler.stop_workflow(workflow)

    print("Workflow completed!")
    print("  Workflow: customer_support_pipeline")
    print("  Agents executed: 2 (classifier, billing_support)")
    print("  Total LLM calls: 2")
    print("\n")


# ---------------------------------------------------------------------------
# Evaluation Test Runner
# ---------------------------------------------------------------------------


def load_test_cases(limit: int | None = None) -> list[dict[str, Any]]:
    """Load test cases from the travel_agent_test_cases.json file."""
    test_file = Path(__file__).parent / "travel_agent_test_cases.json"
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    if limit:
        test_cases = test_cases[:limit]

    return test_cases


def create_evaluator() -> Any:
    """Create the batched evaluator for evaluation."""
    try:
        from opentelemetry.util.evaluator.deepeval_batched import (
            DeepevalBatchedEvaluator,
        )

        return DeepevalBatchedEvaluator(
            metrics=[
                "bias",
                "toxicity",
                "answer_relevancy",
                "hallucination",
                "sentiment",
            ],
            invocation_type="LLMInvocation",
        )
    except ImportError as e:
        print(f"Error: Could not import evaluator: {e}")
        print("Make sure opentelemetry-util-genai-evals-deepeval is installed")
        sys.exit(1)


def run_evaluation_tests(limit: int | None = None, verbose: bool = True):
    """
    Run evaluation tests from the travel_agent_test_cases.json file.

    This function:
    1. Loads test cases from the JSON file
    2. Creates LLM invocations for each test case
    3. Runs the batched evaluator on each invocation
    4. Collects and reports evaluation metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION TEST RUNNER")
    print("=" * 80 + "\n")

    # Load test cases
    test_cases = load_test_cases(limit)
    print(f"Loaded {len(test_cases)} test cases")

    # Create evaluator
    evaluator = create_evaluator()
    print(f"Using evaluator: {type(evaluator).__name__}")
    print(
        "Metrics: bias, toxicity, answer_relevancy, hallucination, sentiment"
    )
    print("-" * 80)

    handler = get_telemetry_handler()

    # Evaluation results summary
    results_summary: dict[str, dict[str, list[float]]] = {
        "good": {
            "bias": [],
            "toxicity": [],
            "answer_relevancy": [],
            "hallucination": [],
            "sentiment": [],
        },
        "bad_toxic": {
            "bias": [],
            "toxicity": [],
            "answer_relevancy": [],
            "hallucination": [],
            "sentiment": [],
        },
        "bad_biased": {
            "bias": [],
            "toxicity": [],
            "answer_relevancy": [],
            "hallucination": [],
            "sentiment": [],
        },
        "bad_hallucination": {
            "bias": [],
            "toxicity": [],
            "answer_relevancy": [],
            "hallucination": [],
            "sentiment": [],
        },
        "bad_irrelevant": {
            "bias": [],
            "toxicity": [],
            "answer_relevancy": [],
            "hallucination": [],
            "sentiment": [],
        },
        "bad_negative": {
            "bias": [],
            "toxicity": [],
            "answer_relevancy": [],
            "hallucination": [],
            "sentiment": [],
        },
    }

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case.get("id", i)
        category = test_case.get("category", "unknown")
        agent_name = test_case.get("agent", "unknown_agent")
        user_input = test_case.get("input", "")
        assistant_output = test_case.get("output", "")
        expected_quality = test_case.get("expected_quality", "good")

        if verbose:
            print(f"\n[{i}/{len(test_cases)}] Test ID: {test_id}")
            print(f"  Category: {category}")
            print(f"  Agent: {agent_name}")
            print(f"  Expected: {expected_quality}")
            print(
                f"  Input: {user_input[:60]}..."
                if len(user_input) > 60
                else f"  Input: {user_input}"
            )
            print(
                f"  Output: {assistant_output[:60]}..."
                if len(assistant_output) > 60
                else f"  Output: {assistant_output}"
            )

        # Create LLM invocation for this test case
        llm = LLMInvocation(
            request_model="gpt-4o-mini",
            provider="openai",
            framework="travel_agent",
            agent_name=agent_name,
            input_messages=[
                InputMessage(
                    role="system",
                    parts=[
                        Text(
                            content=f"You are a {agent_name} for a travel assistance service."
                        )
                    ],
                ),
                InputMessage(
                    role="user",
                    parts=[Text(content=user_input)],
                ),
            ],
        )

        # Start the invocation
        handler.start_llm(llm)

        # Add the output
        llm.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=assistant_output)],
                finish_reason="stop",
            )
        ]
        llm.input_tokens = len(user_input.split()) * 2  # Rough estimate
        llm.output_tokens = len(assistant_output.split()) * 2

        # Stop the invocation (this triggers any emitters)
        handler.stop_llm(llm)

        # Run evaluation
        try:
            eval_results = evaluator.evaluate(llm)

            if verbose:
                print("  Evaluation Results:")

            for result in eval_results:
                metric_name = result.metric_name
                score = result.score or 0.0
                passed = result.attributes.get(
                    "gen_ai.evaluation.passed", False
                )

                if verbose:
                    status = "✓" if passed else "✗"
                    print(f"    {metric_name}: {score:.3f} [{status}]")

                # Collect results for summary
                if expected_quality in results_summary:
                    if metric_name in results_summary[expected_quality]:
                        results_summary[expected_quality][metric_name].append(
                            score
                        )

            # Emit evaluation results through the handler
            handler.evaluation_results(eval_results, llm)

        except Exception as e:
            print(f"  Evaluation error: {e}")
            continue

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for quality, metrics_data in results_summary.items():
        non_empty = any(len(v) > 0 for v in metrics_data.values())
        if not non_empty:
            continue

        print(f"\n{quality.upper()} responses:")
        for metric, scores in metrics_data.items():
            if scores:
                avg = sum(scores) / len(scores)
                min_s = min(scores)
                max_s = max(scores)
                print(
                    f"  {metric:20s}: avg={avg:.3f}, min={min_s:.3f}, max={max_s:.3f}, n={len(scores)}"
                )

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80 + "\n")


def print_usage():
    """Print usage information."""
    print(
        """
Usage: python invocation_example.py [TYPE] [OPTIONS]

Types:
    llm       Run LLM invocation example
    agent     Run Agent invocation example
    workflow  Run Workflow invocation example
    all       Run all examples (default)
    eval      Run evaluation test cases from travel_agent_test_cases.json

Options:
    --exporter console|otlp   Export telemetry to console (default) or OTLP endpoint

Options for 'eval':
    [limit]   Optional number of test cases to run (default: all 100)

Examples:
    python invocation_example.py llm
    python invocation_example.py llm --exporter otlp
    python invocation_example.py agent
    python invocation_example.py workflow
    python invocation_example.py all
    python invocation_example.py eval        # Run all 100 test cases
    python invocation_example.py eval 10     # Run first 10 test cases
    python invocation_example.py eval 10 --exporter otlp

Environment Variables:
    OPENAI_API_KEY                              Required for 'eval' mode
    OTEL_EXPORTER_OTLP_ENDPOINT                 OTLP endpoint (default: http://localhost:4317)
    OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING Enable eval metrics (default: true)
    OTEL_INSTRUMENTATION_GENAI_EVALS_DEEPEVAL_MODE  'batched' or 'deepeval'
"""
    )


def main():
    """Main entry point."""
    # Parse command line arguments
    args = sys.argv[1:]

    # Extract --exporter option
    exporter = "console"
    if "--exporter" in args:
        idx = args.index("--exporter")
        if idx + 1 < len(args):
            exporter = args[idx + 1].lower()
            args = args[:idx] + args[idx + 2 :]  # Remove --exporter and value
        else:
            print("Error: --exporter requires a value (console or otlp)")
            sys.exit(1)

    if exporter not in {"console", "otlp"}:
        print(
            f"Error: Invalid exporter '{exporter}'. Use 'console' or 'otlp'."
        )
        sys.exit(1)

    # Extract --session-id option
    session_id = None
    if "--session-id" in args:
        idx = args.index("--session-id")
        if idx + 1 < len(args):
            session_id = args[idx + 1]
            args = (
                args[:idx] + args[idx + 2 :]
            )  # Remove --session-id and value
        else:
            print("Error: --session-id requires a value")
            sys.exit(1)

    # Parse invocation type
    if len(args) > 0:
        invocation_type = args[0].lower()
    else:
        invocation_type = "all"

    # Parse optional limit for eval mode
    eval_limit = None
    if invocation_type == "eval" and len(args) > 1:
        try:
            eval_limit = int(args[1])
        except ValueError:
            print(f"Error: Invalid limit '{args[1]}' - must be a number")
            sys.exit(1)

    # Handle help
    if invocation_type in {"help", "-h", "--help"}:
        print_usage()
        sys.exit(0)

    # Validate argument
    valid_types = {"llm", "agent", "workflow", "all", "eval"}
    if invocation_type not in valid_types:
        print(f"Error: Invalid invocation type '{invocation_type}'")
        print_usage()
        sys.exit(1)

    # Check for API key if running eval mode
    if invocation_type == "eval":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: OPENAI_API_KEY environment variable is required for eval mode"
            )
            print("Set it with: export OPENAI_API_KEY='your-key'")
            sys.exit(1)

    # Set up telemetry
    setup_telemetry(exporter=exporter)

    # Print session info if provided
    if session_id:
        print(f"Using session context: session_id={session_id}")

    # Run the requested example(s) with optional session context
    ctx = (
        session_context(session_id=session_id) if session_id else nullcontext()
    )
    with ctx:
        if invocation_type == "llm":
            run_llm_invocation()
        elif invocation_type == "agent":
            run_agent_invocation()
        elif invocation_type == "workflow":
            run_workflow_invocation()
        elif invocation_type == "eval":
            run_evaluation_tests(limit=eval_limit)
        else:  # all
            run_llm_invocation()
            run_agent_invocation()
            run_workflow_invocation()

    # Wait for metrics to be exported
    print("Waiting for metrics export...")
    time.sleep(30)

    print("\n" + "=" * 80)
    print("Example completed! Check the console output above for:")
    print("  • Span hierarchy for each invocation type")
    print("  • Metrics (duration, token usage)")
    print("  • Events (if content capture enabled)")
    if invocation_type == "eval":
        print("  • Evaluation metrics (gen_ai.evaluation.*)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
