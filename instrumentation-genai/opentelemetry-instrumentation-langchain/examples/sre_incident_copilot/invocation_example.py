"""Invocation example — demonstrates interrupt/resume with manual instrumentation.

This script runs the SRE Incident Copilot with interrupt/resume in a single
process, producing two distinct telemetry traces under the same
conversation_id (gen_ai.conversation.id):

  Trace 1  triage → action_planner → ⏸️  interrupt
  Trace 2  human_review → quality_gate → ✅  done

OpenTelemetry is configured programmatically (no zero-code agent required).
After both phases complete the script waits for ``--wait-after-completion``
seconds so LLM-as-a-judge evaluations can finish exporting.

Usage:
    # Auto-approve (default) — wait 300 s for evals
    python invocation_example.py --scenario scenario-001 \
        --wait-after-completion 300

    # Auto-reject with feedback
    python invocation_example.py --scenario scenario-001 --auto-reject \
        --feedback "add rollback step for database connections"

    # Interactive — prompts for approval
    python invocation_example.py --scenario scenario-001

Environment variables:
    OPENAI_API_KEY          Required — OpenAI API key
    OTEL_EXPORTER_OTLP_ENDPOINT  Optional — OTLP endpoint (default: http://localhost:4317)
"""

import argparse
import json
import os
import sys
import time
from uuid import uuid4

# Ensure the example directory is on the path for local imports
sys.path.insert(0, os.path.dirname(__file__))


def configure_instrumentation():
    """Set up OpenTelemetry with OTLP export and LangChain instrumentation."""
    from opentelemetry import _events, _logs, metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    from opentelemetry.sdk._events import EventLoggerProvider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    _logs.set_logger_provider(LoggerProvider())
    _logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )
    _events.set_event_logger_provider(EventLoggerProvider())

    LangchainInstrumentor().instrument()


def main():
    parser = argparse.ArgumentParser(
        description="SRE Incident Copilot — interrupt/resume invocation example"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario-001",
        help="Scenario ID to run (default: scenario-001)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve the mitigation plan (skip interactive prompt)",
    )
    parser.add_argument(
        "--auto-reject",
        action="store_true",
        help="Automatically reject the mitigation plan (skip interactive prompt)",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="",
        help="Feedback to include when approving or rejecting",
    )
    parser.add_argument(
        "--wait-after-completion",
        type=int,
        default=60,
        help=(
            "Seconds to wait after both phases complete for "
            "LLM-as-a-judge evaluations to finish (default: 60)"
        ),
    )
    args = parser.parse_args()

    # ---- Environment -------------------------------------------------------
    os.environ.setdefault("OTEL_SERVICE_NAME", "sre-incident-copilot-example")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "DELTA")
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", "SPAN_AND_EVENT"
    )
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event")

    # ---- Instrumentation ---------------------------------------------------
    configure_instrumentation()

    # ---- Imports after instrumentation is configured -----------------------
    from config import Config
    from main import resume_scenario, run_scenario

    config = Config.from_env()
    config.scenario_id = args.scenario
    conversation_id = str(uuid4())

    print("\U0001f6a8 SRE Incident Copilot — Interrupt/Resume Example")
    print("=" * 60)
    print(f"Scenario:        {args.scenario}")
    print(f"Conversation ID: {conversation_id}")
    print("Mode:            interrupt/resume (human review after action planner)")
    print()

    # ---- Phase 1: Run until interrupt --------------------------------------
    print("Phase 1: Running workflow until human review interrupt...")
    print("-" * 60)

    result = run_scenario(args.scenario, config, conversation_id, enable_interrupt=True)

    if not isinstance(result, dict) or result.get("status") != "interrupted":
        print("\n⚠️  Workflow completed without interrupting.")
        print("    (This can happen if triage fails before reaching action_planner)")
        return

    # ---- Display interrupt payload -----------------------------------------
    interrupt_payload = result.get("interrupt", {})
    print("\n" + "=" * 60)
    print("⏸️  WORKFLOW INTERRUPTED — Human Review Required")
    print("=" * 60)
    print(f"\n  Session ID:      {result['session_id']}")
    print(f"  Confidence:      {interrupt_payload.get('confidence_score', 0.0):.2f}")
    print(f"  Top Hypothesis:  {interrupt_payload.get('top_hypothesis', 'N/A')}")
    print("\n  Proposed Mitigation Plan:")
    for i, step in enumerate(interrupt_payload.get("mitigation_plan", []), 1):
        print(f"    {i}. {step}")
    print("\n  Tickets Created:")
    for ticket in interrupt_payload.get("tickets_created", []):
        print(f"    - {ticket}")

    # ---- Determine approval ------------------------------------------------
    if args.auto_approve:
        answer = {"approved": True, "feedback": args.feedback}
        print(f"\n  [auto-approve] answer = {json.dumps(answer)}")
    elif args.auto_reject:
        answer = {"approved": False, "feedback": args.feedback or "rejected by script"}
        print(f"\n  [auto-reject] answer = {json.dumps(answer)}")
    else:
        # Interactive
        print("\n" + "-" * 40)
        approval_input = input("  Approve? (y/n): ").strip().lower()
        feedback_input = ""
        if approval_input not in ("y", "yes"):
            feedback_input = input("  Feedback (optional): ").strip()
        else:
            feedback_input = input("  Feedback (optional): ").strip()
        answer = {
            "approved": approval_input in ("y", "yes"),
            "feedback": feedback_input,
        }

    # ---- Phase 2: Resume ---------------------------------------------------
    print("\n" + "-" * 60)
    print(f"Phase 2: Resuming workflow with: {json.dumps(answer)}")
    print("-" * 60)

    final_state = resume_scenario(args.scenario, config, conversation_id, answer)

    # ---- Results -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ Workflow Completed")
    print("=" * 60)

    quality_result = final_state.get("quality_gate_result") or {}
    print(f"\n  Quality Gate Passed:  {quality_result.get('validation_passed', False)}")
    print(f"  Writeback Allowed:   {quality_result.get('writeback_allowed', False)}")
    print(f"  Confidence Score:    {final_state.get('confidence_score', 0.0):.2f}")

    human_review = final_state.get("human_review_decision")
    if human_review:
        print("\n  Human Review:")
        print(f"    Approved: {human_review.get('approved', 'N/A')}")
        if human_review.get("feedback"):
            print(f"    Feedback: {human_review['feedback']}")

    # ---- Telemetry flush ---------------------------------------------------
    if args.wait_after_completion > 0:
        print(
            f"\n⏳ Waiting {args.wait_after_completion}s for LLM-as-a-judge "
            "evaluations to finish…"
        )
        time.sleep(args.wait_after_completion)

    print("\nDone.")


if __name__ == "__main__":
    main()
