"""SRE Incident Copilot - Main application."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import (
    action_planner_agent,
    quality_gate_agent,
    triage_agent,
)
from config import Config
from data_loader import DataLoader
from validation import ValidationHarness
from incident_types import IncidentState


def route_after_triage(state: IncidentState) -> str:
    """Route after triage agent - orchestrator decision.

    Note: Investigation is done as agent-as-tool (investigation_agent_mcp called by triage),
    so we route directly to action_planner.
    """
    # Check if triage was successful
    service_id = state.get("service_id")
    triage_result = state.get("triage_result", {})

    if not service_id and not triage_result:
        # Triage failed - end workflow
        return END

    # Check if investigation was done via agent-as-tool
    hypotheses = state.get("hypotheses", [])
    investigation_result = state.get("investigation_result", {})

    if not hypotheses and not investigation_result:
        # Investigation should have been done by triage via investigation_agent_mcp
        # If not, we still proceed but log a warning
        pass

    # Route directly to action_planner (investigation was done as agent-as-tool)
    return "action_planner"


def route_after_action_planner(state: IncidentState) -> str:
    """Route after action planner agent - orchestrator decision."""
    action_plan = state.get("action_plan", {})
    mitigation_plan = action_plan.get("mitigation_plan", [])

    if not mitigation_plan:
        # Will be logged after node execution
        pass

    # Always proceed to quality gate (routing message will be printed after node execution)
    return "quality_gate"


def route_after_quality_gate(state: IncidentState) -> str:
    """Route after quality gate - orchestrator decision with conditional logic."""
    # Always end workflow (routing message will be printed after node execution)
    return END


def build_workflow(config: Config) -> StateGraph:
    """Build the LangGraph workflow with conditional routing (orchestrator pattern).

    Note: Investigation Agent is called as a tool (agent-as-tool pattern) by Triage Agent,
    not as a separate node in the workflow.
    """
    graph = StateGraph(IncidentState)

    # Add nodes (investigation is NOT a node - it's called as a tool by triage)
    graph.add_node("triage", lambda state: triage_agent(state, config))
    graph.add_node("action_planner", lambda state: action_planner_agent(state, config))
    graph.add_node("quality_gate", lambda state: quality_gate_agent(state, config))

    # Add conditional edges - orchestrator pattern with explicit routing
    graph.add_edge(START, "triage")

    # After triage: route to action_planner (investigation was done as agent-as-tool)
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "action_planner": "action_planner",
            END: END,
        },
    )

    # After action_planner: route to quality_gate
    graph.add_conditional_edges(
        "action_planner",
        route_after_action_planner,
        {
            "quality_gate": "quality_gate",
            END: END,
        },
    )

    # After quality_gate: end workflow
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            END: END,
        },
    )

    return graph


def save_artifacts(state: IncidentState, config: Config, run_id: str):
    """Save artifacts from the run."""
    artifacts_dir = Path(config.artifacts_dir) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs
    inputs = {
        "alert_id": state.get("alert_id"),
        "scenario_id": state.get("scenario_id"),
        "service_id": state.get("service_id"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(artifacts_dir / "inputs.json", "w") as f:
        json.dump(inputs, f, indent=2)

    # Save outputs
    outputs = {
        "triage_result": state.get("triage_result"),
        "investigation_result": state.get("investigation_result"),
        "hypotheses": state.get("hypotheses", []),
        "action_plan": state.get("action_plan"),
        "quality_gate_result": state.get("quality_gate_result"),
        "tickets_created": state.get("tickets_created", []),
        "confidence_score": state.get("confidence_score"),
        "eval_metrics": state.get("eval_metrics", {}),
    }
    with open(artifacts_dir / "outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    # Save metadata
    metadata = {
        "run_id": run_id,
        "session_id": state.get("session_id"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "scenario_id": config.scenario_id,
            "drift_enabled": config.drift_enabled,
            "drift_mode": config.drift_mode,
        },
    }
    with open(artifacts_dir / "run_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate incident summary
    if state.get("action_plan"):
        summary = _generate_incident_summary(state)
        with open(artifacts_dir / "incident_summary.md", "w") as f:
            f.write(summary)
        state["incident_summary"] = summary

    # Generate postmortem draft
    postmortem = _generate_postmortem_draft(state)
    with open(artifacts_dir / "postmortem_draft.md", "w") as f:
        f.write(postmortem)
    state["postmortem_draft"] = postmortem


def _generate_incident_summary(state: IncidentState) -> str:
    """Generate incident summary markdown."""
    alert_id = state.get("alert_id", "unknown")
    service_id = state.get("service_id", "unknown")
    hypotheses = state.get("hypotheses", [])
    action_plan = state.get("action_plan", {})
    confidence_score = state.get("confidence_score", 0.0)

    summary = f"""# Incident Summary

## Alert Information
- Alert ID: {alert_id}
- Service: {service_id}
- Confidence Score: {confidence_score:.2f}

## Root Cause Hypotheses
"""
    for i, hyp in enumerate(hypotheses[:3], 1):
        summary += f"""
### Hypothesis {i}
- **Description**: {hyp.get("hypothesis", "N/A")}
- **Confidence**: {hyp.get("confidence", 0.0):.2f}
- **Evidence**: {len(hyp.get("evidence", []))} pieces
"""

    summary += "\n## Recommended Actions\n"
    mitigation_plan = action_plan.get("mitigation_plan", [])
    for i, action in enumerate(mitigation_plan[:5], 1):
        summary += f"{i}. {action}\n"

    summary += "\n## Created Tasks\n"
    tickets = state.get("tickets_created", [])
    for ticket in tickets:
        summary += f"- {ticket.get('id', 'N/A')}: {ticket.get('title', 'N/A')}\n"

    return summary


def _generate_postmortem_draft(state: IncidentState) -> str:
    """Generate postmortem draft markdown."""
    alert_id = state.get("alert_id", "unknown")
    service_id = state.get("service_id", "unknown")

    postmortem = f"""# Postmortem Draft

## Incident Overview
- Alert ID: {alert_id}
- Service: {service_id}
- Timestamp: {datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}

## Timeline
- TBD: Add timeline of events

## Root Cause
- TBD: Add root cause analysis

## Impact
- TBD: Add impact assessment

## Resolution
- TBD: Add resolution steps

## Action Items
- TBD: Add action items

## Lessons Learned
- TBD: Add lessons learned
"""
    return postmortem


def run_scenario(scenario_id: str, config: Config) -> IncidentState:
    """Run a scenario end-to-end."""
    data_loader = DataLoader(data_dir=config.data_dir)

    # Get alert for scenario
    alert = data_loader.get_alert_by_scenario(scenario_id)
    if not alert:
        raise ValueError(f"Scenario {scenario_id} not found")

    # Initialize state
    session_id = str(uuid4())
    initial_state: IncidentState = {
        "messages": [HumanMessage(content=f"Investigate alert: {alert['title']}")],
        "alert_id": alert["id"],
        "scenario_id": scenario_id,
        "service_id": alert["service_id"],
        "incident_context": alert,
        "triage_result": None,
        "investigation_result": None,
        "hypotheses": [],
        "action_plan": None,
        "quality_gate_result": None,
        "incident_summary": None,
        "postmortem_draft": None,
        "tickets_created": [],
        "session_id": session_id,
        "current_agent": "start",
        "confidence_score": 0.0,
        "eval_metrics": {},
    }

    # Build and run workflow
    workflow = build_workflow(config)
    app = workflow.compile()

    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }

    # Execute workflow
    final_state: IncidentState = initial_state
    nodes_executed = []
    previous_node = None

    for step in app.stream(initial_state, config_dict):
        node_name, node_state = next(iter(step.items()))
        final_state = node_state
        nodes_executed.append(node_name)

        # Print routing decision AFTER previous node (if applicable)
        if previous_node:
            # Routing decision was already made, just show it was executed
            pass

        # Print node execution
        print(f"\n🤖 {node_name.replace('_', ' ').title()} Agent")
        if node_state.get("current_agent"):
            print(f"   Status: {node_state['current_agent']}")

        # Print routing decision AFTER node execution
        if node_name == "triage":
            service_id = node_state.get("service_id", "unknown")
            hypotheses = node_state.get("hypotheses", [])
            investigation_result = node_state.get("investigation_result", {})
            if hypotheses or investigation_result:
                print(
                    f"   → Investigation completed via agent-as-tool ({len(hypotheses)} hypotheses)"
                )
                print(f"   → Routing to action_planner (service: {service_id})")
            else:
                print(f"   → Routing to action_planner (service: {service_id})")
                print("   ⚠️  Warning: Investigation not completed via agent-as-tool")
        elif node_name == "action_planner":
            action_plan = node_state.get("action_plan", {})
            mitigation_plan = (
                action_plan.get("mitigation_plan", []) if action_plan else []
            )
            print(f"   → Routing to quality_gate ({len(mitigation_plan)} actions)")
        elif node_name == "quality_gate":
            quality_result = node_state.get("quality_gate_result") or {}
            validation_passed = quality_result.get("validation_passed", False)
            confidence_score = node_state.get("confidence_score", 0.0)
            if validation_passed:
                print(
                    f"   → Quality gate passed (confidence: {confidence_score:.2f}), ending workflow"
                )
            else:
                print(
                    f"   → Quality gate failed (confidence: {confidence_score:.2f}), ending workflow"
                )

        previous_node = node_name

    print("\n📋 Workflow execution summary:")
    print(f"   Nodes executed: {nodes_executed}")
    expected_nodes = ["triage", "action_planner", "quality_gate"]
    missing_nodes = [n for n in expected_nodes if n not in nodes_executed]
    if missing_nodes:
        print(f"   ⚠️  Missing nodes: {missing_nodes}")

    # Check if investigation was done via agent-as-tool
    if final_state.get("hypotheses") or final_state.get("investigation_result"):
        print(
            "   ✓ Investigation completed via agent-as-tool (investigation_agent_mcp)"
        )

    return final_state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SRE Incident Copilot")
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario ID to run (e.g., scenario-001)",
    )
    parser.add_argument(
        "--manual-instrumentation",
        action="store_true",
        help="Enable manual OpenTelemetry instrumentation",
    )
    parser.add_argument(
        "--wait-after-completion",
        type=int,
        default=0,
        help="Number of seconds to wait after completion to ensure evaluations finish (default: 0)",
    )
    args = parser.parse_args()

    # Load config
    config = Config.from_env()
    if args.scenario:
        config.scenario_id = args.scenario

    if not config.scenario_id:
        print("Error: --scenario is required or set SCENARIO_ID env var")
        sys.exit(1)

    # Configure manual instrumentation if requested
    if args.manual_instrumentation:
        _configure_manual_instrumentation(config)

    # Set up OpenTelemetry environment
    os.environ.setdefault("OTEL_SERVICE_NAME", config.otel_service_name)
    if config.otel_exporter_otlp_endpoint:
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_ENDPOINT", config.otel_exporter_otlp_endpoint
        )
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_PROTOCOL", config.otel_exporter_otlp_protocol
    )
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    print("🚨 SRE Incident Copilot")
    print("=" * 60)
    print(f"Scenario: {config.scenario_id}")
    print(f"Service: {config.otel_service_name}")
    print()

    # Run scenario
    run_id = (
        f"{config.scenario_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    try:
        final_state = run_scenario(config.scenario_id, config)

        # Save artifacts
        save_artifacts(final_state, config, run_id)

        # Generate validation report (business logic validation only)
        # Note: Evaluation metrics (bias, toxicity, etc.) are computed automatically
        # by opentelemetry-util-genai-evals via instrumentation
        validation_harness = ValidationHarness(config)
        validation_report = validation_harness.generate_validation_report(
            final_state, run_id
        )
        artifacts_dir = Path(config.artifacts_dir) / run_id
        with open(artifacts_dir / "validation_report.json", "w") as f:
            json.dump(validation_report, f, indent=2)

        print(f"\n✅ Run completed: {run_id}")
        print(f"   Artifacts saved to: {config.artifacts_dir}/{run_id}")

        # Print summary
        quality_result = final_state.get("quality_gate_result") or {}
        print("\n📊 Quality Gate Results:")
        print(
            f"   Validation Passed: {quality_result.get('validation_passed', False) if quality_result else False}"
        )
        print(
            f"   Writeback Allowed: {quality_result.get('writeback_allowed', False) if quality_result else False}"
        )
        print(f"   Confidence Score: {final_state.get('confidence_score', 0.0):.2f}")

        # Print validation results
        print("\n📈 Business Logic Validation:")
        print(
            f"   Overall Validation: {validation_report.get('validation_passed', False)}"
        )
        hypothesis_val = validation_report["validations"]["hypothesis"]
        print(f"   Hypothesis Match: {hypothesis_val.get('hypothesis_match', False)}")
        evidence_val = validation_report["validations"]["evidence"]
        print(
            f"   Evidence Sufficient: {evidence_val.get('evidence_sufficient', False)}"
        )
        action_val = validation_report["validations"]["action_safety"]
        print(f"   Action Safety: {action_val.get('action_safety_validated', False)}")

        # Wait for instrumentation-side evaluations to complete if requested
        if args.wait_after_completion > 0:
            print(
                f"\n⏳ Waiting {args.wait_after_completion} seconds for evaluations to complete..."
            )
            time.sleep(args.wait_after_completion)
            print("   Evaluations should be complete.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _configure_manual_instrumentation(config: Config):
    """Configure manual OpenTelemetry instrumentation."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
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

    instrumentor = LangchainInstrumentor()
    instrumentor.instrument()


if __name__ == "__main__":
    main()
