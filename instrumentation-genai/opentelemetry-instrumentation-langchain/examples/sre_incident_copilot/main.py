"""SRE Incident Copilot - Main application."""

import argparse
import atexit
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


def _flush_evaluations(timeout: float = 30.0) -> None:
    """Force evaluation processing and wait for completion."""
    try:
        from opentelemetry.util.genai.handler import get_telemetry_handler

        handler = get_telemetry_handler()
        if handler is not None:
            handler.wait_for_evaluations(timeout)
    except ImportError:
        pass
    except Exception:
        pass


def _flush_telemetry_providers() -> None:
    """Flush all OTLP telemetry providers to ensure data is exported."""
    try:
        from opentelemetry import trace, metrics, _logs

        trace_provider = trace.get_tracer_provider()
        if hasattr(trace_provider, "force_flush"):
            trace_provider.force_flush(timeout_millis=10000)

        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "force_flush"):
            meter_provider.force_flush(timeout_millis=10000)

        logger_provider = _logs.get_logger_provider()
        if hasattr(logger_provider, "force_flush"):
            logger_provider.force_flush(timeout_millis=10000)
    except Exception:
        pass


def _graceful_shutdown() -> None:
    """Perform graceful shutdown: wait for evals and flush telemetry."""
    _flush_evaluations(timeout=60.0)
    time.sleep(1.0)
    _flush_telemetry_providers()
    sys.stdout.flush()


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

    try:
        for step in app.stream(initial_state, config_dict):
            node_name, node_state = next(iter(step.items()))
            final_state = node_state
            nodes_executed.append(node_name)

            # Simple progress indicator
            print(f"  → {node_name.replace('_', ' ').title()}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
    except Exception as e:
        print(f"\n  ⚠️ Workflow error after {nodes_executed}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()

    print(f"  ✓ Completed: {nodes_executed}", flush=True)
    sys.stdout.flush()
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
        "--eval-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds to wait for evaluations (default: 60)",
    )
    args = parser.parse_args()

    # Register graceful shutdown handler
    atexit.register(_graceful_shutdown)

    # Load config
    config = Config.from_env()
    if args.scenario:
        config.scenario_id = args.scenario

    if not config.scenario_id:
        print("Error: --scenario is required or set SCENARIO_ID env var")
        sys.exit(1)

    # Set up OpenTelemetry environment BEFORE configuring instrumentation
    # Exporters read env vars at construction time
    os.environ.setdefault("OTEL_SERVICE_NAME", config.otel_service_name)
    if config.otel_exporter_otlp_endpoint:
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_ENDPOINT", config.otel_exporter_otlp_endpoint
        )
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_PROTOCOL", config.otel_exporter_otlp_protocol
    )
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    # Configure manual instrumentation if requested (AFTER env vars are set)
    if args.manual_instrumentation:
        _configure_manual_instrumentation(config)

    print(f"🚨 SRE Incident Copilot | {config.scenario_id}", flush=True)

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

        quality_result = final_state.get("quality_gate_result") or {}
        validation_passed = validation_report.get('validation_passed', False)
        confidence = final_state.get('confidence_score', 0.0)
        
        status = "✅ PASSED" if validation_passed else "❌ FAILED"
        print(f"\n{status} | Confidence: {confidence:.2f} | Artifacts: {config.artifacts_dir}/{run_id}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        _flush_evaluations(timeout=args.eval_timeout)

    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        _flush_evaluations(timeout=10.0)
        sys.exit(1)


def _configure_manual_instrumentation(config: Config):
    """Configure manual OpenTelemetry instrumentation for production use.
    
    Production best practices:
    - Resource attributes for service identification
    - Batch processors with appropriate timeouts
    - Graceful error handling (don't crash if OTEL fails)
    - Support for both gRPC (default) and HTTP protocols
    
    Environment variables:
    - OTEL_EXPORTER_OTLP_PROTOCOL: "grpc" (default) or "http/protobuf"
    - OTEL_EXPORTER_OTLP_ENDPOINT: Collector endpoint
    - OTEL_EXPORTER_OTLP_TIMEOUT: Export timeout in milliseconds (default: 10000)
    """
    # Suppress gRPC C-level logging BEFORE importing gRPC modules
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GRPC_TRACE", "")
    
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry import _logs, metrics, trace
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    timeout_ms = int(os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "10000"))
    
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    else:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    # Production resource attributes
    resource = Resource.create({
        SERVICE_NAME: config.otel_service_name,
        SERVICE_VERSION: os.environ.get("SERVICE_VERSION", "1.0.0"),
        "deployment.environment": os.environ.get("DEPLOYMENT_ENV", "development"),
    })

    try:
        # Configure TracerProvider with batch processing
        tracer_provider = trace.get_tracer_provider()
        if not hasattr(tracer_provider, 'add_span_processor'):
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
        
        if hasattr(tracer_provider, 'add_span_processor'):
            span_exporter = OTLPSpanExporter(timeout=timeout_ms)
            # BatchSpanProcessor with production settings
            span_processor = BatchSpanProcessor(
                span_exporter,
                max_queue_size=2048,           # Buffer up to 2048 spans
                max_export_batch_size=512,     # Export in batches of 512
                schedule_delay_millis=5000,    # Export every 5 seconds
                export_timeout_millis=timeout_ms,
            )
            tracer_provider.add_span_processor(span_processor)

        # Configure MeterProvider
        meter_provider = metrics.get_meter_provider()
        if not hasattr(meter_provider, 'force_flush'):
            metric_exporter = OTLPMetricExporter(timeout=timeout_ms)
            metric_reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=60000,  # Export metrics every 60 seconds
                export_timeout_millis=timeout_ms,
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)

        # Configure LoggerProvider
        logger_provider = _logs.get_logger_provider()
        if not hasattr(logger_provider, 'add_log_record_processor'):
            logger_provider = LoggerProvider(resource=resource)
            _logs.set_logger_provider(logger_provider)
        
        if hasattr(logger_provider, 'add_log_record_processor'):
            log_exporter = OTLPLogExporter(timeout=timeout_ms)
            log_processor = BatchLogRecordProcessor(
                log_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
                export_timeout_millis=timeout_ms,
            )
            logger_provider.add_log_record_processor(log_processor)

    except Exception as e:
        # Don't crash the app if OTEL setup fails - log and continue
        print(f"Warning: Failed to configure OTEL exporters: {e}", file=sys.stderr)

    # Instrument Langchain (should not fail even if exporters failed)
    try:
        instrumentor = LangchainInstrumentor()
        instrumentor.instrument()
    except Exception as e:
        print(f"Warning: Failed to instrument Langchain: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
