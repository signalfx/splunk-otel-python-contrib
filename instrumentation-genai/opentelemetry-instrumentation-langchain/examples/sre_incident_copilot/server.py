"""SRE Incident Copilot - FastAPI Server.

Production-ready server for handling incident investigation requests.
Designed to be deployed in Kubernetes and called via CronJob or HTTP requests.

Usage:
    # With zero-code instrumentation (recommended for production):
    opentelemetry-instrument uvicorn server:app --host 0.0.0.0 --port 8080

    # Or run directly (instrumentation configured in code):
    uvicorn server:app --host 0.0.0.0 --port 8080
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import action_planner_agent, quality_gate_agent, triage_agent
from config import Config
from data_loader import DataLoader
from incident_types import IncidentState
from validation import ValidationHarness


# Request/Response models
class RunScenarioRequest(BaseModel):
    scenario_id: str
    drift_mode: Optional[str] = None
    drift_intensity: Optional[float] = 0.0
    wait_for_evals: Optional[bool] = False  # Wait for evaluations before responding


class RunScenarioResponse(BaseModel):
    run_id: str
    scenario_id: str
    validation_passed: bool
    confidence_score: float
    artifacts_path: str
    hypotheses_count: int
    tasks_created: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# Initialize instrumentation at module load for zero-code support
def _setup_instrumentation():
    """Setup LangChain instrumentation only (no OTEL providers)."""
    try:
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor
        
        instrumentor = LangchainInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    except Exception as e:
        print(f"Warning: Failed to setup instrumentation: {e}", file=sys.stderr)


def _configure_manual_instrumentation():
    """Configure OTEL providers manually with OTLP exporters.
    
    Set MANUAL_INSTRUMENTATION=true to enable.
    """
    from opentelemetry import trace, metrics, _events, _logs
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk._events import EventLoggerProvider
    
    config = Config.from_env()
    protocol = config.otel_exporter_otlp_protocol
    
    if protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    else:  # grpc (default)
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    
    resource = Resource.create({
        SERVICE_NAME: config.otel_service_name,
        SERVICE_VERSION: os.environ.get("SERVICE_VERSION", "1.0.0"),
        "deployment.environment": os.environ.get("DEPLOYMENT_ENV", "development"),
    })
    
    try:
        # Setup TracerProvider
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
        
        # Setup MeterProvider
        metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        
        # Setup LoggerProvider (for logs)
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        _logs.set_logger_provider(logger_provider)
        
        # Setup EventLoggerProvider (required for GenAI events like gen_ai.client.inference.operation.details)
        event_logger_provider = EventLoggerProvider(logger_provider)
        _events.set_event_logger_provider(event_logger_provider)
        
    except Exception as e:
        print(f"Warning: Failed to configure OTEL providers: {e}", file=sys.stderr)
    
    # Now instrument LangChain
    _setup_instrumentation()


def _wait_for_evaluations(timeout: float = 30.0):
    """Wait for pending evaluations to complete.
    
    Required because evaluations run in daemon threads.
    """
    try:
        from opentelemetry.util.genai.evals import wait_for_evaluations
        wait_for_evaluations(timeout=timeout)
    except ImportError:
        pass  # Evals not installed
    except Exception as e:
        print(f"Warning: Error waiting for evaluations: {e}", file=sys.stderr)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup - check for manual instrumentation
    manual_instr = os.environ.get("MANUAL_INSTRUMENTATION", "").lower() in ("true", "1", "yes")
    if manual_instr:
        print("[SRE Copilot] Using manual instrumentation", flush=True)
        _configure_manual_instrumentation()
    else:
        # Zero-code instrumentation: opentelemetry-instrument handles everything
        # (OTEL providers + auto-instrumentation of LangChain)
        print("[SRE Copilot] Using zero-code instrumentation", flush=True)
    print("[SRE Copilot] Server started", flush=True)
    yield
    # Shutdown - wait for any pending evaluations
    print("[SRE Copilot] Shutting down...", flush=True)
    _wait_for_evaluations(timeout=60.0)
    print("[SRE Copilot] Shutdown complete", flush=True)


# Create FastAPI app
app = FastAPI(
    title="SRE Incident Copilot",
    description="AI-powered incident investigation and response",
    version="1.0.0",
    lifespan=lifespan,
)


# Routing functions
def route_after_triage(state: IncidentState) -> str:
    """Route after triage agent."""
    service_id = state.get("service_id")
    triage_result = state.get("triage_result", {})
    if not service_id and not triage_result:
        return END
    return "action_planner"


def route_after_action_planner(state: IncidentState) -> str:
    """Route after action planner agent."""
    return "quality_gate"


def route_after_quality_gate(state: IncidentState) -> str:
    """Route after quality gate."""
    return END


def build_workflow(config: Config) -> StateGraph:
    """Build the LangGraph workflow."""
    graph = StateGraph(IncidentState)
    
    graph.add_node("triage", lambda state: triage_agent(state, config))
    graph.add_node("action_planner", lambda state: action_planner_agent(state, config))
    graph.add_node("quality_gate", lambda state: quality_gate_agent(state, config))
    
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route_after_triage, {"action_planner": "action_planner", END: END})
    graph.add_conditional_edges("action_planner", route_after_action_planner, {"quality_gate": "quality_gate", END: END})
    graph.add_conditional_edges("quality_gate", route_after_quality_gate, {END: END})
    
    return graph


def save_artifacts(state: IncidentState, config: Config, run_id: str) -> str:
    """Save artifacts from the run. Returns artifacts path."""
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
    
    # Generate postmortem draft
    postmortem = _generate_postmortem_draft(state)
    with open(artifacts_dir / "postmortem_draft.md", "w") as f:
        f.write(postmortem)
    
    return str(artifacts_dir)


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
    
    return f"""# Postmortem Draft

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


def run_scenario(scenario_id: str, config: Config) -> IncidentState:
    """Run a scenario end-to-end."""
    data_loader = DataLoader(data_dir=config.data_dir)
    
    alert = data_loader.get_alert_by_scenario(scenario_id)
    if not alert:
        raise ValueError(f"Scenario {scenario_id} not found")
    
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
    
    workflow = build_workflow(config)
    compiled_app = workflow.compile()
    
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }
    
    # Execute workflow and track nodes
    final_state = initial_state
    nodes_executed = []
    for event in compiled_app.stream(initial_state, config_dict):
        for node_name in event:
            nodes_executed.append(node_name)
            final_state = event[node_name]
    
    print(f"[SRE Copilot] Workflow: {' -> '.join(nodes_executed)}", flush=True)
    
    return final_state


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for K8s probes."""
    return HealthResponse(
        status="healthy",
        service="sre-incident-copilot",
        version="1.0.0",
    )


@app.post("/run", response_model=RunScenarioResponse)
async def run_incident_scenario(request: RunScenarioRequest):
    """Run an incident investigation scenario.
    
    This endpoint is designed to be called by K8s CronJob or external triggers.
    """
    print(f"[SRE Copilot] Starting scenario: {request.scenario_id}", flush=True)
    
    config = Config.from_env()
    config.scenario_id = request.scenario_id
    
    if request.drift_mode:
        config.drift_enabled = True
        config.drift_mode = request.drift_mode
        config.drift_intensity = request.drift_intensity or 0.1
    
    try:
        # Run the scenario
        print(f"[SRE Copilot] Running workflow...", flush=True)
        final_state = run_scenario(request.scenario_id, config)
        
        # Generate run ID
        run_id = f"{request.scenario_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        
        # Save artifacts
        print(f"[SRE Copilot] Saving artifacts to {run_id}", flush=True)
        artifacts_path = save_artifacts(final_state, config, run_id)
        
        # Generate validation report
        print(f"[SRE Copilot] Generating validation report", flush=True)
        validation_harness = ValidationHarness(config)
        validation_report = validation_harness.generate_validation_report(final_state, run_id)
        
        # Save validation report
        artifacts_dir = Path(config.artifacts_dir) / run_id
        with open(artifacts_dir / "validation_report.json", "w") as f:
            json.dump(validation_report, f, indent=2)
        
        # Optionally wait for evaluations before responding
        if request.wait_for_evals:
            print(f"[SRE Copilot] Waiting for evaluations...", flush=True)
            _wait_for_evaluations(timeout=30.0)
        
        validation_passed = validation_report.get("validation_passed", False)
        confidence = final_state.get("confidence_score", 0.0)
        status = "PASSED" if validation_passed else "FAILED"
        print(f"[SRE Copilot] Complete: {status} | confidence={confidence:.2f} | artifacts={run_id}", flush=True)
        
        return RunScenarioResponse(
            run_id=run_id,
            scenario_id=request.scenario_id,
            validation_passed=validation_passed,
            confidence_score=confidence,
            artifacts_path=artifacts_path,
            hypotheses_count=len(final_state.get("hypotheses", [])),
            tasks_created=len(final_state.get("tickets_created", [])),
        )
        
    except ValueError as e:
        print(f"[SRE Copilot] Error: {e}", flush=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[SRE Copilot] Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/scenarios")
async def list_scenarios():
    """List available scenarios."""
    config = Config.from_env()
    data_loader = DataLoader(data_dir=config.data_dir)
    alerts = data_loader.get_all_alerts()
    
    return {
        "scenarios": [
            {
                "id": alert.get("scenario_id"),
                "title": alert.get("title"),
                "service_id": alert.get("service_id"),
                "severity": alert.get("severity"),
            }
            for alert in alerts
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
