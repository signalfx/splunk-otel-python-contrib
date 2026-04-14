"""SRE Incident Copilot - Main application.

Demonstrates automatic gen_ai.conversation.id inference from LangGraph's
configurable thread_id. No manual genai_context() wrapping needed — the
instrumentation extracts thread_id from callback metadata and maps it to
gen_ai.conversation.id on all root spans automatically.

Supports interrupt/resume for human-in-the-loop review.

Usage:
    # Normal run (no interrupt)
    python main.py --scenario scenario-001

    # Simulate interrupt/resume in a SINGLE process (two traces, one
    # conversation_id) — ideal for demonstrating evaluations:
    python main.py --scenario scenario-001 --simulate-interrupt-resume \\
        --conversation-id troubleshooting-chat-1 \\
        --manual-instrumentation --wait-after-completion 300

    # Same, but with a Workflow root span instead of AgentInvocation:
    python main.py --scenario scenario-001 --simulate-interrupt-resume \\
        --conversation-id troubleshooting-chat-1 \\
        --manual-instrumentation --wait-after-completion 300 \\
        --root-as-workflow "SRE Incident Copilot"

    # Cross-process interrupt/resume (TWO separate invocations):
    CONVERSATION_ID="troubleshooting-chat-1"
    python main.py --scenario scenario-001 --enable-interrupt \\
        --conversation-id $CONVERSATION_ID
    python main.py --scenario scenario-001 --resume --approve \\
        --conversation-id $CONVERSATION_ID
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents import (
    action_planner_agent,
    quality_gate_agent,
    triage_agent,
)
from config import Config
from data_loader import DataLoader
from validation import ValidationHarness
from incident_types import IncidentState

# Persistent checkpoint directory — stores SQLite DBs keyed by conversation_id
CHECKPOINTS_DIR = Path("checkpoints")


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


def human_review_node(state: IncidentState) -> IncidentState:
    """Pause execution for human review of the proposed mitigation plan.

    Uses LangGraph's ``interrupt()`` to suspend the graph after the action
    planner produces a mitigation plan. The caller resumes via
    ``Command(resume=answer)`` where *answer* is a dict like
    ``{"approved": True}`` or ``{"approved": False, "feedback": "..."}``.
    """
    action_plan = state.get("action_plan", {})
    mitigation_plan = action_plan.get("mitigation_plan", [])
    hypotheses = state.get("hypotheses", [])
    top_hypothesis = hypotheses[0] if hypotheses else {}

    answer = interrupt(
        {
            "question": (
                "Please review the proposed mitigation plan and approve "
                "to proceed with quality gate validation."
            ),
            "top_hypothesis": top_hypothesis.get("hypothesis", "N/A"),
            "confidence_score": state.get("confidence_score", 0.0),
            "mitigation_plan": mitigation_plan,
            "tickets_created": [
                t.get("title", "N/A") for t in state.get("tickets_created", [])
            ],
        }
    )
    # Store the human review decision
    state["human_review_decision"] = answer
    # Incorporate optional feedback into conversation
    if isinstance(answer, dict) and answer.get("feedback"):
        state["messages"].append(
            HumanMessage(
                content=f"SRE engineer feedback on mitigation plan: {answer['feedback']}"
            )
        )
    state["current_agent"] = "quality_gate"
    return state


def _interrupt_enabled() -> bool:
    """Return True when the SRE_COPILOT_INTERRUPT env var is set."""
    return os.getenv("SRE_COPILOT_INTERRUPT", "").lower() in ("1", "true", "yes")


def _get_checkpointer(conversation_id: str) -> SqliteSaver:
    """Return a SqliteSaver backed by a per-conversation SQLite file."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = CHECKPOINTS_DIR / f"{conversation_id}.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_workflow(config: Config, enable_interrupt: bool = False) -> StateGraph:
    """Build the LangGraph workflow with conditional routing (orchestrator pattern).

    When *enable_interrupt* is True, a ``human_review`` node is inserted
    between ``action_planner`` and ``quality_gate``.  The node calls
    ``interrupt()`` so an SRE engineer can approve the mitigation plan
    before the quality gate runs.

    Note: Investigation Agent is called as a tool (agent-as-tool pattern) by Triage Agent,
    not as a separate node in the workflow.
    """
    graph = StateGraph(IncidentState)

    # Add nodes (investigation is NOT a node - it's called as a tool by triage)
    graph.add_node("triage", lambda state: triage_agent(state, config))
    graph.add_node("action_planner", lambda state: action_planner_agent(state, config))
    graph.add_node("quality_gate", lambda state: quality_gate_agent(state, config))

    if enable_interrupt:
        graph.add_node("human_review", human_review_node)

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

    if enable_interrupt:
        # action_planner -> human_review -> quality_gate
        graph.add_edge("action_planner", "human_review")
        graph.add_edge("human_review", "quality_gate")
    else:
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


def run_scenario(
    scenario_id: str,
    config: Config,
    conversation_id: str | None = None,
    enable_interrupt: bool = False,
    workflow_name: str | None = None,
) -> IncidentState | Dict:
    """Run a scenario end-to-end.

    When *enable_interrupt* is True the graph pauses after the action planner
    for human review.  Returns a dict with ``"status": "interrupted"`` and
    the interrupt payload so the caller can inspect the mitigation plan and
    resume.
    """
    data_loader = DataLoader(data_dir=config.data_dir)

    # Get alert for scenario
    alert = data_loader.get_alert_by_scenario(scenario_id)
    if not alert:
        raise ValueError(f"Scenario {scenario_id} not found")

    # Initialize state - use provided conversation_id or generate new one
    conversation_id = conversation_id or str(uuid4())
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
        "human_review_decision": None,
        "incident_summary": None,
        "postmortem_draft": None,
        "tickets_created": [],
        "session_id": conversation_id,
        "current_agent": "start",
        "confidence_score": 0.0,
        "eval_metrics": {},
    }

    # Build and run workflow
    workflow = build_workflow(config, enable_interrupt=enable_interrupt)
    checkpointer = _get_checkpointer(conversation_id) if enable_interrupt else None
    compiled_app = workflow.compile(checkpointer=checkpointer)

    # conversation_id is passed as LangGraph's configurable thread_id.
    # The instrumentation automatically infers gen_ai.conversation.id from it.
    config_dict = {
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": 20,
    }

    # When workflow_name is provided, add it to config metadata so the
    # instrumentation creates a Workflow root span instead of AgentInvocation.
    if workflow_name:
        config_dict["metadata"] = {"workflow_name": workflow_name}

    # LangGraph's thread_id is automatically inferred as gen_ai.conversation.id
    # on all telemetry spans (root Workflow/AgentInvocation and their children).
    # No manual genai_context() wrapping needed.
    # Priority: explicit genai_context(conversation_id=...) > inferred thread_id > none.
    final_state: IncidentState = initial_state
    nodes_executed = []
    previous_node = None

    for step in compiled_app.stream(initial_state, config_dict):
        node_name, node_state = next(iter(step.items()))

        # When interrupt() fires, LangGraph emits a special __interrupt__
        # entry whose value is a tuple, not a dict — skip it.
        if not isinstance(node_state, dict):
            continue

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

    # ---- Check if the graph was interrupted --------------------------------
    if enable_interrupt:
        graph_state = compiled_app.get_state(config_dict)
        if graph_state.next:
            # Graph paused at human_review — return interrupt payload
            interrupt_value = None
            if graph_state.tasks:
                for task in graph_state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        break

            return {
                "status": "interrupted",
                "session_id": conversation_id,
                "interrupt": interrupt_value,
                "nodes_executed": nodes_executed,
                "action_plan": final_state.get("action_plan"),
                "hypotheses": final_state.get("hypotheses", []),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "tickets_created": final_state.get("tickets_created", []),
            }

    # ---- Normal completion -------------------------------------------------
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


def resume_scenario(
    scenario_id: str,
    config: Config,
    conversation_id: str,
    answer: Dict,
    workflow_name: str | None = None,
) -> IncidentState:
    """Resume a previously interrupted scenario.

    Rebuilds the graph with the same persistent ``SqliteSaver`` so
    LangGraph can restore state from the SQLite checkpoint.  *answer*
    is forwarded to the graph via ``Command(resume=answer)``.
    """
    workflow = build_workflow(config, enable_interrupt=True)
    checkpointer = _get_checkpointer(conversation_id)
    compiled_app = workflow.compile(checkpointer=checkpointer)

    config_dict = {
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": 20,
    }
    if workflow_name:
        config_dict["metadata"] = {"workflow_name": workflow_name}

    final_state: Optional[IncidentState] = None
    nodes_executed = []

    for step in compiled_app.stream(Command(resume=answer), config_dict):
        node_name, node_state = next(iter(step.items()))
        if not isinstance(node_state, dict):
            continue
        final_state = node_state
        nodes_executed.append(node_name)
        print(f"\n🤖 {node_name.replace('_', ' ').title()} Agent")
        if node_state.get("current_agent"):
            print(f"   Status: {node_state['current_agent']}")

    print("\n📋 Resume execution summary:")
    print(f"   Nodes executed: {nodes_executed}")

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
        "--conversation-id",
        type=str,
        default=None,
        help="Conversation ID mapped to gen_ai.conversation.id (default: random UUID)",
    )
    parser.add_argument(
        "--enable-interrupt",
        action="store_true",
        help="Enable human review interrupt after action planner",
    )
    parser.add_argument(
        "--simulate-interrupt-resume",
        action="store_true",
        help=(
            "Run interrupt and resume in a single process. "
            "Produces two distinct traces under the same conversation ID. "
            "Auto-approves by default; combine with --reject/--feedback to change."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted workflow (requires --conversation-id)",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the mitigation plan when resuming (non-interactive)",
    )
    parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject the mitigation plan when resuming (non-interactive)",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="",
        help="Feedback to include when approving or rejecting",
    )
    parser.add_argument(
        "--root-as-workflow",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Use a Workflow root span instead of the default AgentInvocation. "
            "The provided NAME becomes the workflow_name in LangGraph config "
            "metadata (e.g. --root-as-workflow 'SRE Incident Copilot')."
        ),
    )
    parser.add_argument(
        "--wait-after-completion",
        type=int,
        default=0,
        help="Seconds to wait after completion for telemetry flush",
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
    #if args.manual_instrumentation:
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

    conversation_id = args.conversation_id or str(uuid4())
    enable_interrupt = args.enable_interrupt or _interrupt_enabled()

    print("\U0001f6a8 SRE Incident Copilot")
    print("=" * 60)
    print(f"Scenario: {config.scenario_id}")
    print(f"Service: {config.otel_service_name}")
    print(f"Conversation ID (→ gen_ai.conversation.id): {conversation_id}")
    if args.root_as_workflow:
        print(f"Root span type: Workflow (name: {args.root_as_workflow})")
    else:
        print("Root span type: AgentInvocation (default)")
    if args.resume:
        print("Mode: resume (continuing interrupted workflow)")
    elif args.simulate_interrupt_resume:
        print("Mode: simulate-interrupt-resume (single process, two traces)")
    elif enable_interrupt:
        print("Mode: interrupt (will pause for human review after action planner)")
    print()

    # Run scenario
    run_id = (
        f"{config.scenario_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    try:
        # ---- Resume path ---------------------------------------------------
        if args.resume:
            if not args.conversation_id:
                print("Error: --resume requires --conversation-id")
                sys.exit(1)

            # Determine approval answer
            if args.approve:
                answer = {"approved": True, "feedback": args.feedback}
            elif args.reject:
                answer = {
                    "approved": False,
                    "feedback": args.feedback or "rejected",
                }
            else:
                # Interactive
                approval = input("Approve? (y/n): ").strip().lower()
                feedback = input("Feedback (optional): ").strip()
                answer = {
                    "approved": approval in ("y", "yes"),
                    "feedback": feedback,
                }

            print(f"▶️  Resuming with: {json.dumps(answer)}")
            final_state = resume_scenario(
                config.scenario_id,
                config,
                conversation_id,
                answer,
                workflow_name=args.root_as_workflow,
            )

        # ---- Simulate interrupt + resume in one process --------------------
        elif args.simulate_interrupt_resume:
            # Phase 1: run until interrupt (produces Trace 1)
            print("Phase 1: Running workflow until human review interrupt…")
            print("-" * 60)

            result = run_scenario(
                config.scenario_id,
                config,
                conversation_id,
                enable_interrupt=True,
                workflow_name=args.root_as_workflow,
            )

            if not isinstance(result, dict) or result.get("status") != "interrupted":
                print(
                    "\n⚠️  Workflow completed without interrupting."
                    "\n    (This can happen if triage fails before "
                    "reaching action_planner)"
                )
                final_state = result
            else:
                # Display the interrupt payload
                interrupt_payload = result.get("interrupt", {})
                print("\n" + "=" * 60)
                print("⏸️  WORKFLOW INTERRUPTED — Human Review Required")
                print("=" * 60)
                print(
                    f"\n  Confidence:      "
                    f"{interrupt_payload.get('confidence_score', 0.0):.2f}"
                )
                print(
                    f"  Top Hypothesis:  "
                    f"{interrupt_payload.get('top_hypothesis', 'N/A')}"
                )
                print("\n  Proposed Mitigation Plan:")
                for i, step in enumerate(
                    interrupt_payload.get("mitigation_plan", []), 1
                ):
                    print(f"    {i}. {step}")
                print("\n  Tickets Created:")
                for ticket in interrupt_payload.get("tickets_created", []):
                    print(f"    - {ticket}")

                # Determine approval answer
                if args.reject:
                    answer = {
                        "approved": False,
                        "feedback": args.feedback or "rejected",
                    }
                else:
                    # Default: auto-approve
                    answer = {"approved": True, "feedback": args.feedback}

                # Phase 2: resume (produces Trace 2)
                print("\n" + "-" * 60)
                print(f"Phase 2: Resuming workflow with: {json.dumps(answer)}")
                print("-" * 60)

                final_state = resume_scenario(
                    config.scenario_id,
                    config,
                    conversation_id,
                    answer,
                    workflow_name=args.root_as_workflow,
                )

        # ---- Normal / interrupt path ---------------------------------------
        else:
            result = run_scenario(
                config.scenario_id,
                config,
                conversation_id,
                enable_interrupt,
                workflow_name=args.root_as_workflow,
            )

            if isinstance(result, dict) and result.get("status") == "interrupted":
                interrupt_payload = result.get("interrupt", {})
                print("\n" + "=" * 60)
                print("⏸️  WORKFLOW INTERRUPTED — Human Review Required")
                print("=" * 60)
                print(f"\nQuestion: {interrupt_payload.get('question', 'N/A')}")
                print(
                    f"Top Hypothesis: {interrupt_payload.get('top_hypothesis', 'N/A')}"
                )
                print(
                    f"Confidence: {interrupt_payload.get('confidence_score', 0.0):.2f}"
                )
                print("\nProposed Mitigation Plan:")
                for i, step in enumerate(
                    interrupt_payload.get("mitigation_plan", []), 1
                ):
                    print(f"  {i}. {step}")
                print("\nTickets Created:")
                for ticket in interrupt_payload.get("tickets_created", []):
                    print(f"  - {ticket}")

                # Print the resume command
                print("\n" + "-" * 60)
                print("To resume, run:")
                print(
                    f"  python main.py --scenario {config.scenario_id}"
                    f" --conversation-id {conversation_id}"
                    f" --resume --approve"
                )
                print(
                    f"  python main.py --scenario {config.scenario_id}"
                    f" --conversation-id {conversation_id}"
                    f' --resume --reject --feedback "your feedback"'
                )

                # Wait for telemetry flush before exiting
                if args.wait_after_completion > 0:
                    print(
                        f"\n⏳ Waiting for {args.wait_after_completion} seconds before exit..."
                    )
                    time.sleep(args.wait_after_completion)
                return

            final_state = result

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

        # Wait after completion if requested (for evaluator to evalute invocations)
        if args.wait_after_completion > 0:
            print(
                f"\n⏳ Waiting for {args.wait_after_completion} seconds before exit..."
            )
            time.sleep(args.wait_after_completion)

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
