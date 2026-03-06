"""Type definitions for SRE Incident Copilot."""

from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class IncidentState(TypedDict):
    """State that flows through the LangGraph workflow."""

    # Messages
    messages: Annotated[List[AnyMessage], add_messages]

    # Incident context
    alert_id: Optional[str]
    scenario_id: Optional[str]
    service_id: Optional[str]
    incident_context: Optional[Dict]

    # Agent outputs
    triage_result: Optional[Dict]
    investigation_result: Optional[Dict]
    hypotheses: List[Dict]
    action_plan: Optional[Dict]
    quality_gate_result: Optional[Dict]

    # Artifacts
    incident_summary: Optional[str]
    postmortem_draft: Optional[str]
    tickets_created: List[Dict]

    # Metadata
    session_id: str
    current_agent: str
    confidence_score: float
    eval_metrics: Dict
