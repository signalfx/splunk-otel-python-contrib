"""CRM Ops Desk — Records → Policy → Action → Audit → Summarize → END.

Sub-agents (Records, Policy, Action, Audit) each carry agent_name metadata,
which causes the SDOT instrumentor to emit a nested invoke_agent span per node.

The Summarize node has NO agent_name, so its LLM call runs directly under the
root "CRM Ops Desk" invoke_agent span.  Token usage and latency metrics for
the summary LLM call are therefore attributed to the root agent, making the
overall workflow visible in Splunk AI Agent Monitoring.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agents.action import action_node
from app.agents.audit import audit_node
from app.agents.policy import policy_node
from app.agents.records import records_node
from app.agents.summarize import summarize_node


class CRMState(TypedDict, total=False):
    """State passed between agents in the graph."""

    # Input
    user_query: str
    user_id: str
    scenario: str
    # Agent outputs — stored as serialisable dicts / lists
    records: dict[str, Any]  # {requests, tickets, orders, intent_summary}
    policies: list[dict[str, Any]]
    policy_guidance: str  # LLM interpretation of applicable policy from Policy Agent
    action_output: dict[str, Any]  # ActionOutput.model_dump()
    audit_output: dict[str, Any]  # AuditOutput.model_dump()
    # Summarization
    summary: str  # user-facing natural-language response
    # LangChain messages — used by SDOT to capture input/output on the root GenAI
    # span: first HumanMessage → gen_ai.input.messages, last AIMessage → gen_ai.output.messages
    messages: Annotated[list[AnyMessage], add_messages]
    # Control
    status: str
    error: str


def build_graph():
    """Build and compile the CRM Ops Desk LangGraph."""
    workflow = StateGraph(CRMState)

    # Sub-agents — each gets its own nested invoke_agent span
    workflow.add_node("records", records_node, metadata={"agent_name": "Records Agent"})
    workflow.add_node("policy", policy_node, metadata={"agent_name": "Policy Agent"})
    workflow.add_node("action", action_node, metadata={"agent_name": "Action Agent"})
    workflow.add_node("audit", audit_node, metadata={"agent_name": "Audit Agent"})

    # Summarize — no agent_name → LLM call is a child of the root graph span
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("records")
    workflow.add_edge("records", "policy")
    workflow.add_edge("policy", "action")
    workflow.add_edge("action", "audit")
    workflow.add_edge("audit", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile(name="CRM Ops Desk")
