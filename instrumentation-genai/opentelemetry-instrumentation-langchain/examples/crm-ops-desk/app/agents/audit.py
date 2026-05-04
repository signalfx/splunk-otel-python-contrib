from __future__ import annotations

import json
import time
from datetime import datetime

from colorama import Fore, Style
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.llm import create_chat_llm
from app.models.action_output import ToolReceipt
from app.models.audit_output import AuditOutput, Citation

_AUDIT_PROMPT = """\
You are the Audit Agent in a CRM Operations Desk.
Write a concise compliance rationale (3-5 sentences) explaining the decision made for
this customer interaction. This rationale is stored for internal compliance records.

Cover:
1. What the customer requested
2. Which policy governed the decision and its key constraints
3. What actions were taken (tools executed) and whether they succeeded
4. Whether the resolution is compliant with the policy

Be precise and factual. Reference policy versions and tool names explicitly."""


async def audit_node(state: dict, config: RunnableConfig) -> dict:
    """Audit Agent — generate LLM-written compliance rationale and citations."""
    print(f"{Fore.BLUE}-> Audit Agent: Starting{Style.RESET_ALL}")
    policies = state.get("policies", [])
    action_raw = state.get("action_output", {})
    records = state.get("records", {})

    citations = [
        Citation(
            source="policy_database",
            doc_id=str(p.get("_id", "")),
            version=p.get("version", "?"),
            relevance_score=0.95,
        )
        for p in policies
    ]

    tool_receipts = [ToolReceipt(**r) for r in action_raw.get("tool_receipts", [])]

    # Build structured context for the LLM
    policy_summary = [
        {
            "version": p.get("version"),
            "region": p.get("region"),
            "refund_window_days": p.get("refund_window_days"),
            "max_refund_amount": p.get("max_refund_amount"),
            "effective_until": str(p.get("effective_until", "current")),
        }
        for p in policies[:3]
    ]
    tool_results = [
        {
            "tool": r.tool,
            "status": r.status,
            "response": r.response,
        }
        for r in tool_receipts
    ]
    audit_context = json.dumps(
        {
            "user_id": state.get("user_id"),
            "user_query": state.get("user_query"),
            "intent_summary": records.get("intent_summary", ""),
            "policy_guidance": state.get("policy_guidance", ""),
            "policies_applied": policy_summary,
            "resolution": action_raw.get("resolution", "unknown"),
            "tools_executed": tool_results,
        },
        indent=2,
        default=str,
    )

    # LLM call — generate compliance rationale
    llm = create_chat_llm(temperature=0.1)
    response = await llm.ainvoke(
        [
            SystemMessage(content=_AUDIT_PROMPT),
            HumanMessage(content=audit_context),
        ],
        config=config,
    )
    rationale = response.content.strip()

    span_ids = [
        f"policy_{int(time.time() * 1000)}",
        f"records_{int(time.time() * 1000)}",
        f"action_{int(time.time() * 1000)}",
        f"audit_{int(time.time() * 1000)}",
    ]

    audit = AuditOutput(
        interaction_id=f"int_{int(time.time() * 1000)}",
        span_ids=span_ids,
        citations=citations,
        tool_receipts=tool_receipts,
        final_verdict=action_raw.get("resolution", "unknown"),
        rationale=rationale,
        created_at=datetime.utcnow().isoformat(),
    )

    print(f"  {Fore.BLUE}Audit complete{Style.RESET_ALL}")
    return {"audit_output": audit.model_dump(), "status": "completed"}
