from __future__ import annotations

import time
from datetime import datetime

from colorama import Fore, Style

from app.models.action_output import ToolReceipt
from app.models.audit_output import AuditOutput, Citation


async def audit_node(state: dict) -> dict:
    """Audit Agent — generate human-readable rationale and citations."""
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

    # Build rationale
    parts: list[str] = []
    parts.append(
        f"Customer {state['user_id']} submitted: "
        f"'{state['user_query'][:100]}{'...' if len(state['user_query']) > 100 else ''}'"
    )

    if policies:
        regions = list({p.get("region") for p in policies})
        versions = list({p.get("version") for p in policies})
        parts.append(
            f"Retrieved {len(policies)} policies (regions: {', '.join(regions)}, versions: {', '.join(versions)})"
        )
        expired = [p for p in policies if p.get("effective_until")]
        if expired:
            parts.append(f"Policy drift: expired policy {expired[0].get('version')}")

    reqs = records.get("requests", [])
    tkts = records.get("tickets", [])
    if reqs:
        total = sum(r.get("amount", 0) for r in reqs)
        parts.append(f"Found {len(reqs)} refund requests (total: ${total:.2f})")
    if tkts:
        parts.append(f"Found {len(tkts)} support tickets")

    ok_tools = [r.tool for r in tool_receipts if 200 <= r.status < 300]
    fail_tools = [r.tool for r in tool_receipts if r.status >= 300]
    if ok_tools:
        parts.append(f"Executed: {', '.join(ok_tools)}")
    if fail_tools:
        parts.append(f"Failed: {', '.join(fail_tools)}")

    resolution = action_raw.get("resolution", "unknown")
    parts.append(f"Resolution: {resolution}")
    cost = action_raw.get("cost_token_usd", 0)
    if cost:
        parts.append(f"Cost: ${cost:.4f}")

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
        final_verdict=resolution,
        rationale=" | ".join(parts),
        created_at=datetime.utcnow().isoformat(),
    )

    print(f"  {Fore.BLUE}Audit complete{Style.RESET_ALL}")
    return {"audit_output": audit.model_dump(), "status": "completed"}
