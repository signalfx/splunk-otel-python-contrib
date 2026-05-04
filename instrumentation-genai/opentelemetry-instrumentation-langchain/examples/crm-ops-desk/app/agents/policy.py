from __future__ import annotations

import json
import os
from datetime import datetime

from colorama import Fore, Style
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.llm import create_chat_llm, embed
from app.rag.store import get_db

_POLICY_PROMPT = """\
You are the Policy Agent in a CRM Operations Desk.
Given the customer's intent and the retrieved policies, identify which policy applies
and state the key rules the Action Agent must follow.

Respond in 2-3 sentences covering:
1. Which policy version applies and why (region, effective date)
2. The refund window and auto-approval threshold
3. Whether escalation is required based on the order amount

Be precise. Quote the policy values (days, dollar amounts) — do not paraphrase loosely."""


async def policy_node(state: dict, config: RunnableConfig) -> dict:
    """Policy Agent — retrieve policies, then interpret applicability with an LLM call."""
    print(f"{Fore.CYAN}-> Policy Agent: Starting{Style.RESET_ALL}")
    user_query = state["user_query"]
    records = state.get("records", {})
    orders = records.get("orders", [])

    # Determine region from first order's shipping address
    if not orders:
        region = "US"
    else:
        country = orders[0].get("shipping_address", {}).get("country", "").upper()
        if country in ("US", "USA", "UNITED STATES"):
            region = "US"
        elif country in (
            "UK",
            "GB",
            "UNITED KINGDOM",
            "FR",
            "FRANCE",
            "DE",
            "GERMANY",
            "IT",
            "ITALY",
            "ES",
            "SPAIN",
        ):
            region = "EU"
        else:
            region = "EU"  # Default to EU for non-US countries

    db = get_db()
    try:
        vec = embed([user_query])[0]
        policies = db.policies.vector_search(vec, limit=10)
    except Exception as e:
        print(f"  {Fore.YELLOW}Vector search failed: {e}{Style.RESET_ALL}")
        policies = db.policies.find(limit=5)

    # Filter by region
    policies = [p for p in policies if p.get("region") == region]

    # Use old expired policies if drift toggle is set
    if os.getenv("POLICY_FORCE_OLD_VERSION", "false").lower() == "true":
        expired = [p for p in policies if p.get("effective_until") is not None]
        if expired:
            expired.sort(
                key=lambda x: x.get("effective_from", datetime.min), reverse=True
            )
            policies = expired
            print(
                f"  {Fore.YELLOW}Drift mode: using expired policy {policies[0].get('version')}{Style.RESET_ALL}"
            )

    print(
        f"  {Fore.CYAN}Found {len(policies)} policies for region {region}{Style.RESET_ALL}"
    )

    # LLM call — interpret which policy applies and what constraints it sets
    intent_summary = records.get("intent_summary", user_query)
    policy_data = json.dumps(policies, indent=2, default=str)
    order_amount = orders[0].get("unit_price", 0) if orders else 0

    llm = create_chat_llm(temperature=0.1)
    response = await llm.ainvoke(
        [
            SystemMessage(content=_POLICY_PROMPT),
            HumanMessage(
                content=(
                    f"Customer intent: {intent_summary}\n"
                    f"Order amount: ${order_amount:.2f}\n"
                    f"Region: {region}\n\n"
                    f"Retrieved policies:\n{policy_data}"
                )
            ),
        ],
        config=config,
    )
    policy_guidance = response.content.strip()
    print(f"  {Fore.CYAN}Guidance: {policy_guidance[:120]}...{Style.RESET_ALL}")

    return {"policies": policies, "policy_guidance": policy_guidance}
