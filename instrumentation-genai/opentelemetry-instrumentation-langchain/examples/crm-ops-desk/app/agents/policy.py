from __future__ import annotations

import os
from datetime import datetime

from colorama import Fore, Style

from app.rag.store import embed, get_db


async def policy_node(state: dict) -> dict:
    """Policy Agent — retrieve relevant policies via FAISS vector search."""
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
    return {"policies": policies}
