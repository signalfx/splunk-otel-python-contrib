from __future__ import annotations

import json

from colorama import Fore, Style
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.llm import create_chat_llm, embed
from app.rag.store import get_db

_INTENT_PROMPT = """\
You are the Records Agent in a CRM Operations Desk.
Given a customer's query and their retrieved records, produce a concise intent summary
that the downstream Action Agent can act on.

Respond in 2-3 sentences covering:
1. What the customer is asking for (refund, status check, escalation, etc.)
2. Whether relevant records exist (open refund requests, prior tickets, delivered orders)
3. Any notable context (repeated complaints, high-value order, already-escalated ticket)

Be factual. Use only the data provided — do not invent details."""


async def records_node(state: dict, config: RunnableConfig) -> dict:
    """Records Agent — fetch records, then summarize intent with an LLM call."""
    print(
        f"{Fore.GREEN}-> Records Agent: Starting for {state.get('user_id', '?')}{Style.RESET_ALL}"
    )
    user_id = state["user_id"]
    user_query = state["user_query"]
    db = get_db()

    refund_requests = db.refund_requests.find({"user_id": user_id}, limit=3)
    tickets = db.tickets.find({"user_id": user_id}, limit=5)

    # Vector search for relevant orders
    try:
        vec = embed([user_query])[0]
        orders = db.orders.vector_search(vec, limit=1, filter_dict={"user_id": user_id})
    except Exception as e:
        print(
            f"  {Fore.YELLOW}Vector search failed, falling back: {e}{Style.RESET_ALL}"
        )
        orders = db.orders.find({"user_id": user_id}, limit=1)

    print(
        f"  {Fore.GREEN}Found {len(refund_requests)} requests, "
        f"{len(tickets)} tickets, {len(orders)} orders{Style.RESET_ALL}"
    )

    # LLM call — summarize customer intent from retrieved records
    records_context = json.dumps(
        {
            "refund_requests": refund_requests,
            "tickets": tickets,
            "orders": orders,
        },
        indent=2,
        default=str,
    )
    llm = create_chat_llm(temperature=0.1)
    response = await llm.ainvoke(
        [
            SystemMessage(content=_INTENT_PROMPT),
            HumanMessage(
                content=f"Customer query: {user_query}\n\nRetrieved records:\n{records_context}"
            ),
        ],
        config=config,
    )
    intent_summary = response.content.strip()
    print(f"  {Fore.GREEN}Intent: {intent_summary[:120]}...{Style.RESET_ALL}")

    return {
        "records": {
            "requests": refund_requests,
            "tickets": tickets,
            "orders": orders,
            "intent_summary": intent_summary,
        }
    }
