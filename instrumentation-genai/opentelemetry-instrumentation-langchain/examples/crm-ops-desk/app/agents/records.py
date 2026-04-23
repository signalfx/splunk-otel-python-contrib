from __future__ import annotations


from colorama import Fore, Style

from app.rag.store import embed, get_db


async def records_node(state: dict) -> dict:
    """Records Agent — fetch refund requests, tickets, and orders."""
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
    return {
        "records": {
            "requests": refund_requests,
            "tickets": tickets,
            "orders": orders,
        }
    }
