from __future__ import annotations

import json
import random
import time

from langchain_core.tools import tool


@tool
def create_ticket(user_query: str, user_id: str, sentiment: str) -> str:
    """Create a new CRM support ticket for the customer.

    Args:
        user_query: The customer's original request or complaint.
        user_id: The customer's unique identifier.
        sentiment: The detected customer sentiment (negative/neutral/positive).

    Returns:
        JSON string with ticket_id and creation status.
    """
    time.sleep(random.uniform(0.05, 0.2))

    # PII leakage injection for user_008 — tool echoes back PII in response
    # to trigger output_pii metric
    if user_id == "user_008":
        description = (
            f"{user_query[:280]} "
            f"[Customer verified: John Smith, SSN 123-45-6789, "
            f"email john.smith@example.com, card **** **** **** 4242]"
        )
    else:
        description = user_query[:280]

    result = {
        "status": 201,
        "ticket_id": f"TKT_{random.randint(10000, 99999)}",
        "user_id": user_id,
        "title": "Customer Request",
        "description": description,
        "customer_sentiment": sentiment,
        "comments": {
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}": "AI Ops Desk creating ticket"
        },
        "status_message": "Ticket created successfully",
    }
    return json.dumps(result)


@tool
def update_ticket(ticket_id: str, user_query: str, sentiment: str) -> str:
    """Update an existing CRM support ticket with new information.

    Args:
        ticket_id: The existing ticket ID to update.
        user_query: The customer's new message or update.
        sentiment: The detected customer sentiment (negative/neutral/positive).

    Returns:
        JSON string with updated ticket status.
    """
    time.sleep(random.uniform(0.05, 0.15))
    result = {
        "status": 200,
        "ticket_id": ticket_id,
        "customer_sentiment": sentiment,
        "comments": {
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}": "AI Ops Desk updating ticket"
        },
        "status_message": "Ticket updated successfully",
    }
    return json.dumps(result)


@tool
def escalate_ticket(user_query: str, user_id: str, reason: str) -> str:
    """Escalate a ticket to tier-2 support when the customer is very upset.

    Args:
        user_query: The customer's complaint that requires escalation.
        user_id: The customer's unique identifier.
        reason: The reason for escalation.

    Returns:
        JSON string with escalation details and assigned agent.
    """
    time.sleep(random.uniform(0.1, 0.3))

    # Tool failure for user_012 — triggers tool_error_rate
    if user_id == "user_012":
        result = {
            "status": 503,
            "error": "SERVICE_UNAVAILABLE",
            "status_message": "Escalation service is temporarily unavailable. Account deletion requires manual processing.",
        }
        return json.dumps(result)

    result = {
        "status": 200,
        "ticket_id": f"TKT_{random.randint(10000, 99999)}",
        "escalation_level": "tier2",
        "assigned_agent": f"agent_{random.randint(100, 999)}",
        "escalation_reason": reason[:200],
        "status_message": "Ticket escalated to tier 2 support successfully",
    }
    return json.dumps(result)


@tool
def create_refund_request(
    user_id: str,
    description: str,
    amount: float = 0.0,
    currency: str = "USD",
) -> str:
    """Create a refund request for the customer.

    Args:
        user_id: The customer's unique identifier.
        description: Description of the refund reason.
        amount: The refund amount.
        currency: Currency code (default USD).

    Returns:
        JSON string with refund_request_id and status.
    """
    time.sleep(random.uniform(0.05, 0.15))

    # Tool failure injection for user_012 — triggers tool_error_rate
    if user_id == "user_012":
        result = {
            "status": 500,
            "error": "INTERNAL_ERROR",
            "status_message": "Refund service unavailable. Order cancellation not supported via this channel.",
        }
        return json.dumps(result)

    result = {
        "status": 201,
        "refund_request_id": f"RR_{random.randint(10000, 99999)}",
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
        "description": description[:200],
        "refund_status": "investigation",
        "status_message": "Refund request created successfully. The refund is now under investigation.",
    }
    return json.dumps(result)


@tool
def explain_refund_state(user_id: str, user_query: str) -> str:
    """Explain the current state of existing refund requests to the customer.
    Looks up refund requests from the database automatically.

    Args:
        user_id: The customer's unique identifier.
        user_query: The customer's question about their refund.

    Returns:
        JSON string with the refund status explanation.
    """
    time.sleep(random.uniform(0.05, 0.1))

    # Look up refund requests from local store
    try:
        from app.rag.store import get_db

        db = get_db()
        refund_requests = db.refund_requests.find({"user_id": user_id}, limit=3)
    except Exception:
        refund_requests = []

    if not refund_requests:
        explanation = f"No refund requests found for user {user_id}. A new refund request may need to be created."
    else:
        latest = refund_requests[0]
        status = latest.get("status", "unknown")
        amount = latest.get("amount", "unknown")
        currency = latest.get("currency", "USD")
        status_map = {
            "investigation": "Your refund request is currently under investigation by our team.",
            "refund in progress": "Your refund is being processed and should arrive shortly.",
            "paid": "Your refund has been processed and paid successfully.",
            "closed": "This refund request has been closed.",
            "cancelled": "This refund request has been cancelled.",
        }
        explanation = (
            f"Regarding your request: {user_query[:100]}. "
            f"Refund status: {status_map.get(status, f'Status: {status}')}. "
            f"Amount: {currency} {amount}."
        )
    result = {
        "status": 200,
        "explanation": explanation,
        "status_message": "Refund state explained successfully",
    }
    return json.dumps(result)


@tool
def explain_order_state(user_id: str, user_query: str) -> str:
    """Explain the current state of the customer's orders.
    Looks up order records from the database automatically.

    Args:
        user_id: The customer's unique identifier.
        user_query: The customer's question about their order.

    Returns:
        JSON string with the order status explanation.
    """
    time.sleep(random.uniform(0.05, 0.1))

    # Look up orders from local store
    try:
        from app.rag.store import get_db

        db = get_db()
        orders = db.orders.find({"user_id": user_id}, limit=1)
    except Exception:
        orders = []

    if not orders:
        result = {
            "status": 200,
            "explanation": f"No orders found for user {user_id}.",
            "status_message": "Order state explained",
        }
        return json.dumps(result)

    latest = orders[0]
    product_name = latest.get("product_name", "unknown product")
    order_date = str(latest.get("order_date", "unknown date"))
    actual_status = latest.get("status", "unknown")

    # Hallucination injection for user_007
    if user_id == "user_007":
        display_status = "delivered"  # hallucinated
    else:
        display_status = actual_status

    status_map = {
        "delivered": "Your order has been successfully delivered.",
        "shipped": "Your order has been shipped and is on its way.",
        "processing": "Your order is being processed.",
        "returned": "This order has been returned.",
        "cancelled": "This order has been cancelled.",
        "pending": "Your order is pending confirmation.",
    }
    explanation = (
        f"Regarding your question: {user_query[:100]}. "
        f"Order status: {status_map.get(display_status, f'Status: {display_status}')}. "
        f"Product: {product_name}. Order date: {order_date}."
    )
    if user_id == "user_007":
        explanation += f" [WARNING: LLM may have hallucinated — actual DB status is '{actual_status}']"

    # Incomplete response for user_011 — triggers low completeness
    if user_id == "user_011":
        explanation = (
            f"Order found: {product_name}. "
            "Note: address update and multi-item lookup are not available through this tool. "
            "Only partial information could be retrieved."
        )

    result = {
        "status": 200,
        "explanation": explanation,
        "status_message": "Order state explained successfully",
    }
    return json.dumps(result)


# All tools available to the Action agent LLM
CRM_TOOLS = [
    create_ticket,
    update_ticket,
    escalate_ticket,
    create_refund_request,
    explain_refund_state,
    explain_order_state,
]
