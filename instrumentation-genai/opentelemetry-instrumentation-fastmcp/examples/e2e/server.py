#!/usr/bin/env python3
"""
Simple MCP Calculator Server

A minimal MCP server with calculator tools for demonstrating
OpenTelemetry instrumentation in an end-to-end scenario.

This server is designed to be called by the client.py in this directory.
"""

from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Calculator Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract second number from first.

    Args:
        a: First number
        b: Second number

    Returns:
        Difference (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide first number by second.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient (a / b)

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def calculate_expression(expression: str) -> str:
    """Evaluate a simple mathematical expression.

    Supports +, -, *, / operators with numbers.
    Example: "2 + 3 * 4"

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result as a string with the original expression
    """
    # Simple and safe evaluation for basic math
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"

    try:
        # Use eval with restricted builtins for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@mcp.tool()
def get_session_info() -> str:
    """Return the current session context as seen by the server.

    This tool validates that session context (gen_ai.conversation.id, user.id,
    customer.id) is properly propagated from the client via OTel Baggage.

    Returns:
        JSON string with the session context values visible on the server side.
    """
    import json

    result: dict[str, str | None] = {
        "gen_ai.conversation.id": None,
        "user.id": None,
        "customer.id": None,
    }

    # Try reading from OTel Baggage (cross-service propagation)
    try:
        from opentelemetry import baggage

        result["gen_ai.conversation.id"] = baggage.get_baggage("gen_ai.conversation.id")
        result["user.id"] = baggage.get_baggage("user.id")
        result["customer.id"] = baggage.get_baggage("customer.id")
    except ImportError:
        pass

    # Try reading from GenAI session context (contextvar propagation)
    try:
        from opentelemetry.util.genai.handler import get_session_context

        ctx = get_session_context()
        if ctx.session_id and not result["gen_ai.conversation.id"]:
            result["gen_ai.conversation.id"] = ctx.session_id
        if ctx.user_id and not result["user.id"]:
            result["user.id"] = ctx.user_id
        if ctx.customer_id and not result["customer.id"]:
            result["customer.id"] = ctx.customer_id
    except ImportError:
        pass

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    # Run the server (stdio mode for MCP)
    mcp.run()
