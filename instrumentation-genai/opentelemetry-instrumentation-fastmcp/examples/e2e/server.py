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


if __name__ == "__main__":
    # Run the server (stdio mode for MCP)
    mcp.run()
