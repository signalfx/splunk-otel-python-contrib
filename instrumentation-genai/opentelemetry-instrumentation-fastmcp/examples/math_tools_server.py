#!/usr/bin/env python3
"""
Simple Math Tools MCP Server

A minimal MCP server with simple mathematical tools
for demonstrating OpenTelemetry instrumentation.

Usage:
    python math_tools_server.py
"""

import math
from typing import List

from fastmcp import FastMCP

# Import and apply instrumentation
from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

# Configure OpenTelemetry with console exporter for demo
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Apply FastMCP instrumentation
FastMCPInstrumentor().instrument()

# Initialize the MCP server
server = FastMCP("math-tools")


@server.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@server.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract second number from first.

    Args:
        a: First number
        b: Second number

    Returns:
        Difference (a - b)
    """
    return a - b


@server.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@server.tool()
def divide(a: float, b: float) -> float:
    """
    Divide first number by second.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Quotient (a / b)

    Raises:
        ValueError: If divisor is zero
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


@server.tool()
def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n (n!)

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)


@server.tool()
def power(base: float, exponent: float) -> float:
    """
    Raise a number to a power.

    Args:
        base: Base number
        exponent: Exponent

    Returns:
        base raised to the power of exponent
    """
    return math.pow(base, exponent)


@server.tool()
def sqrt(n: float) -> float:
    """
    Calculate the square root of a number.

    Args:
        n: Non-negative number

    Returns:
        Square root of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Square root is not defined for negative numbers")
    return math.sqrt(n)


@server.tool()
def average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        Average (mean) of the numbers

    Raises:
        ValueError: If list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)


@server.tool()
def fibonacci(n: int) -> List[int]:
    """
    Generate the first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of first n Fibonacci numbers

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Number of Fibonacci terms must be non-negative")
    if n == 0:
        return []
    if n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


@server.tool()
def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def main():
    """Main entry point for the server."""
    import os
    import sys

    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    print(f"Starting Math Tools MCP Server on {transport}", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - add, subtract, multiply, divide", file=sys.stderr)
    print("  - power, sqrt, factorial", file=sys.stderr)
    print("  - average, fibonacci, is_prime", file=sys.stderr)

    server.run(transport=transport)


if __name__ == "__main__":
    main()
