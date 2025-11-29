"""
Custom tools for the MCP chatbot.
These are additional tools that can be used alongside MCP server tools.
"""

import math
from typing import List
from langchain_core.tools import StructuredTool


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: first number
        b: second number
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a: first number (minuend)
        b: second number (subtrahend)
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: first number
        b: second number
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: dividend
        b: divisor (must not be zero)
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.

    Args:
        base: the base number
        exponent: the exponent
    """
    return base**exponent


def square_root(x: float) -> float:
    """Calculate the square root of a number.

    Args:
        x: the number (must be non-negative)
    """
    if x < 0:
        raise ValueError("Square root of negative number is not allowed")
    return math.sqrt(x)


def modulo(a: float, b: float) -> float:
    """Calculate the remainder when a is divided by b.

    Args:
        a: dividend
        b: divisor (must not be zero)
    """
    if b == 0:
        raise ValueError("Modulo by zero is not allowed")
    return a % b


def absolute_value(x: float) -> float:
    """Calculate the absolute value of a number.

    Args:
        x: the number
    """
    return abs(x)


def get_multiply_tool() -> StructuredTool:
    """
    Get the multiply tool as a LangChain StructuredTool.

    Returns:
        StructuredTool: The multiply tool ready to be added to the tool list
    """
    return StructuredTool.from_function(multiply)


def get_all_calculation_tools() -> List[StructuredTool]:
    """
    Get all calculation tools as LangChain StructuredTools.

    Returns:
        List[StructuredTool]: List of all calculation tools ready to be added to the tool list
    """
    return [
        StructuredTool.from_function(add),
        StructuredTool.from_function(subtract),
        StructuredTool.from_function(multiply),
        StructuredTool.from_function(divide),
        StructuredTool.from_function(power),
        StructuredTool.from_function(square_root),
        StructuredTool.from_function(modulo),
        StructuredTool.from_function(absolute_value),
    ]
