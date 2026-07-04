"""Arithmetic calculator tool.

The model routes numeric operations through this tool instead of
next-token-guessing them. `SCHEMA` is the Anthropic tool definition;
`handle` executes one call and returns the result as a string (the
Messages API expects `tool_result` content as text).
"""

import operator
from typing import Any

from gene.tool import Tool

_OPS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}

SCHEMA: dict[str, Any] = {
    "name": "calculator",
    "description": (
        "Perform exact arithmetic on two numbers. Use this whenever the "
        "user asks for a numeric result — do not compute mentally."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPS.keys()),
                "description": "Which arithmetic operation to perform.",
            },
            "a": {"type": "number", "description": "First operand."},
            "b": {"type": "number", "description": "Second operand."},
        },
        "required": ["operation", "a", "b"],
    },
}


def handle(inputs: dict[str, Any]) -> str:
    op = inputs["operation"]
    a = inputs["a"]
    b = inputs["b"]
    if op == "divide" and b == 0:
        return "error: division by zero"
    return str(_OPS[op](a, b))


CALCULATOR = Tool(schema=SCHEMA, handler=handle)
