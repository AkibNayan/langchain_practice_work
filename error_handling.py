from langchain.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


tools = [search, calculator]


# Default: Catch invocation errors, re-raise execution errors
tool_node = ToolNode(tools)

# Catch all errors and return error message to LLM
tool_node = ToolNode(tools, handle_tool_errors=True)

# Custom error message
tool_node = ToolNode(tools, handle_tool_errors="Something went wrong")


# Custom error handler
def handle_error(e: ValueError) -> str:
    return f"Invalid input: {e}"


tool_node = ToolNode(tools, handle_tool_errors=handle_error)

# Only catch specific exception types
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
