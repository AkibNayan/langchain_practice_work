from langchain.tools import tool


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search term to look for
        limit: Maximum number of records to return
    """
    return f"Found {limit} results for '{query}'"


# Custom tool name
@tool("web_search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}"


print(search.name)  # web_search


# Custom tool description
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems.",
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
