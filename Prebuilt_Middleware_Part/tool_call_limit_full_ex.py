from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from customize_agent_memory import search


@tool
def database_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def strict_limiter(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
search_limiter = ToolCallLimitMiddleware(
    tool_name="search", thread_limit=5, run_limit=3
)
database_limiter = ToolCallLimitMiddleware(tool_name="database_tool", thread_limit=10)
strict_limiter = ToolCallLimitMiddleware(
    tool_name="scrape_webpage", run_limit=2, exit_behavior="error"
)

agent = create_agent(
    model="gpt-4.1",
    tools=[search, database_tool, strict_limiter],
    middleware=[global_limiter, search_limiter, database_limiter, strict_limiter],
)
