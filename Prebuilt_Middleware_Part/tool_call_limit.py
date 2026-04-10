from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from customize_agent_memory import search
from langchain.tools import tool


@tool
def database_tool(query: str) -> str:
    return f"Search for {query}"


agent = create_agent(
    model="gpt-4.1",
    tools=[search, database_tool],
    middleware=[
        # Global Limit
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        # Tool specific limit
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10, tool_name="search"),
    ],
)
